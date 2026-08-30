"""
face_engine.py
==============
Webcam capture, face encoding, face matching, and frame annotation.

Architecture
------------
WebcamStream   — background thread that continuously reads frames from the
                 camera and stores the latest one in a thread-safe buffer.

FaceRecognitionEngine — uses the `face_recognition` library (dlib-backed) to:
    1.  Detect face locations in a frame.
    2.  Compute 128-d embedding vectors (face encodings) for each detected face.
    3.  Compare unknown encodings against the enrolled-employee database to
        find the best match, returning a confidence score.

annotate_frame — pure function that draws bounding boxes, name labels,
                 confidence %, timestamps, and camera metadata onto a frame
                 using OpenCV drawing primitives.

NOTE ON DLIB PERFORMANCE
------------------------
face_recognition.face_locations() uses HOG by default which is CPU-only but
fast enough for 640×480 at ~10 fps on a modern laptop.  CNN-based detection
(`model="cnn"`) is more accurate but much slower without a GPU.  This code
uses HOG (the default) for real-time performance.

To reduce CPU load, recognition is run every N frames rather than every frame
(controlled by RECOGNITION_INTERVAL in the engine).
"""

from __future__ import annotations

import threading
import time
import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

# face_recognition and dlib are imported with a graceful fallback so the app
# can start and show a warning if dlib is not yet installed.
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning(
        "face_recognition / dlib not installed. Face matching is disabled. "
        "Install dlib + face_recognition to enable recognition features."
    )

logger = logging.getLogger(__name__)

# Global lock to prevent dlib from segfaulting when called from multiple threads
FACE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Colour palette — BGR tuples (OpenCV uses BGR, not RGB)
# ---------------------------------------------------------------------------
CLR_ACCENT   = (244, 108, 108)   # #6C6CF4 in BGR → indigo
CLR_SUCCESS  = (132, 189,  61)   # #3DBD84 in BGR → green
CLR_DANGER   = ( 74, 103, 229)   # #E5674A in BGR → coral
CLR_TEXT     = (239, 237, 237)   # #EDEDEF in BGR → near-white
CLR_MUTED    = ( 99,  92,  92)   # #5C5C63 in BGR → muted
CLR_BG_DARK  = ( 11,  10,  10)   # #0A0A0B in BGR → almost black


# ===========================================================================
# WebcamStream — continuous background capture thread
# ===========================================================================

class WebcamStream:
    """
    Opens a camera once and reads frames in a tight background loop.

    By separating capture from the Flask streaming route we avoid dropped
    frames caused by Flask's GIL / request-handling latency, and we ensure
    the latest frame is always immediately available to both the streaming
    route and the recognition engine.
    """

    def __init__(self, src: int = 0) -> None:
        self._src = src
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "WebcamStream":
        """Open the camera and launch the capture thread."""
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="webcam-capture")
        self._thread.start()
        # Give the camera a moment to initialise before the first read
        time.sleep(0.5)
        return self

    def read(self) -> Optional[np.ndarray]:
        """
        Return a copy of the latest captured frame, or None if no frame
        is available yet (camera not yet opened or unavailable).
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        """Signal the capture thread to exit and release the camera."""
        self._running = False
        if self._cap is not None:
            self._cap.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_camera(self) -> bool:
        """Attempt to open the camera device. Returns True on success."""
        cap = cv2.VideoCapture(self._src)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise latency
        self._cap = cap
        self.available = True
        logger.info("Webcam opened on device %d", self._src)
        return True

    def _capture_loop(self) -> None:
        """
        Main capture loop — runs in a daemon thread.
        Retries camera open every 3 seconds on failure.
        """
        while self._running:
            # Attempt to open the camera if not already open
            if self._cap is None or not self._cap.isOpened():
                self.available = False
                logger.warning("Webcam not available — retrying in 3 s …")
                time.sleep(3)
                if not self._open_camera():
                    continue

            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                # Camera returned a bad frame — release and retry
                self._cap.release()
                self._cap = None
                self.available = False
                time.sleep(1)


# ===========================================================================
# FaceRecognitionEngine — encoding, matching, annotation
# ===========================================================================

class FaceRecognitionEngine:
    """
    Manages the face recognition lifecycle:
    - Runs recognition every RECOGNITION_INTERVAL frames to limit CPU usage.
    - Caches the last detected results so the annotator can draw smoothly
      on every frame even when recognition only runs occasionally.
    - Provides encode_face() for enrollment and match_face() for check-in.
    """

    #: Run the heavy recognition pass every N frames (trade accuracy for CPU)
    RECOGNITION_INTERVAL = 5

    #: Maximum Euclidean distance to accept a face as a match.
    #: Lower = stricter.  face_recognition default is 0.6; we use 0.50 to
    #: reduce false positives in an attendance (security-sensitive) context.
    MATCH_THRESHOLD = 0.50

    def __init__(self, webcam: WebcamStream) -> None:
        self._webcam = webcam
        self._frame_counter = 0

        # Thread-safe cache of the latest recognition results
        # Each item: {'name': str, 'status': str, 'confidence': float, 'bbox': (t,r,b,l)}
        self._detections: list[dict] = []
        self._detections_lock = threading.Lock()

        # Latest confirmed check-in event for the toast strip
        self._recent_event: Optional[dict] = None
        self._recent_event_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Frame acquisition helpers
    # ------------------------------------------------------------------

    def get_annotated_frame(self, known_employees: list[tuple]) -> Optional[np.ndarray]:
        """
        Grab the latest webcam frame, run recognition (if due), annotate it,
        and return the JPEG-encoded bytes for MJPEG streaming.

        Args:
            known_employees : list of (employee_id, name, encodings_list) from DB

        Returns:
            Raw BGR numpy frame with annotations drawn on it, or a placeholder
            frame if the camera is unavailable.
        """
        frame = self._webcam.read()

        if frame is None:
            return self._make_no_camera_frame()

        self._frame_counter += 1

        # Run the expensive recognition pass every RECOGNITION_INTERVAL frames
        if self._frame_counter % self.RECOGNITION_INTERVAL == 0 and FACE_RECOGNITION_AVAILABLE:
            detections = self._run_recognition(frame, known_employees)
            with self._detections_lock:
                self._detections = detections

        with self._detections_lock:
            current_detections = list(self._detections)

        annotate_frame(frame, current_detections)
        return frame

    def get_current_detections(self) -> list[dict]:
        """Return the most recently computed detection results (thread-safe)."""
        with self._detections_lock:
            return list(self._detections)

    def get_recent_event(self) -> Optional[dict]:
        """Return the most recently confirmed check-in event dict."""
        with self._recent_event_lock:
            return self._recent_event

    def set_recent_event(self, event: dict) -> None:
        """Store a new check-in event (called from the recognition loop)."""
        with self._recent_event_lock:
            self._recent_event = event

    def capture_current_frame(self) -> Optional[np.ndarray]:
        """Return a raw (un-annotated) frame suitable for enrollment encoding."""
        return self._webcam.read()

    # ------------------------------------------------------------------
    # Face encoding — used during employee enrollment
    # ------------------------------------------------------------------

    @staticmethod
    def encode_face(image: np.ndarray) -> list:
        """
        Extract 128-d face embedding vectors from an image.

        Uses face_recognition.face_encodings() which internally:
        1.  Detects face bounding boxes with HOG + SVM (dlib)
        2.  Applies a 68-point face landmark model
        3.  Aligns and normalises the face chip
        4.  Passes the chip through a ResNet-based metric-learning network
            to produce a 128-d vector where similar faces cluster together

        Args:
            image : BGR numpy array (OpenCV format)

        Returns:
            List of 128-d numpy arrays — one per detected face.
            Empty list if no face is found in the image.
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return []

        # face_recognition expects RGB; OpenCV gives us BGR
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with FACE_LOCK:
            # Detect face locations first (HOG model — CPU friendly)
            locations = face_recognition.face_locations(rgb, model="hog")
            if not locations:
                return []

            # Compute 128-d encodings for each detected face
            encodings = face_recognition.face_encodings(rgb, locations)
        return encodings

    # ------------------------------------------------------------------
    # Face matching — used during live check-in
    # ------------------------------------------------------------------

    @staticmethod
    def match_face(unknown_encoding: np.ndarray,
                   known_employees: list[tuple],
                   threshold: float = 0.50) -> tuple[Optional[int], Optional[str], float]:
        """
        Compare an unknown face encoding against all enrolled employees and
        return the best match above the confidence threshold.

        How Euclidean distance → confidence works
        -----------------------------------------
        face_recognition.face_distance() computes the L2 (Euclidean) distance
        between two 128-d vectors.  The distance represents how "different"
        two faces are in the embedding space:
            distance = 0.0   → identical (perfect match)
            distance = 0.4   → very likely the same person
            distance = 0.6   → borderline
            distance ≥ 0.6   → likely different people (default library threshold)

        We convert distance to a percentage confidence for display:
            confidence = (1 - distance / threshold) × 100
        This maps: distance=0 → 100%, distance=threshold → 0%.

        We use threshold=0.50 (stricter than the library default of 0.60) to
        reduce false positives — in an attendance system a wrong match is
        worse than a missed match.

        Args:
            unknown_encoding  : 128-d numpy array of the face to identify
            known_employees   : list of (employee_id, name, list[np.ndarray])
                                as returned by database.get_enrolled_employees()
            threshold         : max distance to count as a match

        Returns:
            (employee_id, employee_name, confidence_percent)
            Returns (None, None, 0.0) if no match above threshold is found.
        """
        if not FACE_RECOGNITION_AVAILABLE or not known_employees:
            return None, None, 0.0

        best_id         = None
        best_name       = None
        best_distance   = float("inf")

        for emp_id, emp_name, encodings in known_employees:
            if not encodings:
                continue  # employee enrolled but no encodings stored yet

            # Compute distances between the unknown face and ALL stored
            # encodings for this employee (multiple reference photos per person
            # improve robustness across lighting / pose variation)
            distances = face_recognition.face_distance(encodings, unknown_encoding)

            # Use the MINIMUM distance — the best-matching reference photo
            min_dist = float(np.min(distances))

            if min_dist < best_distance:
                best_distance = min_dist
                best_id       = emp_id
                best_name     = emp_name

        if best_distance <= threshold:
            # Linear mapping: distance=0 → 100%, distance=threshold → 0%
            confidence = (1.0 - best_distance / threshold) * 100.0
            confidence = round(min(confidence, 100.0), 1)
            return best_id, best_name, confidence

        # No employee matched within the threshold
        return None, None, 0.0

    # ------------------------------------------------------------------
    # Internal recognition pass
    # ------------------------------------------------------------------

    def _run_recognition(self, frame: np.ndarray,
                         known_employees: list[tuple]) -> list[dict]:
        """
        Run face detection and matching on a single frame.
        Returns a list of detection dicts for annotate_frame().
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Downsample for faster detection, then scale bounding boxes back up
        small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
        
        with FACE_LOCK:
            locations = face_recognition.face_locations(small, model="hog")
            if not locations:
                return []

            # Scale locations back to original resolution
            locations_full = [(t*2, r*2, b*2, l*2) for (t, r, b, l) in locations]

            encodings = face_recognition.face_encodings(rgb, locations_full)

        detections = []
        for encoding, bbox in zip(encodings, locations_full):
            emp_id, emp_name, confidence = self.match_face(
                encoding, known_employees, threshold=self.MATCH_THRESHOLD
            )
            detections.append({
                "employee_id": emp_id,
                "name":        emp_name or "Unknown",
                "confidence":  confidence,
                "bbox":        bbox,      # (top, right, bottom, left)
                "matched":     emp_id is not None,
            })

        return detections

    # ------------------------------------------------------------------
    # Placeholder frame for missing / unavailable camera
    # ------------------------------------------------------------------

    @staticmethod
    def _make_no_camera_frame() -> np.ndarray:
        """
        Generate a styled 'camera unavailable' placeholder frame.
        This is what gets streamed when cv2.VideoCapture cannot open the device.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = CLR_BG_DARK

        # Camera icon — simple cross-hair style placeholder
        cx, cy = 320, 210
        cv2.circle(frame, (cx, cy), 55, (50, 50, 55), 2)
        cv2.line(frame, (cx - 65, cy), (cx + 65, cy), (50, 50, 55), 1)
        cv2.line(frame, (cx, cy - 65), (cx, cy + 65), (50, 50, 55), 1)

        _draw_text(frame, "CAMERA UNAVAILABLE", (640 // 2, 300),
                   font_scale=0.65, color=CLR_MUTED, centered=True)
        _draw_text(frame, "Check that your webcam is connected and not", (640 // 2, 328),
                   font_scale=0.38, color=CLR_MUTED, centered=True)
        _draw_text(frame, "in use by another application.", (640 // 2, 348),
                   font_scale=0.38, color=CLR_MUTED, centered=True)

        _draw_overlay_metadata(frame)
        return frame


# ===========================================================================
# Frame annotation — pure function, no side effects
# ===========================================================================

def annotate_frame(frame: np.ndarray, detections: list[dict]) -> None:
    """
    Draw bounding boxes, name labels, confidence scores, timestamp, and
    camera metadata onto the frame in-place (modifies frame directly).

    Args:
        frame      : BGR numpy array — the live webcam frame
        detections : list of detection dicts from FaceRecognitionEngine._run_recognition()
                     Each dict: {name, confidence, bbox:(t,r,b,l), matched:bool}
    """
    for det in detections:
        top, right, bottom, left = det["bbox"]
        matched    = det["matched"]
        name       = det["name"]
        confidence = det["confidence"]

        # Choose colour based on match status
        box_color  = CLR_SUCCESS if matched else CLR_DANGER
        label_bg   = (34, 100, 45) if matched else (30, 45, 100)   # dim tint

        # ------------------------------------------------------------------
        # Bounding box — double-line effect for a clean professional look
        # ------------------------------------------------------------------
        thickness = 2
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness)

        # Corner accents (4 corners, each with a small L-shaped mark)
        corner_len = 14
        corner_w   = 3
        _draw_corner_accents(frame, left, top, right, bottom,
                             corner_len, corner_w, box_color)

        # ------------------------------------------------------------------
        # Label background pill above the bounding box
        # ------------------------------------------------------------------
        label_line1 = name
        label_line2 = f"{confidence:.1f}% match" if matched else "Not recognised"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        scale1     = 0.52
        scale2     = 0.38
        thickness1 = 1

        (w1, h1), _ = cv2.getTextSize(label_line1, font, scale1, thickness1)
        (w2, h2), _ = cv2.getTextSize(label_line2, font, scale2, 1)

        pill_w  = max(w1, w2) + 20
        pill_h  = h1 + h2 + 20
        pill_x1 = left
        pill_y1 = max(0, top - pill_h - 6)
        pill_x2 = pill_x1 + pill_w
        pill_y2 = pill_y1 + pill_h

        # Filled rect with slight alpha-blend style (pure OpenCV — addWeighted trick)
        overlay = frame.copy()
        cv2.rectangle(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), label_bg, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Border on pill matching box colour
        cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), box_color, 1)

        # Text lines
        text_x = pill_x1 + 10
        cv2.putText(frame, label_line1, (text_x, pill_y1 + h1 + 7),
                    font, scale1, CLR_TEXT, thickness1, cv2.LINE_AA)
        cv2.putText(frame, label_line2, (text_x, pill_y1 + h1 + h2 + 14),
                    font, scale2, box_color, 1, cv2.LINE_AA)

    _draw_overlay_metadata(frame)


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _draw_overlay_metadata(frame: np.ndarray) -> None:
    """Draw the timestamp (top-right) and CAM_01 label (bottom-left) onto the frame."""
    h, w = frame.shape[:2]
    now_str  = datetime.now().strftime("%H:%M:%S")
    date_str = datetime.now().strftime("%Y-%m-%d")
    cam_str  = "CAM_01 · Local"

    # Top-right — timestamp
    _draw_text(frame, now_str,  (w - 10, 20), font_scale=0.45, color=CLR_MUTED, right_align=True)
    _draw_text(frame, date_str, (w - 10, 36), font_scale=0.38, color=CLR_MUTED, right_align=True)

    # Bottom-left — camera ID
    _draw_text(frame, cam_str, (10, h - 10), font_scale=0.40, color=CLR_MUTED)


def _draw_text(frame: np.ndarray, text: str, pos: tuple,
               font_scale: float = 0.5, color: tuple = CLR_TEXT,
               centered: bool = False, right_align: bool = False,
               thickness: int = 1) -> None:
    """Convenience wrapper for cv2.putText with alignment options."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    if centered:
        x -= tw // 2
    elif right_align:
        x -= tw
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_corner_accents(frame: np.ndarray,
                         left: int, top: int, right: int, bottom: int,
                         length: int, thickness: int, color: tuple) -> None:
    """Draw L-shaped corner accents at all four corners of a bounding box."""
    # Top-left
    cv2.line(frame, (left,  top),           (left + length,  top),          color, thickness)
    cv2.line(frame, (left,  top),           (left,           top + length),  color, thickness)
    # Top-right
    cv2.line(frame, (right, top),           (right - length, top),          color, thickness)
    cv2.line(frame, (right, top),           (right,          top + length),  color, thickness)
    # Bottom-left
    cv2.line(frame, (left,  bottom),        (left + length,  bottom),        color, thickness)
    cv2.line(frame, (left,  bottom),        (left,           bottom - length), color, thickness)
    # Bottom-right
    cv2.line(frame, (right, bottom),        (right - length, bottom),        color, thickness)
    cv2.line(frame, (right, bottom),        (right,          bottom - length), color, thickness)
