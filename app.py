"""
app.py
======
Main Flask application — routes, background recognition thread, and MJPEG streaming.

Startup sequence
----------------
1.  init_db()         — create/migrate SQLite schema, seed sample data
2.  WebcamStream.start() — daemon thread begins capturing frames
3.  _start_recognition_loop() — daemon thread runs face recognition every N frames
4.  app.run()         — Flask begins serving HTTP

Key design choices
------------------
- The MJPEG stream (GET /video_feed) is a generator that yields annotated JPEG
  frames using Flask's Response(stream_with_context(generate()), ...) pattern.
  The <img src="/video_feed"> tag in the browser keeps the connection open
  and renders each JPEG as it arrives — no WebSocket, no WebRTC needed.

- The recognition loop writes directly to the database via log_checkin() when
  a confident match is found and the employee hasn't checked in today.  Flask
  request handlers never block waiting for recognition results.

- Enrollment photos are accumulated in-process (PENDING_ENCODINGS list) and
  saved atomically on /api/enroll/submit.  Since this is a single-machine
  local app there is no concurrency concern here.
"""

from __future__ import annotations

import io
import csv
import base64
import logging
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, render_template,
                   request, send_file, stream_with_context)

import database as db
from attendance_logic import evaluate_check_in, format_status_label
from face_engine import FaceRecognitionEngine, WebcamStream, FACE_RECOGNITION_AVAILABLE

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "nti-attendance-local-secret-2024"


# ---------------------------------------------------------------------------
# Global singletons — initialised in main()
# ---------------------------------------------------------------------------
webcam_stream:   WebcamStream            | None = None
face_engine:     FaceRecognitionEngine   | None = None

# In-memory store for enrollment photos collected before submit
PENDING_ENCODINGS: list = []
PENDING_LOCK = threading.Lock()


# ===========================================================================
# Background recognition loop
# ===========================================================================

def _recognition_loop() -> None:
    """
    Daemon thread: repeatedly grabs frames, runs face recognition, and logs
    confirmed check-ins to the database.

    The loop sleeps for 0.1 s between iterations (~10 fps effective rate on
    the recognition path; the webcam capture thread runs faster independently).
    """
    logger.info("Recognition loop started")
    while True:
        try:
            if face_engine is None or webcam_stream is None:
                time.sleep(1)
                continue

            # Load enrolled employees fresh from DB every ~30 s to pick up
            # new enrollments without a server restart
            pass   # handled via frame counter inside face_engine

            known_employees = db.get_enrolled_employees()

            # get_annotated_frame() internally throttles recognition to every
            # RECOGNITION_INTERVAL frames and caches results
            frame = face_engine.get_annotated_frame(known_employees)

            # Check current detections for confident matches to log
            detections = face_engine.get_current_detections()
            _handle_detections(detections)

        except Exception as exc:
            logger.error("Recognition loop error: %s", exc, exc_info=True)

        time.sleep(0.05)   # ~20 fps loop rate; recognition itself is throttled


def _handle_detections(detections: list[dict]) -> None:
    """
    For each detection that is a confident face match and the employee has not
    yet checked in today, log the attendance record and update the recent event.
    """
    if not detections:
        return

    policy = db.get_policy()
    now    = datetime.now()

    for det in detections:
        emp_id = det.get("employee_id")
        if emp_id is None:
            continue   # unknown face — nothing to log

        # Avoid duplicate check-ins: is_checked_in_today queries the DB
        if db.is_checked_in_today(emp_id):
            continue

        # Determine status and deduction based on the active policy
        status, deduction = evaluate_check_in(now, policy)

        # Persist the attendance record (INSERT OR IGNORE handles race conditions)
        record_id = db.log_checkin(
            employee_id = emp_id,
            check_in_time = now,
            status      = status,
            deduction   = deduction,
            confidence  = det["confidence"],
        )

        if record_id:
            emp = db.get_employee_by_id(emp_id)
            label = format_status_label(status, now.isoformat(), policy)
            event = {
                "employee_id": emp_id,
                "name":        emp["name"] if emp else det["name"],
                "department":  emp["department"] if emp else "—",
                "status":      status,
                "status_label": label,
                "check_in_time": now.strftime("%H:%M:%S"),
                "deduction":   deduction,
                "confidence":  det["confidence"],
            }
            face_engine.set_recent_event(event)
            logger.info(
                "Check-in logged: %s | %s | deduction=$%.2f | confidence=%.1f%%",
                event["name"], status, deduction, det["confidence"],
            )


# ===========================================================================
# MJPEG streaming
# ===========================================================================

def _generate_frames():
    """
    Generator that yields annotated JPEG frames as a multipart MJPEG stream.
    Flask's Response wraps this in a multipart/x-mixed-replace content type,
    which the browser renders as a continuously updating <img> tag.
    """
    known_cache      = []
    cache_last_fetch = 0.0
    CACHE_TTL        = 5.0   # refresh enrolled-employees cache every 5 s

    while True:
        now = time.time()
        if now - cache_last_fetch > CACHE_TTL:
            known_cache      = db.get_enrolled_employees()
            cache_last_fetch = now

        if face_engine is not None:
            frame = face_engine.get_annotated_frame(known_cache)
        elif webcam_stream is not None:
            frame = webcam_stream.read()
        else:
            frame = None

        if frame is None:
            frame = FaceRecognitionEngine._make_no_camera_frame()

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            time.sleep(0.033)
            continue

        jpg_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
        )

        time.sleep(0.033)   # ~30 fps cap


# ===========================================================================
# Routes — Dashboard
# ===========================================================================

@app.route("/")
def dashboard():
    """Main dashboard page."""
    return render_template("dashboard.html", active_page="dashboard")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint — embed as <img src='/video_feed'>."""
    return Response(
        stream_with_context(_generate_frames()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stats")
def api_stats():
    """JSON: today's aggregate statistics for the stat cards."""
    try:
        stats = db.get_today_stats()
        return jsonify({"ok": True, "data": stats})
    except Exception as exc:
        logger.error("api_stats error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/recent_checkin")
def api_recent_checkin():
    """JSON: most recent check-in event (from recognition loop or DB)."""
    try:
        # Prefer the in-memory recent event (more up-to-date, has confidence)
        if face_engine:
            event = face_engine.get_recent_event()
            if event:
                return jsonify({"ok": True, "data": event})

        # Fallback: latest DB record
        record = db.get_recent_checkin()
        if record:
            policy = db.get_policy()
            data = dict(record)
            data["status_label"] = format_status_label(
                data["status"], data["check_in_time"], policy
            )
            return jsonify({"ok": True, "data": data})

        return jsonify({"ok": True, "data": None})
    except Exception as exc:
        logger.error("api_recent_checkin error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/today_log")
def api_today_log():
    """JSON: full today attendance table for auto-refresh."""
    try:
        records = db.get_today_records()
        policy  = db.get_policy()
        rows = []
        for r in records:
            rows.append({
                "id":              r["id"],
                "employee_id":     r["employee_id"],
                "name":            r["name"],
                "department":      r["department"],
                "check_in_time":   r["check_in_time"][11:19],   # HH:MM:SS
                "status":          r["status"],
                "status_label":    format_status_label(r["status"], r["check_in_time"], policy),
                "deduction_amount": r["deduction_amount"],
                "confidence":      r["recognition_confidence"],
            })
        return jsonify({"ok": True, "data": rows})
    except Exception as exc:
        logger.error("api_today_log error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


# ===========================================================================
# Routes — Enrollment
# ===========================================================================

@app.route("/enroll")
def enroll():
    """Enrollment page — register a new employee with face capture."""
    employees = db.get_all_employees()
    return render_template("enroll.html", active_page="enroll",
                           employees=employees,
                           face_recognition_available=FACE_RECOGNITION_AVAILABLE)


@app.route("/api/enroll/capture", methods=["POST"])
def api_enroll_capture():
    """
    Capture a single frame from the webcam, extract face encoding,
    and add it to the in-process pending encodings list.
    Returns: {ok, count, image_b64} — count is how many captures so far.
    """
    global PENDING_ENCODINGS

    if not FACE_RECOGNITION_AVAILABLE:
        return jsonify({"ok": False, "error": "face_recognition library not installed."}), 503

    if webcam_stream is None or not webcam_stream.available:
        return jsonify({"ok": False, "error": "Webcam is not available."}), 503

    frame = webcam_stream.read()
    if frame is None:
        return jsonify({"ok": False, "error": "Could not read a frame from the webcam."}), 503

    encodings = FaceRecognitionEngine.encode_face(frame)
    if not encodings:
        # Return a snapshot image even if no face was found, for UX feedback
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        return jsonify({
            "ok":     False,
            "error":  "No face detected in the frame. Please adjust position and try again.",
            "image_b64": img_b64,
        })

    # Use only the first detected face per capture
    encoding = encodings[0]

    # Annotate the snapshot with a green bounding box for feedback
    import face_recognition as fr
    from face_engine import FACE_LOCK
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with FACE_LOCK:
        locs  = fr.face_locations(rgb, model="hog")
        
    annotated = frame.copy()
    if locs:
        top, right, bottom, left = locs[0]
        cv2.rectangle(annotated, (left, top), (right, bottom), (132, 189, 61), 2)

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    with PENDING_LOCK:
        PENDING_ENCODINGS.append(encoding)
        count = len(PENDING_ENCODINGS)

    return jsonify({"ok": True, "count": count, "image_b64": img_b64})


@app.route("/api/enroll/clear", methods=["POST"])
def api_enroll_clear():
    """Clear pending encodings (e.g. when the user starts a new enrollment)."""
    global PENDING_ENCODINGS
    with PENDING_LOCK:
        PENDING_ENCODINGS.clear()
    return jsonify({"ok": True})


@app.route("/api/enroll/submit", methods=["POST"])
def api_enroll_submit():
    """
    Save a new employee with the accumulated pending face encodings.
    Required fields: name, department, base_salary.
    Minimum 3 face captures required.
    """
    global PENDING_ENCODINGS
    data   = request.get_json(force=True)
    name   = (data.get("name") or "").strip()
    dept   = (data.get("department") or "").strip()
    salary = data.get("base_salary")

    # Validate inputs
    if not name:
        return jsonify({"ok": False, "error": "Employee name is required."}), 400
    if not dept:
        return jsonify({"ok": False, "error": "Department is required."}), 400
    try:
        salary = float(salary)
        if salary <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Base salary must be a positive number."}), 400

    with PENDING_LOCK:
        captures = list(PENDING_ENCODINGS)

    if FACE_RECOGNITION_AVAILABLE and len(captures) < 3:
        return jsonify({
            "ok":    False,
            "error": f"Need at least 3 face captures ({len(captures)} recorded). "
                     "Please capture more photos.",
        }), 400

    # Guard against duplicate enrollment by name
    if db.check_duplicate_enrollment(name):
        return jsonify({
            "ok":    False,
            "error": f'An employee named "{name}" is already enrolled. '
                     "Use a unique full name.",
        }), 409

    # Persist the new employee
    try:
        emp_id = db.create_employee(
            name          = name,
            department    = dept,
            base_salary   = salary,
            face_encodings = captures if captures else None,
        )
    except Exception as exc:
        logger.error("Enrollment save error: %s", exc)
        return jsonify({"ok": False, "error": "Database error — could not save employee."}), 500

    # Clear pending list after successful save
    with PENDING_LOCK:
        PENDING_ENCODINGS.clear()

    logger.info("Enrolled new employee: %s (id=%d, captures=%d)", name, emp_id, len(captures))
    return jsonify({
        "ok":          True,
        "employee_id": emp_id,
        "message":     f'{name} enrolled successfully with {len(captures)} face captures.',
    })


@app.route("/api/enroll/delete/<int:emp_id>", methods=["DELETE"])
def api_enroll_delete(emp_id: int):
    """Delete an employee and their attendance history."""
    emp = db.get_employee_by_id(emp_id)
    if not emp:
        return jsonify({"ok": False, "error": "Employee not found."}), 404
    db.delete_employee(emp_id)
    return jsonify({"ok": True, "message": f'{emp["name"]} deleted.'})


# ===========================================================================
# Routes — Attendance Policy
# ===========================================================================

@app.route("/policy")
def policy():
    """Attendance policy configuration page."""
    current_policy = db.get_policy()
    return render_template("policy.html", active_page="policy", policy=current_policy)


@app.route("/api/policy/save", methods=["POST"])
def api_policy_save():
    """Save updated attendance policy."""
    data = request.get_json(force=True)
    try:
        db.save_policy(
            start_time                  = data.get("start_time", "09:00"),
            grace_period_minutes        = int(data.get("grace_period_minutes", 15)),
            deduction_type              = data.get("deduction_type", "flat_tiers"),
            per_minute_rate             = float(data.get("per_minute_rate", 2.0)),
            late_flat                   = float(data.get("late_flat", 25.0)),
            very_late_flat              = float(data.get("very_late_flat", 75.0)),
            very_late_threshold_minutes = int(data.get("very_late_threshold_minutes", 60)),
        )
        return jsonify({"ok": True, "message": "Policy saved successfully."})
    except Exception as exc:
        logger.error("Policy save error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


# ===========================================================================
# Routes — Attendance Log
# ===========================================================================

@app.route("/log")
def attendance_log():
    """Filterable, paginated attendance history page."""
    employees   = db.get_all_employees()
    departments = db.get_departments()
    return render_template("log.html", active_page="log",
                           employees=employees, departments=departments)


@app.route("/api/log")
def api_log():
    """JSON: paginated + filtered historical attendance records."""
    try:
        employee_id = request.args.get("employee_id", type=int)
        department  = request.args.get("department")
        date_from   = request.args.get("date_from")
        date_to     = request.args.get("date_to")
        status      = request.args.get("status")
        page        = request.args.get("page", 1, type=int)
        per_page    = request.args.get("per_page", 25, type=int)

        records, total = db.get_historical_records(
            employee_id = employee_id,
            department  = department,
            date_from   = date_from,
            date_to     = date_to,
            status      = status,
            page        = page,
            per_page    = per_page,
        )

        policy = db.get_policy()
        rows = []
        for r in records:
            rows.append({
                "id":               r["id"],
                "employee_id":      r["employee_id"],
                "name":             r["name"],
                "department":       r["department"],
                "date":             r["date"],
                "check_in_time":    r["check_in_time"][11:19],
                "status":           r["status"],
                "status_label":     format_status_label(r["status"], r["check_in_time"], policy),
                "deduction_amount": r["deduction_amount"],
                "confidence":       r["recognition_confidence"],
            })

        return jsonify({
            "ok":         True,
            "data":       rows,
            "total":      total,
            "page":       page,
            "per_page":   per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        })
    except Exception as exc:
        logger.error("api_log error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


# ===========================================================================
# Routes — Payroll Export
# ===========================================================================

@app.route("/export")
def export_page():
    """Payroll export page (renders inside base template)."""
    return render_template("export.html", active_page="export")


@app.route("/export/payroll")
def export_payroll():
    """
    Stream a CSV payroll export for a given date range.
    Query params: date_from (YYYY-MM-DD), date_to (YYYY-MM-DD), format (csv|xlsx)
    """
    date_from = request.args.get("date_from", "")
    date_to   = request.args.get("date_to",   "")
    fmt       = request.args.get("format", "csv").lower()

    if not date_from or not date_to:
        return jsonify({"ok": False, "error": "date_from and date_to are required."}), 400

    records = db.get_records_for_export(date_from, date_to)
    rows = [dict(r) for r in records]

    if not rows:
        return jsonify({"ok": False, "error": "No records found for the selected date range."}), 404

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.columns = [c.replace("_", " ").title() for c in df.columns]

        filename = f"payroll_{date_from}_{date_to}"

        if fmt == "xlsx":
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Payroll")
            buf.seek(0)
            return send_file(
                buf,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                as_attachment=True,
                download_name=f"{filename}.xlsx",
            )
        else:
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            buf.seek(0)
            return Response(
                buf.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{filename}.csv"'},
            )

    except ImportError:
        # pandas not installed — fallback to raw csv module
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="payroll_{date_from}_{date_to}.csv"'},
        )
    except Exception as exc:
        logger.error("Export error: %s", exc)
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/employees")
def api_employees():
    """JSON: list of all employees (for select dropdowns)."""
    emps = db.get_all_employees()
    return jsonify({
        "ok":   True,
        "data": [{"id": e["id"], "name": e["name"], "department": e["department"], "base_salary": e["base_salary"]} for e in emps],
    })


@app.route("/api/employees/<int:emp_id>", methods=["PUT"])
def api_edit_employee(emp_id: int):
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    dept = (data.get("department") or "").strip()
    salary = data.get("base_salary")

    if not name: return jsonify({"ok": False, "error": "Name is required."}), 400
    if not dept: return jsonify({"ok": False, "error": "Department is required."}), 400
    try:
        salary = float(salary)
        if salary <= 0: raise ValueError
    except:
        return jsonify({"ok": False, "error": "Invalid salary."}), 400

    emp = db.get_employee_by_id(emp_id)
    if not emp:
        return jsonify({"ok": False, "error": "Employee not found."}), 404

    try:
        db.update_employee(emp_id, name, dept, salary)
        return jsonify({"ok": True, "message": "Employee updated successfully."})
    except Exception as exc:
        logger.error("Edit error: %s", exc)
        return jsonify({"ok": False, "error": "Database error."}), 500


# ===========================================================================
# App startup
# ===========================================================================

def create_app() -> Flask:
    """Initialise the database and return the configured Flask app."""
    db.init_db()
    return app


def _start_background_services() -> None:
    """Start webcam + recognition daemon threads."""
    global webcam_stream, face_engine

    webcam_stream = WebcamStream(src=0).start()
    face_engine   = FaceRecognitionEngine(webcam_stream)

    recog_thread = threading.Thread(
        target=_recognition_loop, daemon=True, name="face-recognition"
    )
    recog_thread.start()
    logger.info("Background services started")


if __name__ == "__main__":
    create_app()
    _start_background_services()
    logger.info("Starting Flask server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
