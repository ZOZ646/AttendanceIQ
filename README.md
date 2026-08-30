# AttendanceIQ — Enterprise Face-Recognition Attendance System

A professional, locally-running employee attendance management system using
Python (Flask) + OpenCV + dlib face recognition.  Zero cloud dependencies —
everything runs on a single machine.

---

## Features

- **Live face recognition** via laptop webcam using the `face_recognition` (dlib) library
- **Automatic check-in logging** with duplicate prevention (one record per employee per day)
- **Configurable attendance policy** — grace period, flat-tier or per-minute salary deductions
- **Auto-refreshing dashboard** — stat cards, live annotated camera feed, today's log table
- **Employee enrollment** — capture 3-5 reference photos directly from the webcam
- **Attendance log** — filterable/paginated history across any date range
- **Payroll export** — one-click CSV or Excel download via pandas
- **Professional dark-theme UI** — Inter + IBM Plex Mono fonts, enterprise SaaS aesthetics

---

## Prerequisites

### 1. Python 3.10 or 3.11 (recommended)

Download from https://www.python.org/downloads/  
Python 3.12+ may have compatibility issues with some dlib wheels.

### 2. dlib + CMake (required for face recognition)

`face_recognition` depends on `dlib`, which must be compiled from C++ source
**or** installed from a pre-built wheel.

#### Option A — Pre-built wheel (Windows, easiest)

1. Go to: https://github.com/z-mahmud22/Dlib_Windows_Python3.x
2. Download the `.whl` file matching your Python version, e.g.:
   `dlib-19.24.0-cp311-cp311-win_amd64.whl` for Python 3.11 64-bit
3. Install it **before** the rest of requirements:
   ```
   pip install dlib-19.24.0-cp311-cp311-win_amd64.whl
   ```

#### Option B — Build from source (Windows)

1. Install **Visual Studio Build Tools** with the "Desktop development with C++" workload:
   https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install **CMake** (add to PATH during install):
   https://cmake.org/download/
3. Then proceed with `pip install -r requirements.txt` — dlib will compile.

#### macOS

```bash
brew install cmake dlib
```

#### Linux (Ubuntu / Debian)

```bash
sudo apt-get install -y cmake build-essential libopenblas-dev liblapack-dev libx11-dev
```

---

## Installation

```bash
# 1. Clone or unzip the project
cd NTI_Project

# 2. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# 3. (Windows only) Install dlib pre-built wheel first if using Option A:
pip install path\to\dlib-19.24.0-cp311-cp311-win_amd64.whl

# 4. Install all dependencies
pip install -r requirements.txt
```

---

## Running the App

```bash
python app.py
```

Then open your browser at: **http://localhost:5000**

> The server starts on port 5000 by default.  
> The webcam opens automatically in the background — no browser permission prompt needed.

---

## First-Run Behaviour

On first launch, the database is created and seeded with **5 sample employees**
(Sarah Mitchell, James Okafor, Priya Sharma, Daniel Torres, Layla Hassan) and
**30 days of realistic historical attendance data** so the dashboard and log pages
immediately show useful data.

These sample employees have **no face encodings** — they will not be matched by
the recognition loop.  To enable live recognition, enroll real employees via:

**Sidebar → Employees → Enroll Employee**

---

## Project Structure

```
NTI_Project/
├── app.py                  # Flask routes, MJPEG streaming, background recognition thread
├── face_engine.py          # WebcamStream, face encoding/matching, frame annotation
├── attendance_logic.py     # Policy evaluation and deduction calculation
├── database.py             # SQLite schema, CRUD queries, seed data
├── requirements.txt        # Python dependencies
├── attendance.db           # Created automatically on first run
├── templates/
│   ├── base.html           # Sidebar layout + nav
│   ├── dashboard.html      # Main dashboard (stat cards, live feed, table)
│   ├── enroll.html         # Employee enrollment with webcam capture
│   ├── policy.html         # Attendance policy configuration
│   ├── log.html            # Filterable attendance history
│   └── export.html         # Payroll CSV/Excel export
└── static/
    ├── css/style.css       # Full dark-theme design system
    └── js/
        ├── dashboard.js    # Auto-polling for stats + table
        └── enroll.js       # Enrollment capture flow
```

---

## Attendance Policy

Configure via **Sidebar → Attendance Policy**.

| Setting | Default | Description |
|---|---|---|
| Expected Start Time | 09:00 | Workday start time |
| Grace Period | 15 min | Arrivals within this window are "On Time" |
| Deduction Type | Flat Tiers | `flat_tiers` or `per_minute` |
| Late Flat | $25.00 | Deduction for the "Late" tier |
| Very Late Flat | $75.00 | Deduction for the "Very Late" tier |
| Very Late Threshold | 60 min | Minutes late (after grace) to classify as Very Late |

---

## Face Recognition Notes

- Uses the `face_recognition` library (dlib HOG model) — CPU-only, no GPU required
- Match threshold: **0.50** (stricter than the default 0.60 to reduce false positives)
- Recognition runs every 5 frames to keep CPU usage manageable (~15-20% on a modern laptop)
- Minimum **3 face captures** per employee for reliable recognition across lighting / pose variation
- Employees with no face encodings (seed data) will never trigger a false match

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'face_recognition'` | Install dlib first (see Prerequisites) |
| `Camera unavailable` on the feed | Close any other app using the webcam (Zoom, Teams, etc.) |
| Recognition not triggering | Ensure employee was enrolled with ≥ 3 photos in good lighting |
| Port 5000 in use | Change `port=5000` in `app.py` to another port e.g. `5001` |
| Database locked error | Only one instance of `app.py` should be running at a time |

---

## License

For internal enterprise use only.  Not for redistribution.
