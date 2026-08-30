"""
database.py
===========
SQLite database layer for the Face-Recognition Attendance System.

Tables
------
- employees          : registered staff with optional face encodings (pickle BLOB)
- attendance_records : per-day check-in events with status and deduction
- attendance_policy  : single-row site-wide policy configuration

All queries return sqlite3.Row objects (accessible like dicts).
"""

from __future__ import annotations

import sqlite3
import pickle
import os
import random
from datetime import datetime, date, time, timedelta

# ---------------------------------------------------------------------------
# Database path — sits alongside this file
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    """Return a new SQLite connection with dict-like Row access."""
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create tables if they do not exist, then seed sample data on a fresh DB.
    Safe to call on every app start — uses CREATE TABLE IF NOT EXISTS.
    """
    conn = get_db()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS employees (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT    NOT NULL,
            department       TEXT    NOT NULL,
            base_salary      REAL    NOT NULL,
            face_encodings   BLOB,           -- pickle-serialized list[np.ndarray]
            created_at       TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS attendance_records (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id             INTEGER NOT NULL,
            date                    TEXT    NOT NULL,   -- ISO date YYYY-MM-DD
            check_in_time           TEXT    NOT NULL,   -- ISO datetime
            status                  TEXT    NOT NULL,   -- on_time | late | very_late
            deduction_amount        REAL    DEFAULT 0.0,
            recognition_confidence  REAL    DEFAULT 0.0,
            FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE,
            UNIQUE (employee_id, date)                 -- one record per employee per day
        );

        CREATE TABLE IF NOT EXISTS attendance_policy (
            id                          INTEGER PRIMARY KEY CHECK (id = 1),
            start_time                  TEXT    NOT NULL DEFAULT '09:00',
            grace_period_minutes        INTEGER NOT NULL DEFAULT 15,
            deduction_type              TEXT    NOT NULL DEFAULT 'flat_tiers',
            per_minute_rate             REAL    DEFAULT 2.0,
            late_flat                   REAL    DEFAULT 25.0,
            very_late_flat              REAL    DEFAULT 75.0,
            very_late_threshold_minutes INTEGER DEFAULT 60
        );
    """)

    conn.commit()
    conn.close()

    _seed_data()


# ---------------------------------------------------------------------------
# Seed data — runs only on a fresh (empty) database
# ---------------------------------------------------------------------------

def _seed_data() -> None:
    """
    Populate sample employees and 30 days of historical attendance on first run.
    Employees are seeded WITHOUT face encodings so they appear in the dashboard
    but face recognition is only activated when enrolled via the webcam flow.
    """
    conn = get_db()
    c = conn.cursor()

    # Abort if employees already exist
    if c.execute("SELECT COUNT(*) FROM employees").fetchone()[0] > 0:
        conn.close()
        return

    # ------------------------------------------------------------------
    # Default attendance policy
    # ------------------------------------------------------------------
    c.execute("""
        INSERT OR IGNORE INTO attendance_policy
            (id, start_time, grace_period_minutes, deduction_type,
             per_minute_rate, late_flat, very_late_flat, very_late_threshold_minutes)
        VALUES (1, '09:00', 15, 'flat_tiers', 2.0, 25.0, 75.0, 60)
    """)

    # ------------------------------------------------------------------
    # Sample employees
    # ------------------------------------------------------------------
    sample_employees = [
        ("Sarah Mitchell",  "Engineering", 95000.0),
        ("James Okafor",    "Product",     88000.0),
        ("Priya Sharma",    "Design",      82000.0),
        ("Daniel Torres",   "Engineering", 91000.0),
        ("Layla Hassan",    "HR",          76000.0),
    ]

    employee_ids = []
    for name, dept, salary in sample_employees:
        c.execute(
            "INSERT INTO employees (name, department, base_salary) VALUES (?, ?, ?)",
            (name, dept, salary),
        )
        employee_ids.append(c.lastrowid)

    conn.commit()

    # ------------------------------------------------------------------
    # Today's attendance — 4 of 5 employees checked in (one absent)
    # ------------------------------------------------------------------
    today = date.today()
    todays_data = [
        # (minutes_offset_from_9am, status, deduction)
        (-13, "on_time",   0.0),   # 08:47 — early
        ( 12, "on_time",   0.0),   # 09:12 — within grace
        ( 28, "late",     25.0),   # 09:28 — late
        ( -5, "on_time",   0.0),   # 08:55 — early
        # employee_ids[4] absent today
    ]
    base_start = datetime.combine(today, time(9, 0))
    for i, (offset, status, deduction) in enumerate(todays_data):
        check_in_dt = base_start + timedelta(minutes=offset)
        c.execute("""
            INSERT OR IGNORE INTO attendance_records
                (employee_id, date, check_in_time, status, deduction_amount, recognition_confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            employee_ids[i],
            today.isoformat(),
            check_in_dt.isoformat(timespec="seconds"),
            status,
            deduction,
            0.0,
        ))

    # ------------------------------------------------------------------
    # 29 days of historical data (Mon–Fri only)
    # ------------------------------------------------------------------
    random.seed(42)
    for day_offset in range(1, 30):
        past_date = today - timedelta(days=day_offset)
        if past_date.weekday() >= 5:          # skip Sat/Sun
            continue
        past_start = datetime.combine(past_date, time(9, 0))

        for emp_id in employee_ids:
            if random.random() > 0.82:        # ~18% absent
                continue

            # Weighted distribution: mostly on-time, some late, rarely very late
            late_mins = random.choices(
                [0, random.randint(16, 45), random.randint(61, 120)],
                weights=[65, 25, 10],
            )[0]
            check_in_dt = past_start + timedelta(minutes=late_mins)

            if late_mins == 0:
                status, deduction = "on_time", 0.0
            elif late_mins <= 60:
                status, deduction = "late", 25.0
            else:
                status, deduction = "very_late", 75.0

            c.execute("""
                INSERT OR IGNORE INTO attendance_records
                    (employee_id, date, check_in_time, status, deduction_amount, recognition_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                emp_id,
                past_date.isoformat(),
                check_in_dt.isoformat(timespec="seconds"),
                status,
                deduction,
                0.0,
            ))

    conn.commit()
    conn.close()


# ===========================================================================
# Employee Queries
# ===========================================================================

def get_all_employees() -> list:
    """Return all employees as a list of Row objects."""
    with get_db() as conn:
        return conn.execute("SELECT * FROM employees ORDER BY name").fetchall()


def get_employee_by_id(emp_id: int):
    """Return a single employee Row or None."""
    with get_db() as conn:
        return conn.execute("SELECT * FROM employees WHERE id = ?", (emp_id,)).fetchone()


def create_employee(name: str, department: str, base_salary: float,
                    face_encodings: list | None = None) -> int:
    """
    Insert a new employee.  face_encodings should be a list of numpy arrays
    (128-d each); it is pickle-serialized before storage.
    Returns the new employee's id.
    """
    blob = pickle.dumps(face_encodings) if face_encodings else None
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO employees (name, department, base_salary, face_encodings) VALUES (?, ?, ?, ?)",
            (name, department, float(base_salary), blob),
        )
        return cur.lastrowid


def update_employee(emp_id: int, name: str, department: str, base_salary: float) -> None:
    """Update text fields for an existing employee."""
    with get_db() as conn:
        conn.execute(
            "UPDATE employees SET name = ?, department = ?, base_salary = ? WHERE id = ?",
            (name, department, float(base_salary), emp_id),
        )

def update_employee_encodings(emp_id: int, face_encodings: list) -> None:
    """Replace the stored face encodings for an existing employee."""
    blob = pickle.dumps(face_encodings)
    with get_db() as conn:
        conn.execute(
            "UPDATE employees SET face_encodings = ? WHERE id = ?",
            (blob, emp_id),
        )


def delete_employee(emp_id: int) -> None:
    """Delete an employee and cascade-delete their attendance records."""
    with get_db() as conn:
        conn.execute("DELETE FROM employees WHERE id = ?", (emp_id,))


def get_enrolled_employees() -> list[tuple]:
    """
    Return employees that have face encodings stored, as a list of
    (employee_id, name, list[np.ndarray]) tuples — ready for face matching.
    """
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, face_encodings FROM employees WHERE face_encodings IS NOT NULL"
        ).fetchall()

    result = []
    for row in rows:
        try:
            encodings = pickle.loads(row["face_encodings"])
            result.append((row["id"], row["name"], encodings))
        except Exception:
            pass   # skip corrupt encoding blobs
    return result


def check_duplicate_enrollment(name: str) -> bool:
    """Return True if an employee with this exact name already exists."""
    with get_db() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM employees WHERE LOWER(name) = LOWER(?)", (name,)
        ).fetchone()[0]
    return count > 0


# ===========================================================================
# Attendance Record Queries
# ===========================================================================

def log_checkin(employee_id: int, check_in_time: datetime,
                status: str, deduction: float, confidence: float) -> int | None:
    """
    Insert an attendance record for today.
    Returns the new record id, or None if the employee already checked in today.
    Uses INSERT OR IGNORE on the UNIQUE (employee_id, date) constraint.
    """
    today = check_in_time.date().isoformat()
    with get_db() as conn:
        cur = conn.execute("""
            INSERT OR IGNORE INTO attendance_records
                (employee_id, date, check_in_time, status, deduction_amount, recognition_confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            employee_id,
            today,
            check_in_time.isoformat(timespec="seconds"),
            status,
            round(deduction, 2),
            round(confidence, 1),
        ))
        return cur.lastrowid if cur.rowcount else None


def get_today_records() -> list:
    """
    Return today's attendance records joined with employee info,
    ordered by check-in time ascending.
    """
    today = date.today().isoformat()
    with get_db() as conn:
        return conn.execute("""
            SELECT ar.*, e.name, e.department, e.base_salary
            FROM attendance_records ar
            JOIN employees e ON e.id = ar.employee_id
            WHERE ar.date = ?
            ORDER BY ar.check_in_time ASC
        """, (today,)).fetchall()


def is_checked_in_today(employee_id: int) -> bool:
    """Return True if this employee already has a record for today."""
    today = date.today().isoformat()
    with get_db() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM attendance_records WHERE employee_id = ? AND date = ?",
            (employee_id, today),
        ).fetchone()[0]
    return count > 0


def get_today_stats() -> dict:
    """
    Aggregate statistics for today's dashboard stat cards.
    Returns a dict with keys:
        total_employees, checked_in, late_count, avg_checkin_time, total_deductions
    """
    today = date.today().isoformat()
    with get_db() as conn:
        total_employees = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]

        row = conn.execute("""
            SELECT
                COUNT(*)                                          AS checked_in,
                SUM(CASE WHEN status != 'on_time' THEN 1 ELSE 0 END) AS late_count,
                AVG(
                    (strftime('%H', check_in_time) * 60) +
                     strftime('%M', check_in_time)
                )                                                 AS avg_min_of_day,
                SUM(deduction_amount)                             AS total_deductions
            FROM attendance_records
            WHERE date = ?
        """, (today,)).fetchone()

    avg_time_str = "—"
    if row["avg_min_of_day"] is not None:
        total_minutes = int(row["avg_min_of_day"])
        avg_time_str = f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"

    return {
        "total_employees":  total_employees,
        "checked_in":       row["checked_in"]       or 0,
        "late_count":       row["late_count"]        or 0,
        "avg_checkin_time": avg_time_str,
        "total_deductions": round(row["total_deductions"] or 0.0, 2),
    }


def get_recent_checkin() -> dict | None:
    """Return the most recent attendance record (any day) with employee info."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT ar.*, e.name, e.department
            FROM attendance_records ar
            JOIN employees e ON e.id = ar.employee_id
            ORDER BY ar.check_in_time DESC
            LIMIT 1
        """).fetchone()
    return dict(row) if row else None


def get_historical_records(employee_id: int | None = None,
                           department: str | None = None,
                           date_from: str | None = None,
                           date_to: str | None = None,
                           status: str | None = None,
                           page: int = 1,
                           per_page: int = 25) -> tuple[list, int]:
    """
    Return paginated historical attendance records with optional filters.
    Returns (records_list, total_count).
    """
    conditions = []
    params: list = []

    if employee_id:
        conditions.append("ar.employee_id = ?")
        params.append(employee_id)
    if department:
        conditions.append("e.department = ?")
        params.append(department)
    if date_from:
        conditions.append("ar.date >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("ar.date <= ?")
        params.append(date_to)
    if status:
        conditions.append("ar.status = ?")
        params.append(status)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    base_query = f"""
        FROM attendance_records ar
        JOIN employees e ON e.id = ar.employee_id
        {where_clause}
    """

    with get_db() as conn:
        total = conn.execute(f"SELECT COUNT(*) {base_query}", params).fetchone()[0]

        offset = (page - 1) * per_page
        records = conn.execute(f"""
            SELECT ar.*, e.name, e.department, e.base_salary
            {base_query}
            ORDER BY ar.date DESC, ar.check_in_time DESC
            LIMIT ? OFFSET ?
        """, params + [per_page, offset]).fetchall()

    return records, total


def get_records_for_export(date_from: str, date_to: str) -> list:
    """Return all records in a date range for payroll CSV/Excel export."""
    with get_db() as conn:
        return conn.execute("""
            SELECT
                e.name        AS employee_name,
                e.department,
                e.base_salary,
                ar.date,
                ar.check_in_time,
                ar.status,
                ar.deduction_amount,
                ar.recognition_confidence
            FROM attendance_records ar
            JOIN employees e ON e.id = ar.employee_id
            WHERE ar.date BETWEEN ? AND ?
            ORDER BY ar.date DESC, e.name ASC
        """, (date_from, date_to)).fetchall()


def get_departments() -> list[str]:
    """Return a sorted list of distinct department names."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT department FROM employees ORDER BY department"
        ).fetchall()
    return [r["department"] for r in rows]


# ===========================================================================
# Attendance Policy Queries
# ===========================================================================

def get_policy() -> dict:
    """
    Return the active attendance policy as a dict.
    Inserts a default policy row if none exists.
    """
    with get_db() as conn:
        row = conn.execute("SELECT * FROM attendance_policy WHERE id = 1").fetchone()
        if row is None:
            conn.execute("""
                INSERT INTO attendance_policy
                    (id, start_time, grace_period_minutes, deduction_type,
                     per_minute_rate, late_flat, very_late_flat, very_late_threshold_minutes)
                VALUES (1, '09:00', 15, 'flat_tiers', 2.0, 25.0, 75.0, 60)
            """)
            row = conn.execute("SELECT * FROM attendance_policy WHERE id = 1").fetchone()
    return dict(row)


def save_policy(start_time: str, grace_period_minutes: int, deduction_type: str,
                per_minute_rate: float, late_flat: float, very_late_flat: float,
                very_late_threshold_minutes: int) -> None:
    """Upsert the single attendance policy row."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO attendance_policy
                (id, start_time, grace_period_minutes, deduction_type,
                 per_minute_rate, late_flat, very_late_flat, very_late_threshold_minutes)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                start_time                  = excluded.start_time,
                grace_period_minutes        = excluded.grace_period_minutes,
                deduction_type              = excluded.deduction_type,
                per_minute_rate             = excluded.per_minute_rate,
                late_flat                   = excluded.late_flat,
                very_late_flat              = excluded.very_late_flat,
                very_late_threshold_minutes = excluded.very_late_threshold_minutes
        """, (start_time, grace_period_minutes, deduction_type,
              per_minute_rate, late_flat, very_late_flat, very_late_threshold_minutes))
