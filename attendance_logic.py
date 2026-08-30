"""
attendance_logic.py
===================
Policy evaluation and salary deduction calculation.

This module is intentionally decoupled from Flask and SQLite — it receives
plain Python dicts/datetimes and returns (status, deduction) tuples, making
it trivial to unit-test independently.

Deduction Modes
---------------
flat_tiers  (default)
    Two fixed penalty amounts: one for 'late', one for 'very_late'.
    Simple, predictable, and easy to communicate to employees.

per_minute
    Every minute of lateness (after the grace period) is multiplied by a
    configured per-minute dollar rate.  This rewards employees who are only
    slightly late over those who are very late, but can be harder to explain.

Both modes respect the grace_period_minutes — time after start_time that is
still counted as "on time".  The very_late tier begins only after
very_late_threshold_minutes of actual lateness (after the grace period).
"""

from __future__ import annotations

from datetime import datetime, time


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_check_in(check_in_time: datetime, policy: dict) -> tuple[str, float]:
    """
    Determine whether a check-in is on-time, late, or very late, and
    calculate the corresponding salary deduction.

    Algorithm
    ---------
    1.  Parse `policy['start_time']` (e.g. "09:00") into the same day as
        `check_in_time` to get an absolute `expected_start` datetime.
    2.  Compute `raw_late_seconds = check_in_time - expected_start`.
        Negative means early arrival → always on_time, deduction = 0.
    3.  Subtract `grace_period_minutes` from the raw lateness.
        Anything ≤ 0 after subtraction is still "on time".
    4.  Classify the remaining lateness against `very_late_threshold_minutes`.
    5.  Apply the configured deduction formula.

    Args:
        check_in_time : datetime — the moment the employee checked in
        policy        : dict — loaded from the attendance_policy table
                        (see database.get_policy() for keys)

    Returns:
        (status, deduction_amount)
        status ∈ {'on_time', 'late', 'very_late'}
        deduction_amount : float in USD (0.0 if on_time)
    """

    # ------------------------------------------------------------------
    # Step 1 — Build the expected start datetime for this calendar day
    # ------------------------------------------------------------------
    try:
        start_h, start_m = map(int, policy["start_time"].split(":"))
    except (KeyError, ValueError):
        start_h, start_m = 9, 0   # fallback: 09:00

    expected_start = check_in_time.replace(
        hour=start_h, minute=start_m, second=0, microsecond=0
    )

    grace_minutes = int(policy.get("grace_period_minutes", 15))

    # ------------------------------------------------------------------
    # Step 2 — How many seconds after expected start did they arrive?
    # ------------------------------------------------------------------
    raw_late_seconds = (check_in_time - expected_start).total_seconds()

    # ------------------------------------------------------------------
    # Step 3 — Subtract grace period to get "effective" lateness in minutes
    # Early arrivals and arrivals within grace → minutes_late = 0
    # ------------------------------------------------------------------
    effective_late_minutes = max(0.0, (raw_late_seconds / 60.0) - grace_minutes)

    if effective_late_minutes <= 0:
        # Employee arrived on time (including within grace window)
        return "on_time", 0.0

    # ------------------------------------------------------------------
    # Step 4 — Classify lateness tier
    # ------------------------------------------------------------------
    very_late_threshold = int(policy.get("very_late_threshold_minutes", 60))

    if effective_late_minutes >= very_late_threshold:
        status = "very_late"
    else:
        status = "late"

    # ------------------------------------------------------------------
    # Step 5 — Calculate deduction based on the configured formula
    # ------------------------------------------------------------------
    deduction_type = policy.get("deduction_type", "flat_tiers")

    if deduction_type == "per_minute":
        deduction = _calc_per_minute(effective_late_minutes, policy)
    else:
        # Default: flat_tiers
        deduction = _calc_flat_tiers(status, policy)

    return status, round(deduction, 2)


# ---------------------------------------------------------------------------
# Deduction formula helpers
# ---------------------------------------------------------------------------

def _calc_per_minute(minutes_late: float, policy: dict) -> float:
    """
    Linear deduction: each minute of lateness (after grace) costs a fixed rate.

    Example: 20 minutes late × $2.00/min = $40.00 deduction.

    Args:
        minutes_late : effective minutes late (already grace-adjusted)
        policy       : full policy dict
    Returns:
        deduction amount in dollars
    """
    rate = float(policy.get("per_minute_rate", 2.0))
    return minutes_late * rate


def _calc_flat_tiers(status: str, policy: dict) -> float:
    """
    Fixed-amount deduction based on the employee's lateness tier.

    'late'      → policy['late_flat']       (e.g. $25.00)
    'very_late' → policy['very_late_flat']  (e.g. $75.00)

    Args:
        status : 'late' or 'very_late'
        policy : full policy dict
    Returns:
        deduction amount in dollars
    """
    if status == "very_late":
        return float(policy.get("very_late_flat", 75.0))
    return float(policy.get("late_flat", 25.0))


# ---------------------------------------------------------------------------
# Formatting helpers used by templates and JSON responses
# ---------------------------------------------------------------------------

def format_status_label(status: str, check_in_time: str, policy: dict) -> str:
    """
    Build a human-readable status label such as 'Late · 22 min'.

    Args:
        status        : 'on_time' | 'late' | 'very_late'
        check_in_time : ISO datetime string of the actual check-in
        policy        : policy dict (needed to compute minutes late)
    Returns:
        Short display string
    """
    if status == "on_time":
        return "On Time"

    try:
        dt = datetime.fromisoformat(check_in_time)
        start_h, start_m = map(int, policy["start_time"].split(":"))
        expected = dt.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        grace = int(policy.get("grace_period_minutes", 15))
        raw_mins = (dt - expected).total_seconds() / 60.0
        effective_mins = max(0, int(raw_mins - grace))
    except Exception:
        effective_mins = 0

    label = "Very Late" if status == "very_late" else "Late"
    if effective_mins > 0:
        return f"{label} · {effective_mins} min"
    return label
