/**
 * dashboard.js
 * ============
 * Client-side polling logic for the main dashboard.
 *
 * Polls three endpoints every 3 seconds:
 *   /api/stats         → updates the four stat cards
 *   /api/recent_checkin → updates the "Last Check-In" toast strip
 *   /api/today_log     → rebuilds the today attendance table
 *
 * Uses vanilla fetch() with no framework dependencies.
 */

/* -------------------------------------------------------------------------
   State
   ------------------------------------------------------------------------- */
let lastCheckinId = null;    // Track last seen check-in to trigger toast popup

/* -------------------------------------------------------------------------
   Utility helpers
   ------------------------------------------------------------------------- */

/**
 * Get initials from a full name string.
 * "Sarah Mitchell" → "SM"
 */
function initials(name) {
  if (!name) return '?';
  return name.split(' ').map(p => p[0]).join('').toUpperCase().slice(0, 2);
}

/**
 * Show a transient pop-up notification at the bottom-right corner.
 */
function showToastPopup(message, type = 'info') {
  const container = document.getElementById('toast-popup');
  if (!container) return;

  const item = document.createElement('div');
  item.className = 'toast-popup-item';

  const colors = { success: '#3DBD84', danger: '#E5674A', info: '#6C6CF4' };
  const color = colors[type] || colors.info;
  item.style.borderLeftColor = color;
  item.style.borderLeftWidth = '3px';
  item.textContent = message;

  container.appendChild(item);

  // Auto-remove after 4 seconds
  setTimeout(() => {
    item.classList.add('fade-out');
    setTimeout(() => item.remove(), 300);
  }, 4000);
}

/**
 * Format a deduction amount for display.
 * 0 → "—"   |   25 → "$25.00"
 */
function fmtDeduction(amount) {
  if (!amount || parseFloat(amount) === 0) return '—';
  return '$' + parseFloat(amount).toFixed(2);
}

/**
 * Determine badge class and label for a status string.
 */
function statusBadge(status, label) {
  const map = {
    on_time:   { cls: 'badge-success', dot: true },
    late:      { cls: 'badge-danger',  dot: true },
    very_late: { cls: 'badge-warning', dot: true },
  };
  const cfg = map[status] || { cls: 'badge-muted', dot: false };
  const dotHtml = cfg.dot ? `<span class="badge-dot"></span>` : '';
  return `<span class="badge ${cfg.cls}">${dotHtml}${label || status}</span>`;
}

/* -------------------------------------------------------------------------
   Stat cards
   ------------------------------------------------------------------------- */

async function pollStats() {
  try {
    const res  = await fetch('/api/stats');
    const json = await res.json();
    if (!json.ok) return;

    const d = json.data;

    // Checked-in card  e.g. "4 / 5"
    setEl('stat-checkin',    `${d.checked_in} / ${d.total_employees}`);
    // Late count
    setEl('stat-late',       d.late_count.toString());
    // Average check-in time
    setEl('stat-avg-time',   d.avg_checkin_time || '—');
    // Total deductions
    setEl('stat-deductions', d.total_deductions > 0 ? `$${d.total_deductions.toFixed(2)}` : '$0.00');

  } catch (err) {
    console.warn('pollStats error:', err);
  }
}

function setEl(id, text) {
  const el = document.getElementById(id);
  if (el && el.textContent !== text) el.textContent = text;
}

/* -------------------------------------------------------------------------
   Recent check-in toast strip
   ------------------------------------------------------------------------- */

async function pollRecentCheckin() {
  try {
    const res  = await fetch('/api/recent_checkin');
    const json = await res.json();
    if (!json.ok || !json.data) return;

    const evt = json.data;
    const key = `${evt.employee_id}-${evt.check_in_time}`;

    // Show a pop-up notification only if this is a new check-in
    if (key !== lastCheckinId) {
      if (lastCheckinId !== null) {
        showToastPopup(
          `${evt.name} checked in — ${evt.status_label || evt.status}`,
          evt.status === 'on_time' ? 'success' : 'danger'
        );
      }
      lastCheckinId = key;
    }

    // Update the toast strip card in the sidebar panel
    const strip = document.getElementById('recent-checkin-strip');
    if (!strip) return;

    const badgeHtml = statusBadge(evt.status, evt.status_label);
    const initStr   = initials(evt.name);

    strip.innerHTML = `
      <div class="toast-event">
        <div class="toast-avatar">${initStr}</div>
        <div class="toast-info">
          <div class="toast-name">${escHtml(evt.name)}</div>
          <div class="toast-dept">${escHtml(evt.department || '—')}</div>
          <div class="toast-time mono">${escHtml(evt.check_in_time || '')}</div>
        </div>
        <div>${badgeHtml}</div>
      </div>`;

  } catch (err) {
    console.warn('pollRecentCheckin error:', err);
  }
}

/* -------------------------------------------------------------------------
   Today's attendance table
   ------------------------------------------------------------------------- */

async function pollTodayLog() {
  try {
    const res  = await fetch('/api/today_log');
    const json = await res.json();
    if (!json.ok) return;

    const rows = json.data;
    const tbody = document.getElementById('today-log-tbody');
    if (!tbody) return;

    if (!rows || rows.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="5">
            <div class="empty-state">
              <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24"
                   fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              <div class="empty-state-title">No check-ins yet today</div>
              <div class="empty-state-desc">Check-ins will appear here automatically</div>
            </div>
          </td>
        </tr>`;
      // Update header count
      setEl('today-log-count', '0 records');
      return;
    }

    // Rebuild table rows
    const html = rows.map(r => `
      <tr>
        <td>
          <div class="employee-cell">
            <div class="avatar">${initials(r.name)}</div>
            <div class="employee-info">
              <div class="emp-name">${escHtml(r.name)}</div>
              <div class="emp-dept">${escHtml(r.department)}</div>
            </div>
          </div>
        </td>
        <td><span class="mono">${escHtml(r.check_in_time)}</span></td>
        <td>${statusBadge(r.status, r.status_label)}</td>
        <td>
          ${r.deduction_amount > 0
            ? `<span class="deduction-amount mono">-$${parseFloat(r.deduction_amount).toFixed(2)}</span>`
            : `<span class="deduction-none mono">—</span>`
          }
        </td>
        <td>
          ${r.confidence > 0
            ? `<span class="mono text-secondary">${parseFloat(r.confidence).toFixed(1)}%</span>`
            : `<span class="text-muted">—</span>`
          }
        </td>
      </tr>`).join('');

    tbody.innerHTML = html;
    setEl('today-log-count', `${rows.length} record${rows.length !== 1 ? 's' : ''}`);

  } catch (err) {
    console.warn('pollTodayLog error:', err);
  }
}

/* -------------------------------------------------------------------------
   HTML escaping
   ------------------------------------------------------------------------- */
function escHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* -------------------------------------------------------------------------
   Init — start polling on DOMContentLoaded
   ------------------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  // Initial fetch immediately
  pollStats();
  pollRecentCheckin();
  pollTodayLog();

  // Then poll every 3 seconds
  setInterval(pollStats,          3000);
  setInterval(pollRecentCheckin,  3000);
  setInterval(pollTodayLog,       5000);
});
