/**
 * enroll.js
 * =========
 * Handles the employee enrollment flow:
 *
 * 1.  User fills in name / department / salary fields.
 * 2.  User clicks "Capture Photo" — this POSTs to /api/enroll/capture
 *     which grabs the current webcam frame server-side and returns a
 *     base64 JPEG thumbnail of the captured frame.
 * 3.  Each successful capture increments a counter and fills a thumbnail slot.
 * 4.  After MIN_CAPTURES (3) successful captures, the "Enroll Employee" submit
 *     button becomes active.
 * 5.  On submit, the employee data is POSTed to /api/enroll/submit.
 * 6.  On success, the page resets and shows a confirmation alert.
 *
 * The server accumulates face encodings in-process between captures, so
 * this script does not need to transmit any image data — just trigger signals.
 */

const MIN_CAPTURES  = 3;
const MAX_CAPTURES  = 5;

let captureCount    = 0;
let isCapturing     = false;
let isSubmitting    = false;

/* -------------------------------------------------------------------------
   DOM references (assigned after DOMContentLoaded)
   ------------------------------------------------------------------------- */
let btnCapture, btnSubmit, btnReset;
let captureCountEl, progressFill, captureMinLabel;
let alertContainer;
let thumbnailSlots; // Array of <div class="capture-thumb"> elements

/* -------------------------------------------------------------------------
   Utility
   ------------------------------------------------------------------------- */
function showAlert(message, type = 'danger') {
  if (!alertContainer) return;
  const html = `<div class="alert alert-${type}">
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      ${type === 'success'
        ? '<polyline points="20 6 9 17 4 12"></polyline>'
        : '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'}
    </svg>
    <span>${escHtml(message)}</span>
  </div>`;
  alertContainer.innerHTML = html;
  alertContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function clearAlert() {
  if (alertContainer) alertContainer.innerHTML = '';
}

function escHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function updateProgress() {
  if (!captureCountEl) return;
  captureCountEl.textContent = captureCount;

  // Progress bar
  const pct = Math.min(captureCount / MIN_CAPTURES, 1) * 100;
  if (progressFill) {
    progressFill.style.width = pct + '%';
    progressFill.classList.toggle('complete', captureCount >= MIN_CAPTURES);
  }

  // Update submit button state
  if (btnSubmit) {
    btnSubmit.disabled = captureCount < MIN_CAPTURES || isSubmitting;
  }

  // Update capture button state
  if (btnCapture) {
    btnCapture.disabled = captureCount >= MAX_CAPTURES || isCapturing;
  }

  // Update hint text
  if (captureMinLabel) {
    if (captureCount >= MAX_CAPTURES) {
      captureMinLabel.textContent = 'Maximum captures reached.';
    } else if (captureCount >= MIN_CAPTURES) {
      captureMinLabel.textContent = `${captureCount} / ${MAX_CAPTURES} captures · Ready to enroll`;
      captureMinLabel.style.color = 'var(--success)';
    } else {
      captureMinLabel.textContent = `${captureCount} / ${MIN_CAPTURES} minimum captures needed`;
      captureMinLabel.style.color = '';
    }
  }
}

function fillThumbnail(index, imgSrc) {
  if (!thumbnailSlots || !thumbnailSlots[index]) return;
  const slot = thumbnailSlots[index];
  slot.innerHTML = `<img src="${imgSrc}" alt="Capture ${index + 1}">`;
  slot.classList.add('filled');
}

/* -------------------------------------------------------------------------
   Capture a photo
   ------------------------------------------------------------------------- */
async function handleCapture() {
  if (isCapturing || captureCount >= MAX_CAPTURES) return;
  clearAlert();
  isCapturing = true;
  btnCapture.disabled = true;

  // Temporary visual feedback on the button
  const origText = btnCapture.innerHTML;
  btnCapture.innerHTML = `<span class="spinner"></span> Capturing…`;

  try {
    // Signal server to grab a webcam frame and extract face encoding
    const res  = await fetch('/api/enroll/capture', { method: 'POST' });
    const json = await res.json();

    if (!json.ok) {
      showAlert(json.error || 'Capture failed. Try again.', 'danger');
    } else {
      // Display thumbnail if returned
      if (json.image_b64) {
        fillThumbnail(captureCount, json.image_b64);
      }
      captureCount = json.count || (captureCount + 1);
      updateProgress();
      clearAlert();
    }
  } catch (err) {
    showAlert('Network error — is the server running?', 'danger');
    console.error('Capture error:', err);
  } finally {
    isCapturing = false;
    btnCapture.innerHTML = origText;
    updateProgress();  // re-evaluate disabled state
  }
}

/* -------------------------------------------------------------------------
   Submit the enrollment form
   ------------------------------------------------------------------------- */
async function handleSubmit(e) {
  e.preventDefault();
  if (isSubmitting || captureCount < MIN_CAPTURES) return;
  clearAlert();

  // Validate form fields
  const name   = (document.getElementById('emp-name')?.value   || '').trim();
  const dept   = (document.getElementById('emp-dept')?.value   || '').trim();
  const salary = (document.getElementById('emp-salary')?.value || '').trim();

  if (!name)   { showAlert('Employee name is required.', 'danger'); return; }
  if (!dept)   { showAlert('Department is required.', 'danger'); return; }
  if (!salary || isNaN(parseFloat(salary)) || parseFloat(salary) <= 0) {
    showAlert('Enter a valid positive salary.', 'danger'); return;
  }

  isSubmitting = true;
  btnSubmit.disabled = true;
  const origText = btnSubmit.innerHTML;
  btnSubmit.innerHTML = `<span class="spinner"></span> Enrolling…`;

  try {
    const res = await fetch('/api/enroll/submit', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ name, department: dept, base_salary: salary }),
    });
    const json = await res.json();

    if (json.ok) {
      showAlert(json.message || `${name} enrolled successfully!`, 'success');
      resetForm();
      // Reload enrolled employees table if it exists on page
      if (typeof loadEnrolledEmployees === 'function') loadEnrolledEmployees();
    } else {
      showAlert(json.error || 'Enrollment failed.', 'danger');
    }
  } catch (err) {
    showAlert('Network error during enrollment.', 'danger');
    console.error('Submit error:', err);
  } finally {
    isSubmitting = false;
    btnSubmit.innerHTML = origText;
    updateProgress();
  }
}

/* -------------------------------------------------------------------------
   Reset the form + pending captures
   ------------------------------------------------------------------------- */
async function resetForm() {
  captureCount = 0;
  updateProgress();

  // Clear thumbnails
  if (thumbnailSlots) {
    thumbnailSlots.forEach(slot => {
      slot.innerHTML = `<div class="empty-slot">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
             fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
          <circle cx="8.5" cy="8.5" r="1.5"/>
          <polyline points="21 15 16 10 5 21"/>
        </svg>
      </div>`;
      slot.classList.remove('filled');
    });
  }

  // Clear form fields
  ['emp-name', 'emp-dept', 'emp-salary'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });

  // Tell server to clear pending encodings
  try {
    await fetch('/api/enroll/clear', { method: 'POST' });
  } catch (e) { /* ignore */ }
}

/* -------------------------------------------------------------------------
   Load enrolled employees table
   ------------------------------------------------------------------------- */
async function loadEnrolledEmployees() {
  const tbody = document.getElementById('enrolled-tbody');
  if (!tbody) return;

  try {
    const res  = await fetch('/api/employees');
    const json = await res.json();
    if (!json.ok) return;

    const emps = json.data;
    if (!emps.length) {
      tbody.innerHTML = `<tr><td colspan="4" class="empty-state-row">
        <div class="empty-state"><div class="empty-state-title">No employees yet</div></div>
      </td></tr>`;
      return;
    }

    tbody.innerHTML = emps.map(e => {
      const init = (e.name || '?').split(' ').map(p => p[0]).join('').toUpperCase().slice(0,2);
      return `<tr>
        <td>
          <div class="employee-cell">
            <div class="avatar">${init}</div>
            <div class="employee-info">
              <div class="emp-name">${escHtml(e.name)}</div>
              <div class="emp-dept">${escHtml(e.department)}</div>
            </div>
          </div>
        </td>
        <td><span class="text-secondary">${escHtml(e.department)}</span></td>
        <td>
          <div class="flex gap-8">
            <button class="btn btn-sm" onclick="openEditModal(${e.id}, '${escHtml(e.name)}', '${escHtml(e.department)}', ${e.base_salary})">
              <i class="ti ti-edit"></i> Edit
            </button>
            <button class="btn btn-danger btn-sm"
                    onclick="deleteEmployee(${e.id}, '${escHtml(e.name)}')">
              <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24"
                   fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6"/>
                <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                <path d="M10 11v6"/><path d="M14 11v6"/>
              </svg>
              Delete
            </button>
          </div>
        </td>
      </tr>`;
    }).join('');
  } catch (err) {
    console.warn('loadEnrolledEmployees error:', err);
  }
}

async function deleteEmployee(empId, name) {
  if (!confirm(`Delete ${name}? This will remove all their attendance records.`)) return;
  try {
    const res  = await fetch(`/api/enroll/delete/${empId}`, { method: 'DELETE' });
    const json = await res.json();
    if (json.ok) {
      showAlert(json.message || 'Employee deleted.', 'success');
      loadEnrolledEmployees();
    } else {
      showAlert(json.error || 'Delete failed.', 'danger');
    }
  } catch (err) {
    showAlert('Network error.', 'danger');
  }
}

/* -------------------------------------------------------------------------
   Edit Employee Modal Logic
   ------------------------------------------------------------------------- */
function openEditModal(empId, name, dept, salary) {
  const modal = document.getElementById('edit-modal');
  if (!modal) return;
  
  document.getElementById('edit-emp-id').value = empId;
  document.getElementById('edit-emp-name').value = name;
  document.getElementById('edit-emp-dept').value = dept;
  document.getElementById('edit-emp-salary').value = salary;
  
  modal.style.display = 'flex';
}

function closeEditModal() {
  const modal = document.getElementById('edit-modal');
  if (modal) modal.style.display = 'none';
}

async function handleEditSubmit(e) {
  e.preventDefault();
  const empId  = document.getElementById('edit-emp-id').value;
  const name   = document.getElementById('edit-emp-name').value.trim();
  const dept   = document.getElementById('edit-emp-dept').value.trim();
  const salary = document.getElementById('edit-emp-salary').value.trim();

  if (!name || !dept || !salary) {
    alert('All fields are required.');
    return;
  }

  const btn = document.getElementById('btn-edit-submit');
  const origText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Saving…`;

  try {
    const res = await fetch(`/api/employees/${empId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, department: dept, base_salary: salary })
    });
    const json = await res.json();
    
    if (json.ok) {
      showAlert(json.message || 'Employee updated.', 'success');
      closeEditModal();
      loadEnrolledEmployees();
    } else {
      alert(json.error || 'Update failed.');
    }
  } catch (err) {
    alert('Network error.');
  } finally {
    btn.disabled = false;
    btn.innerHTML = origText;
  }
}

/* -------------------------------------------------------------------------
   Init
   ------------------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  btnCapture      = document.getElementById('btn-capture');
  btnSubmit       = document.getElementById('btn-enroll-submit');
  btnReset        = document.getElementById('btn-enroll-reset');
  captureCountEl  = document.getElementById('capture-count');
  progressFill    = document.getElementById('capture-progress-fill');
  captureMinLabel = document.getElementById('capture-min-label');
  alertContainer  = document.getElementById('enroll-alert');
  thumbnailSlots  = Array.from(document.querySelectorAll('.capture-thumb'));

  if (btnCapture) btnCapture.addEventListener('click', handleCapture);
  if (btnSubmit)  btnSubmit.addEventListener('click', handleSubmit);
  if (btnReset)   btnReset.addEventListener('click', () => { clearAlert(); resetForm(); });
  
  const editForm = document.getElementById('edit-form');
  if (editForm) editForm.addEventListener('submit', handleEditSubmit);

  updateProgress();
  loadEnrolledEmployees();

  // Also call clear on page load so stale server-side encodings are wiped
  fetch('/api/enroll/clear', { method: 'POST' }).catch(() => {});
});
