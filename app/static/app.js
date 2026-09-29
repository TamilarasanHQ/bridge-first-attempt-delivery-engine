// BRIDGE Dispatch Manager Dashboard & Customer Portal Logic

let currentManifest = null;
let currentFilter = 'ALL';
let activePortalPkg = 'PKG-DLA3-9941';
let currentSelectedOption = 'ACCESS_POINT';

function switchView(viewName) {
  document.getElementById('view-dashboard').style.display = (viewName === 'dashboard') ? 'block' : 'none';
  document.getElementById('view-portal').style.display = (viewName === 'portal') ? 'block' : 'none';
  document.getElementById('view-simulator').style.display = (viewName === 'simulator') ? 'block' : 'none';

  document.querySelectorAll('.nav-tabs .tab-btn').forEach(btn => btn.classList.remove('active'));
  if (viewName === 'dashboard') document.querySelectorAll('.nav-tabs .tab-btn')[0].classList.add('active');
  if (viewName === 'portal') document.querySelectorAll('.nav-tabs .tab-btn')[1].classList.add('active');
  if (viewName === 'simulator') {
    document.querySelectorAll('.nav-tabs .tab-btn')[2].classList.add('active');
    runSimulation();
  }
}

function updateRangeLabel(elementId, value) {
  const el = document.getElementById(elementId);
  if (el) el.textContent = value;
}

// -----------------------------------------------------------------------------
// Manifest Scoring & Rendering
// -----------------------------------------------------------------------------
async function loadSampleManifest() {
  try {
    const btn = document.querySelector('button.btn-primary');
    const origText = btn.innerHTML;
    btn.innerHTML = '⏳ Loading & Evaluating...';
    btn.disabled = true;

    // 1. Fetch sample Amazon packages
    const sampleResp = await fetch('/api/v1/sample-manifest');
    const sampleData = await sampleResp.json();

    // 2. Score with ML model
    const predictResp = await fetch('/api/v1/predict-manifest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sampleData)
    });
    const result = await predictResp.json();
    currentManifest = result;

    // 3. Update KPIs
    document.getElementById('kpi-total').textContent = result.total_packages;
    document.getElementById('kpi-high-risk').textContent = result.high_risk_count;
    document.getElementById('kpi-high-risk-pct').textContent = `${result.high_risk_percentage} of network dispatch`;
    document.getElementById('kpi-prevented').textContent = result.estimated_failures_preventable;
    document.getElementById('kpi-savings').textContent = `$${result.projected_cost_savings_usd.toFixed(2)}`;

    // 4. Render Table
    document.getElementById('table-subtitle').textContent = `Manifest ${result.manifest_id} (${result.total_packages} parcels, ${result.high_risk_count} requiring pre-dispatch action)`;
    renderTable(result.assessments);

    btn.innerHTML = origText;
    btn.disabled = false;
  } catch (err) {
    alert('Error loading sample manifest: ' + err);
  }
}

function renderTable(assessments) {
  const tbody = document.getElementById('manifest-tbody');
  tbody.innerHTML = '';

  let filtered = assessments;
  if (currentFilter === 'HIGH') {
    filtered = assessments.filter(a => a.risk_tier === 'HIGH');
  } else if (currentFilter === 'RESOLVED') {
    filtered = assessments.filter(a => a.status.includes('REROUTED') || a.status.includes('CONFIRMED') || a.status.includes('RESOLVED'));
  }

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-secondary); padding: 30px;">No parcels match the selected filter.</td></tr>`;
    return;
  }

  filtered.forEach(item => {
    const tr = document.createElement('tr');
    tr.id = `row-${item.package_id}`;

    // Fill bar color
    let barColor = '#10b981';
    if (item.risk_tier === 'HIGH') barColor = '#ef4444';
    else if (item.risk_tier === 'MEDIUM') barColor = '#f59e0b';

    // Width bounded between 4% and 100%
    const barWidth = Math.min(Math.max(item.failure_probability * 100 * 8, 6), 100);

    const reasonsList = item.top_risk_factors.map(r => `• ${r}`).join('<br>');

    let actionBtnHtml = '';
    if (item.risk_tier === 'HIGH') {
      actionBtnHtml = `
        <button class="btn btn-primary" style="padding: 6px 10px; font-size: 11px;" onclick="triggerSms('${item.package_id}', this)">
          📱 Send SMS
        </button>
        <button class="btn btn-secondary" style="padding: 6px 10px; font-size: 11px; margin-top: 4px;" onclick="openPortalForPackage('${item.package_id}', '${item.recipient_name}', '${item.address_line}')">
          ⚙️ Portal
        </button>
      `;
    } else {
      actionBtnHtml = `<span style="font-size: 11px; color: var(--text-muted);">Standard Flow</span>`;
    }

    tr.innerHTML = `
      <td>
        <strong style="color: #fff;">${item.package_id}</strong>
        <div style="font-size: 12px; color: var(--text-secondary);">${item.recipient_name || 'Customer'}</div>
      </td>
      <td>
        <span style="font-weight: 700; color: #60a5fa;">${item.package_id.split('-')[1] || 'DLA3'}</span>
        <div style="font-size: 11px; color: var(--text-muted);">${item.address_line ? item.address_line.split(',')[1] || item.address_line : 'Metro Zone'}</div>
      </td>
      <td>
        <span style="font-size: 13px;">${item.failure_probability > 0.03 ? '210–290s dwell' : '45–65s dwell'}</span>
        <div style="font-size: 11px; color: var(--text-muted);">${item.failure_probability > 0.03 ? 'Restricted Window' : 'All-Day Window'}</div>
      </td>
      <td>
        <div style="font-weight: 800; font-size: 15px; color: ${barColor};">${item.failure_percentage}</div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: ${barWidth}%; background: ${barColor};"></div>
        </div>
      </td>
      <td>
        <span class="risk-tag ${item.risk_tier}">${item.risk_tier}</span>
      </td>
      <td>
        <div class="factors-list">${reasonsList}</div>
      </td>
      <td>
        <span class="status-badge ${item.status}" id="status-badge-${item.package_id}">${item.status}</span>
      </td>
      <td>
        <div style="display: flex; flex-direction: column;">${actionBtnHtml}</div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function filterTable(type, btn) {
  currentFilter = type;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  if (currentManifest) {
    renderTable(currentManifest.assessments);
  }
}

// -----------------------------------------------------------------------------
// Live Notification Simulation
// -----------------------------------------------------------------------------
async function triggerSms(pkgId, buttonEl) {
  buttonEl.disabled = true;
  buttonEl.textContent = '⏳ Sending...';

  try {
    const resp = await fetch(`/api/v1/trigger-notification/${pkgId}`, { method: 'POST' });
    const data = await resp.json();

    const badge = document.getElementById(`status-badge-${pkgId}`);
    if (badge) {
      badge.textContent = 'NOTIFICATION_SENT';
      badge.className = 'status-badge NOTIFICATION_SENT';
    }

    buttonEl.textContent = '✅ Alerted';
    buttonEl.className = 'btn btn-success';

    // Update item in local manifest store
    if (currentManifest) {
      const target = currentManifest.assessments.find(a => a.package_id === pkgId);
      if (target) target.status = 'NOTIFICATION_SENT';
    }
  } catch (err) {
    alert('Failed to send notification: ' + err);
    buttonEl.disabled = false;
    buttonEl.textContent = '📱 Send SMS';
  }
}

// -----------------------------------------------------------------------------
// Customer Portal Interaction (Embedded & Preview)
// -----------------------------------------------------------------------------
function openPortalForPackage(pkgId, recipient, address) {
  activePortalPkg = pkgId;
  document.getElementById('portal-pkg-id').textContent = pkgId;
  document.getElementById('portal-pkg-recipient').textContent = `${recipient || 'Valued Customer'} • ${address || ''}`;
  document.getElementById('portal-confirmation-box').style.display = 'none';
  document.getElementById('preference-options-list').style.display = 'flex';
  document.querySelector('#view-portal button.btn-primary').style.display = 'block';
  
  switchView('portal');
}

function selectOption(opt, element) {
  currentSelectedOption = opt;
  document.querySelectorAll('#preference-options-list .option-item').forEach(el => {
    el.classList.remove('selected');
    el.querySelector('input[type="radio"]').checked = false;
  });
  element.classList.add('selected');
  element.querySelector('input[type="radio"]').checked = true;

  document.getElementById('reschedule-input').style.display = (opt === 'RESCHEDULE') ? 'block' : 'none';
  document.getElementById('neighbor-input').style.display = (opt === 'NEIGHBOR') ? 'block' : 'none';
  document.getElementById('gate-input').style.display = (opt === 'CONFIRM_AVAILABLE') ? 'block' : 'none';
}

async function submitPreference() {
  const payload = {
    package_id: activePortalPkg,
    selected_option: currentSelectedOption,
    reschedule_date: document.getElementById('reschedule-date')?.value || null,
    neighbor_notes: document.getElementById('neighbor-notes')?.value || null,
    gate_code: document.getElementById('gate-code')?.value || null
  };

  try {
    const resp = await fetch('/api/v1/preferences', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await resp.json();

    // Show confirmation box
    document.getElementById('preference-options-list').style.display = 'none';
    document.querySelector('#view-portal button.btn-primary').style.display = 'none';
    const confBox = document.getElementById('portal-confirmation-box');
    confBox.style.display = 'block';
    document.getElementById('portal-confirmation-msg').innerHTML = `
      <strong>Routing Decision:</strong> ${result.new_operational_routing}<br>
      <strong>Confirmation Code:</strong> ${result.confirmation_code}<br>
      <span style="color: #9ca3af;">Driver manifest updated. First attempt failure risk successfully averted.</span>
    `;

    // Update status in Dispatch Table
    const badge = document.getElementById(`status-badge-${activePortalPkg}`);
    if (badge) {
      badge.textContent = result.status;
      badge.className = `status-badge ${result.status}`;
    }

    if (currentManifest) {
      const item = currentManifest.assessments.find(a => a.package_id === activePortalPkg);
      if (item) item.status = result.status;
    }
  } catch (err) {
    alert('Error submitting preference: ' + err);
  }
}

// -----------------------------------------------------------------------------
// What-If Simulator Logic
// -----------------------------------------------------------------------------
async function runSimulation() {
  const station = document.getElementById('sim-station').value;
  const dow = document.getElementById('sim-dow').value;
  const dwell = parseFloat(document.getElementById('sim-dwell').value);
  const hasWindow = parseInt(document.getElementById('sim-window').value);
  const util = parseFloat(document.getElementById('sim-util').value) / 100.0;
  const volLiters = parseFloat(document.getElementById('sim-vol').value);
  const volCm3 = volLiters * 1000.0;

  const payload = {
    package_id: "SIM-PARCEL-LIVE",
    station_code: station,
    day_of_week: dow,
    departure_hour: 13,
    is_weekend: (dow === 'Saturday') ? 1 : 0,
    month: 8,
    latitude: 34.05,
    longitude: -118.25,
    depth_cm: 30.0,
    height_cm: 25.0,
    width_cm: 20.0,
    package_volume_cm3: volCm3,
    planned_service_time_seconds: dwell,
    has_time_window: hasWindow,
    time_window_duration_hours: (hasWindow === 1) ? 2.0 : 0.0,
    executor_capacity_cm3: 10000000.0,
    planned_route_package_count: 150,
    planned_route_stop_count: 120,
    planned_route_volume_cm3: 1500000.0,
    route_volume_utilization: util
  };

  try {
    const resp = await fetch('/api/v1/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();

    const probEl = document.getElementById('sim-risk-prob');
    const badgeEl = document.getElementById('sim-risk-badge');
    const factorsEl = document.getElementById('sim-factors');

    probEl.textContent = data.failure_percentage;
    badgeEl.textContent = `${data.risk_tier} RISK (${data.operational_action})`;
    badgeEl.className = `risk-tag ${data.risk_tier}`;

    if (data.risk_tier === 'HIGH') probEl.style.color = '#ef4444';
    else if (data.risk_tier === 'MEDIUM') probEl.style.color = '#f59e0b';
    else probEl.style.color = '#10b981';

    factorsEl.innerHTML = data.top_risk_factors.map(f => `<li>${f}</li>`).join('');
  } catch (err) {
    console.error('Simulator error:', err);
  }
}

// Auto-load sample manifest on initial page open
window.addEventListener('DOMContentLoaded', () => {
  loadSampleManifest();
});
