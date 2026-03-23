/**
 * Config Wizard — Main Application Logic
 *
 * Handles: step navigation, map interaction, form state, config export.
 */

// ==========================================================================
// Constants
// ==========================================================================
const MAX_POPULATION = 5_000_000;

const STEPS = [
  { id: 'region',     label: 'Region' },
  { id: 'surveys',    label: 'Surveys' },
  { id: 'modes',      label: 'Modes' },
  { id: 'transit',    label: 'Transit' },
  { id: 'demand',     label: 'Demand' },
  { id: 'time',       label: 'Time' },
  { id: 'simulation', label: 'Simulation' },
];

const DURATION_DEFAULTS = {
  Work:     { min: 10, max: 600, trim_priority: 1 },
  School:   { min: 10, max: 600, trim_priority: 1 },
  Shopping: { min: 5,  max: 300, trim_priority: 3 },
  Escort:   { min: 5,  max: 300, trim_priority: 3 },
  Dining:   { min: 5,  max: 180, trim_priority: 3 },
  Social:   { min: 10, max: 300, trim_priority: 3 },
  Other:    { min: 5,  max: 300, trim_priority: 3 },
};

// ==========================================================================
// State
// ==========================================================================
let currentStep = 0;
let selectedCounties = {};   // { GEOID: { name, population, feature } }
let countyGeoJSON = null;
let map = null;
let countyLayer = null;
let zoomMessageEl = null;

// ==========================================================================
// Step Progress Bar
// ==========================================================================

function renderStepProgress() {
  const container = document.getElementById('stepProgress');
  container.innerHTML = '';

  STEPS.forEach((step, i) => {
    if (i > 0) {
      const sep = document.createElement('div');
      sep.className = 'step-progress__separator';
      if (i <= currentStep) sep.classList.add('step-progress__item--completed');
      container.appendChild(sep);
    }

    const item = document.createElement('div');
    item.className = 'step-progress__item';
    if (i === currentStep) item.classList.add('step-progress__item--active');
    if (i < currentStep)  item.classList.add('step-progress__item--completed');
    item.onclick = () => { if (i <= currentStep) goToStep(i); };

    const icon = document.createElement('span');
    icon.className = 'step-progress__icon';
    if (i < currentStep) {
      icon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" width="14" height="14"><polyline points="20 6 9 17 4 12"/></svg>';
    } else {
      icon.textContent = i + 1;
    }

    const label = document.createElement('span');
    label.textContent = step.label;

    item.appendChild(icon);
    item.appendChild(label);
    container.appendChild(item);
  });
}

function goToStep(stepIndex) {
  // Validate current step before moving forward
  if (stepIndex > currentStep && !validateStep(currentStep)) return;

  currentStep = stepIndex;
  renderStepProgress();

  // Show/hide panels
  STEPS.forEach((step, i) => {
    const panel = document.getElementById(`panel-${step.id}`);
    panel.classList.toggle('panel--active', i === currentStep);
    if (i < currentStep) {
      panel.classList.add('panel--completed');
    }
  });

  // Scroll to active panel
  const activePanel = document.getElementById(`panel-${STEPS[currentStep].id}`);
  activePanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Re-invalidate map size if returning to region step
  if (currentStep === 0 && map) {
    setTimeout(() => map.invalidateSize(), 100);
  }
}

function validateStep(stepIndex) {
  switch (stepIndex) {
    case 0: // Region
      if (Object.keys(selectedCounties).length === 0) {
        alert('Please select at least one county.');
        return false;
      }
      const totalPop = getTotalPopulation();
      if (totalPop > MAX_POPULATION) {
        alert(`Total population (${formatNumber(totalPop)}) exceeds the maximum of ${formatNumber(MAX_POPULATION)}. Please remove some counties.`);
        return false;
      }
      updateBadge('region', `${Object.keys(selectedCounties).length} counties`);
      return true;

    case 1: // Surveys
      const nhts = document.getElementById('chkNhts').checked;
      const custom = document.getElementById('chkCustomSurvey').checked;
      if (!nhts && !custom) {
        alert('At least one survey must be selected.');
        return false;
      }
      updateBadge('surveys', nhts && custom ? '2 surveys' : '1 survey');
      return true;

    default:
      // Mark badge as completed
      updateBadge(STEPS[stepIndex].id, 'Done');
      return true;
  }
}

function updateBadge(panelId, text) {
  const badge = document.getElementById(`badge-${panelId}`);
  if (badge) {
    badge.textContent = text;
    badge.classList.remove('hidden');
  }
}

// ==========================================================================
// Navigation Buttons
// ==========================================================================

function setupNavigation() {
  // Next buttons
  for (let i = 0; i < STEPS.length - 1; i++) {
    const btn = document.getElementById(`btnNext${i}`);
    if (btn) btn.onclick = () => goToStep(i + 1);
  }

  // Back buttons
  for (let i = 1; i < STEPS.length; i++) {
    const btn = document.getElementById(`btnBack${i}`);
    if (btn) btn.onclick = () => goToStep(i - 1);
  }
}

// ==========================================================================
// Map (Panel 1 — Region)
// ==========================================================================

async function initMap() {
  map = L.map('countyMap', {
    center: [39.8, -98.5],
    zoom: 4,
    minZoom: 3,
    maxZoom: 12,
  });

  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
    maxZoom: 19,
  }).addTo(map);

  zoomMessageEl = document.getElementById('mapMessage');

  // Load county data
  try {
    const resp = await fetch('/api/counties');
    countyGeoJSON = await resp.json();
    addCountyLayer();
  } catch (err) {
    console.error('Failed to load counties:', err);
    zoomMessageEl.textContent = 'Failed to load county data. Is the server running?';
  }

  // Show/hide counties based on zoom
  map.on('zoomend', updateCountyVisibility);
  updateCountyVisibility();
}

function addCountyLayer() {
  countyLayer = L.geoJSON(countyGeoJSON, {
    style: countyStyle,
    onEachFeature: (feature, layer) => {
      const props = feature.properties;
      layer.on({
        click: () => toggleCounty(props.GEOID, props.name, props.population, feature, layer),
        mouseover: (e) => {
          if (!selectedCounties[props.GEOID]) {
            e.target.setStyle({ fillOpacity: 0.3, weight: 2 });
          }
          layer.bindTooltip(
            `<strong>${props.name}</strong><br>Pop: ${formatNumber(props.population)}`,
            { sticky: true }
          ).openTooltip();
        },
        mouseout: (e) => {
          if (!selectedCounties[props.GEOID]) {
            countyLayer.resetStyle(e.target);
          }
          layer.closeTooltip();
        },
      });
    },
  });

  countyLayer.addTo(map);
}

function countyStyle(feature) {
  const geoid = feature.properties.GEOID;
  const isSelected = !!selectedCounties[geoid];
  return {
    fillColor: isSelected ? '#2563eb' : '#94a3b8',
    weight: isSelected ? 2.5 : 1,
    opacity: 1,
    color: isSelected ? '#1d4ed8' : '#cbd5e1',
    fillOpacity: isSelected ? 0.45 : 0.1,
  };
}

function updateCountyVisibility() {
  if (!countyLayer) return;
  const zoom = map.getZoom();
  const show = zoom >= 6;

  if (show) {
    if (!map.hasLayer(countyLayer)) map.addLayer(countyLayer);
    zoomMessageEl.style.display = 'none';
  } else {
    if (map.hasLayer(countyLayer)) map.removeLayer(countyLayer);
    zoomMessageEl.style.display = 'block';
  }
}

function toggleCounty(geoid, name, population, feature, layer) {
  if (selectedCounties[geoid]) {
    // Deselect
    delete selectedCounties[geoid];
    countyLayer.resetStyle(layer);
  } else {
    // Check population cap
    const newTotal = getTotalPopulation() + (population || 0);
    if (newTotal > MAX_POPULATION) {
      alert(`Adding "${name}" (pop: ${formatNumber(population)}) would exceed the ${formatNumber(MAX_POPULATION)} limit.\nCurrent total: ${formatNumber(getTotalPopulation())}`);
      return;
    }
    selectedCounties[geoid] = { name, population, feature };
    layer.setStyle({
      fillColor: '#2563eb',
      weight: 2.5,
      color: '#1d4ed8',
      fillOpacity: 0.45,
    });
  }

  updateMapStats();
  updateSelectedCountiesList();
  updateNextButton0();
}

function getTotalPopulation() {
  return Object.values(selectedCounties).reduce((sum, c) => sum + (c.population || 0), 0);
}

function updateMapStats() {
  const count = Object.keys(selectedCounties).length;
  const pop = getTotalPopulation();
  document.getElementById('statCountyCount').textContent = count;

  const popEl = document.getElementById('statPopulation');
  popEl.textContent = formatNumber(pop);
  popEl.className = 'map-overlay__value';
  if (pop > MAX_POPULATION * 0.9) popEl.classList.add('map-overlay__value--warning');
  if (pop > MAX_POPULATION) popEl.classList.add('map-overlay__value--danger');
}

function updateSelectedCountiesList() {
  const container = document.getElementById('selectedCountiesList');
  container.innerHTML = '';

  Object.entries(selectedCounties).forEach(([geoid, data]) => {
    const tag = document.createElement('span');
    tag.className = 'tag tag--primary';
    tag.innerHTML = `
      ${data.name}
      <span class="text-xs text-muted">(${formatNumber(data.population)})</span>
      <span class="tag__remove" onclick="removeCounty('${geoid}')">&times;</span>
    `;
    container.appendChild(tag);
  });
}

function removeCounty(geoid) {
  delete selectedCounties[geoid];
  // Reset style on the map layer
  if (countyLayer) {
    countyLayer.eachLayer((layer) => {
      if (layer.feature.properties.GEOID === geoid) {
        countyLayer.resetStyle(layer);
      }
    });
  }
  updateMapStats();
  updateSelectedCountiesList();
  updateNextButton0();
}

function updateNextButton0() {
  const btn = document.getElementById('btnNext0');
  const count = Object.keys(selectedCounties).length;
  const pop = getTotalPopulation();
  btn.disabled = count === 0 || pop > MAX_POPULATION;
}

// ==========================================================================
// Panel 2 — Surveys
// ==========================================================================

function setupSurveys() {
  const chkNhts = document.getElementById('chkNhts');
  const chkCustom = document.getElementById('chkCustomSurvey');
  const cardNhts = document.getElementById('card-nhts');
  const cardCustom = document.getElementById('card-custom-survey');
  const customOptions = document.getElementById('customSurveyOptions');

  chkNhts.addEventListener('change', () => {
    cardNhts.classList.toggle('card--selected', chkNhts.checked);
  });

  chkCustom.addEventListener('change', () => {
    cardCustom.classList.toggle('card--selected', chkCustom.checked);
    customOptions.classList.toggle('hidden', !chkCustom.checked);
  });

  // File upload
  const fileUpload = document.getElementById('surveyFileUpload');
  const fileInput = document.getElementById('surveyFileInput');
  fileUpload.onclick = () => fileInput.click();
  fileInput.onchange = () => {
    const file = fileInput.files[0];
    if (file) {
      document.getElementById('surveyFileName').textContent = `Selected: ${file.name}`;
    }
  };
}

// ==========================================================================
// Panel 3 — Modes
// ==========================================================================

function setupModes() {
  const modes = ['car', 'bus', 'rail', 'walk', 'bike'];
  modes.forEach(mode => {
    const toggle = document.getElementById(`modeEnable-${mode}`);
    const card = document.getElementById(`modeCard-${mode}`);
    if (toggle && card) {
      toggle.addEventListener('change', () => {
        card.classList.toggle('mode-card--enabled', toggle.checked);
        card.classList.toggle('mode-card--disabled', !toggle.checked);
      });
    }
  });
}

// ==========================================================================
// Panel 4 — Transit
// ==========================================================================

function setupTransit() {
  const transitToggle = document.getElementById('transit-enabled');
  const transitSettings = document.getElementById('transitSettings');
  transitToggle.addEventListener('change', () => {
    transitSettings.style.display = transitToggle.checked ? 'block' : 'none';
  });
}

// ==========================================================================
// Panel 5 — Demand
// ==========================================================================

function setupDemand() {
  const targetSelect = document.getElementById('demand-target_plans');
  const customInput = document.getElementById('demand-target_plans_custom');
  targetSelect.addEventListener('change', () => {
    customInput.classList.toggle('hidden', targetSelect.value !== 'custom');
  });
}

// ==========================================================================
// Panel 6 — Time & Duration
// ==========================================================================

function buildDurationTable() {
  const tbody = document.getElementById('durationTable');
  tbody.innerHTML = '';

  Object.entries(DURATION_DEFAULTS).forEach(([activity, defaults]) => {
    const row = document.createElement('tr');
    row.style.borderBottom = '1px solid var(--color-gray-200)';
    row.innerHTML = `
      <td style="padding:8px 12px;font-weight:500;">${activity}</td>
      <td style="padding:8px 12px;">
        <input type="number" class="form-input" id="dur-${activity}-min"
               value="${defaults.min}" min="1" max="1440" style="width:80px;">
      </td>
      <td style="padding:8px 12px;">
        <input type="number" class="form-input" id="dur-${activity}-max"
               value="${defaults.max}" min="1" max="1440" style="width:80px;">
      </td>
      <td style="padding:8px 12px;">
        <select class="form-select" id="dur-${activity}-priority" style="width:120px;">
          <option value="1" ${defaults.trim_priority === 1 ? 'selected' : ''}>1 — Mandatory</option>
          <option value="2" ${defaults.trim_priority === 2 ? 'selected' : ''}>2 — Default</option>
          <option value="3" ${defaults.trim_priority === 3 ? 'selected' : ''}>3 — Discretionary</option>
        </select>
      </td>
    `;
    tbody.appendChild(row);
  });
}

// ==========================================================================
// Panel 7 — Simulation
// ==========================================================================

function setupSimulation() {
  const countsToggle = document.getElementById('counts-enabled');
  const countsSettings = document.getElementById('countsSettings');
  countsToggle.addEventListener('change', () => {
    countsSettings.style.display = countsToggle.checked ? 'block' : 'none';
  });

  const evalToggle = document.getElementById('eval-run');
  const evalSettings = document.getElementById('evalSettings');
  evalToggle.addEventListener('change', () => {
    evalSettings.style.display = evalToggle.checked ? 'block' : 'none';
  });
}

// ==========================================================================
// Slider value sync
// ==========================================================================

function setupSliders() {
  const sliders = [
    ['nhtsWeight',                   'nhtsWeightVal'],
    ['customSurveyWeight',           'customSurveyWeightVal'],
    ['mode-bus-blend_weight',        'mode-bus-blend_weight_val'],
    ['mode-rail-blend_weight',       'mode-rail-blend_weight_val'],
    ['demand-scaling_factor',        'demand-scaling_factor_val'],
    ['nonwork-trip_share',           'nonwork-trip_share_val'],
    ['purpose-shopping-blend_weight','purpose-shopping-blend_weight_val'],
    ['purpose-school-blend_weight',  'purpose-school-blend_weight_val'],
    ['purpose-social-blend_weight',  'purpose-social-blend_weight_val'],
    ['purpose-dining-blend_weight',  'purpose-dining-blend_weight_val'],
    ['purpose-other-blend_weight',   'purpose-other-blend_weight_val'],
    ['counts-fha_weight',            'counts-fha_weight_val'],
  ];

  sliders.forEach(([inputId, displayId]) => {
    const input = document.getElementById(inputId);
    const display = document.getElementById(displayId);
    if (input && display) {
      const update = () => {
        display.textContent = parseFloat(input.value).toFixed(
          input.step && input.step.includes('.') ? input.step.split('.')[1].length : 1
        );
        // Special: update scaling hint
        if (inputId === 'demand-scaling_factor') {
          const pct = (parseFloat(input.value) * 100).toFixed(1);
          document.getElementById('demand-scaling_hint').textContent =
            `Simulating ~${pct}% of the population`;
        }
      };
      input.addEventListener('input', update);
      update(); // init
    }
  });
}

// ==========================================================================
// Collapsible sections
// ==========================================================================

function toggleCollapsible(id) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle('collapsible--open');
}

// Wire up all collapsible triggers
function setupCollapsibles() {
  document.querySelectorAll('.collapsible__trigger').forEach(trigger => {
    const collapsible = trigger.closest('.collapsible');
    if (collapsible && !trigger.hasAttribute('onclick')) {
      trigger.addEventListener('click', () => {
        collapsible.classList.toggle('collapsible--open');
      });
    }
  });
}

// ==========================================================================
// Info Popup positioning (fixed, never clipped)
// ==========================================================================

function setupInfoPopups() {
  document.querySelectorAll('.info-icon').forEach(icon => {
    const popup = icon.querySelector('.info-popup');
    if (!popup) return;

    // Move popup to body so it's never clipped by overflow
    document.body.appendChild(popup);

    function showPopup() {
      const rect = icon.getBoundingClientRect();
      const popupW = 300;
      const margin = 8;

      // Horizontal: center on the icon, but clamp to viewport
      let left = rect.left + rect.width / 2 - popupW / 2;
      left = Math.max(margin, Math.min(left, window.innerWidth - popupW - margin));

      // Vertical: prefer above the icon; if not enough room, show below
      popup.style.left = left + 'px';
      popup.style.width = popupW + 'px';
      popup.classList.remove('info-popup--above', 'info-popup--below');

      // Temporarily show to measure height
      popup.style.visibility = 'hidden';
      popup.classList.add('info-popup--visible');
      const popupH = popup.offsetHeight;
      popup.classList.remove('info-popup--visible');
      popup.style.visibility = '';

      if (rect.top - popupH - margin > 0) {
        // Above
        popup.style.top = (rect.top - popupH - margin) + 'px';
        popup.classList.add('info-popup--above');
      } else {
        // Below
        popup.style.top = (rect.bottom + margin) + 'px';
        popup.classList.add('info-popup--below');
      }

      // Adjust arrow horizontal position to point at the icon center
      const arrowLeft = rect.left + rect.width / 2 - left;
      popup.style.setProperty('--arrow-left', arrowLeft + 'px');

      popup.classList.add('info-popup--visible');
    }

    function hidePopup() {
      popup.classList.remove('info-popup--visible');
    }

    icon.addEventListener('mouseenter', showPopup);
    icon.addEventListener('mouseleave', hidePopup);
    icon.addEventListener('focus', showPopup);
    icon.addEventListener('blur', hidePopup);
  });
}

// ==========================================================================
// Purpose card expand/collapse
// ==========================================================================

function togglePurposeCard(purpose) {
  const card = document.getElementById(`purposeCard-${purpose}`);
  if (card) card.classList.toggle('purpose-card--expanded');
}

// ==========================================================================
// Export Config
// ==========================================================================

function collectFormData() {
  const val = (id, fallback) => {
    const el = document.getElementById(id);
    if (!el) return fallback;
    if (el.type === 'checkbox') return el.checked;
    if (el.type === 'number' || el.type === 'range') return parseFloat(el.value);
    return el.value;
  };

  const counties = Object.keys(selectedCounties);

  // Surveys
  const surveys = [];
  if (document.getElementById('chkNhts').checked) {
    surveys.push({
      type: 'nhts',
      year: '2022',
      file: 'nhts/csv/tripv2pub.csv',
      weight: val('nhtsWeight', 1),
    });
  }
  if (document.getElementById('chkCustomSurvey').checked) {
    surveys.push({
      type: val('customSurveyType', 'tbi'),
      year: val('customSurveyYear', '2023'),
      file: '',  // will be set by server
      weight: val('customSurveyWeight', 0.5),
    });
  }

  // Modes
  const modes = {
    car: {
      enabled: val('modeEnable-car', true),
      matsim_mode: 'car',
      availability: 'universal',
      survey_rate: 'auto',
      config_rate: null,
      blend_weight: 0,
      share_adjustment: val('mode-car-share_adj', 0),
    },
    bus: {
      enabled: val('modeEnable-bus', true),
      matsim_mode: 'pt',
      availability: {
        type: 'gtfs',
        route_types: [3, 11],
        access_buffer_meters: val('mode-bus-access_buffer', 900),
      },
      survey_rate: 'auto',
      config_rate: val('mode-bus-config_rate', 0.09),
      blend_weight: val('mode-bus-blend_weight', 0.7),
      share_adjustment: 0,
    },
    rail: {
      enabled: val('modeEnable-rail', true),
      matsim_mode: 'pt',
      availability: {
        type: 'gtfs',
        route_types: [0, 1, 2, 12],
        access_buffer_meters: val('mode-rail-access_buffer', 1500),
      },
      survey_rate: 'auto',
      config_rate: val('mode-rail-config_rate', 0.06),
      blend_weight: val('mode-rail-blend_weight', 0.7),
      share_adjustment: 0,
    },
    walk: {
      enabled: val('modeEnable-walk', true),
      matsim_mode: 'walk',
      availability: {
        type: 'distance',
        max_distance_meters: val('mode-walk-max_distance', 2000),
      },
      survey_rate: 'auto',
      config_rate: null,
      blend_weight: 0,
      share_adjustment: 0,
    },
    bike: {
      enabled: val('modeEnable-bike', true),
      matsim_mode: 'bike',
      availability: {
        type: 'distance',
        max_distance_meters: val('mode-bike-max_distance', 8000),
      },
      survey_rate: 'auto',
      config_rate: null,
      blend_weight: 0,
      share_adjustment: 0,
    },
  };

  // Target plans
  let targetPlans = val('demand-target_plans', 'all');
  if (targetPlans === 'custom') {
    targetPlans = val('demand-target_plans_custom', 1000);
  }

  // Duration constraints
  const activityDurations = {};
  Object.keys(DURATION_DEFAULTS).forEach(activity => {
    activityDurations[activity] = {
      min_minutes: val(`dur-${activity}-min`, DURATION_DEFAULTS[activity].min),
      max_minutes: val(`dur-${activity}-max`, DURATION_DEFAULTS[activity].max),
      trim_priority: parseInt(document.getElementById(`dur-${activity}-priority`)?.value || DURATION_DEFAULTS[activity].trim_priority),
      blend_weight: 0,
    };
  });

  // Non-work purposes
  const purposes = ['Shopping', 'School', 'Social', 'Dining', 'Other'];
  const purposeKeyMap = {
    Shopping: 'shopping',
    School: 'school',
    Social: 'social',
    Dining: 'dining',
    Other: 'other',
  };

  const nonworkPurposes = {
    nonwork_trip_share: val('nonwork-trip_share', 0.8),
  };

  purposes.forEach(purpose => {
    const key = purposeKeyMap[purpose];
    nonworkPurposes[purpose] = {
      enabled: val(`purpose-${key}-enabled`, true),
      trip_generation: {
        survey_rate: 'auto',
        config_rate: val(`purpose-${key}-config_rate`, 0.2),
        blend_weight: val(`purpose-${key}-blend_weight`, 0.7),
      },
      od_matrix: {
        beta: val(`purpose-${key}-od_beta`, 2.0),
        alpha: 0.1,
      },
      poi_weighting: {
        enabled: true,
      },
    };
  });

  return {
    counties,
    surveys,
    modes,
    mode_choice: {
      method: 'survey_rates',
      fallback_mode: val('modeChoice-fallback', 'car'),
      chain_consistency: val('modeChoice-chainConsistency', true),
      min_samples_per_purpose: 30,
      max_chain_mode_retries: 10,
    },
    gtfs: {
      catalog_max_age_days: val('gtfs-catalog_max_age', 7),
      feed_max_age_days: val('gtfs-feed_max_age', 30),
    },
    plan_generation: {
      target_plans: targetPlans,
      scaling_factor: val('demand-scaling_factor', 0.1),
      work_scaling_multiplier: val('demand-work_scaling_mult', 1.0),
      random_seed: val('demand-random_seed', 42),
      skip_if_exists: true,
      supported_chain_types: ['home_work_home'],
      chain_sampling_method: 'generated',
      max_chain_retries: 10,
      num_processes: 30,
      default_mode: 'car',
    },
    chains: {
      home_boost_factor: 1.0,
      use_weighted_chains: true,
      max_length: null,
      min_length: val('chain-min_length', 3),
      max_work_activities: val('chain-max_work', 2),
      early_stop_exponent: val('chain-early_stop', 2.0),
    },
    od_matrix: {
      alpha: 0.1,
      beta: 1.5,
      max_iterations: 200,
      convergence_threshold: 0.03,
    },
    time_models: {
      kde_bandwidth: 'scott',
      max_time_retries: 10,
      max_duration_sample_attempts: 100,
    },
    duration_constraints: {
      activity_durations: activityDurations,
      trip_durations: {
        default: { min_minutes: 1, max_minutes: 180 },
      },
      max_travel_buffer_minutes: val('time-max_travel_buffer', 180),
      min_evening_home_minutes: 0,
      trim_jitter_minutes: val('time-trim_jitter', 15),
      departure_jitter_minutes: val('time-departure_jitter', 5),
    },
    poi_assignment: {
      initial_radius_m: val('poi-initial_radius', 2000),
      radius_increment_m: val('poi-radius_increment', 1000),
      max_poi_retries: val('poi-max_retries', 10),
    },
    matsim: {
      transit_network: val('transit-enabled', true),
      gtfs_sample_day: val('gtfs-sample_day', 'dayWithMostTrips'),
      pt2matsim: {
        numOfThreads: 4,
        candidateDistanceMultiplier: val('pt2m-candidateDistanceMultiplier', 1.6),
        maxLinkCandidateDistance: val('pt2m-maxLinkCandidateDistance', 90),
        maxTravelCostFactor: val('pt2m-maxTravelCostFactor', 5.0),
        nLinkThreshold: val('pt2m-nLinkThreshold', 6),
        travelCostType: val('pt2m-travelCostType', 'linkLength'),
        removeNotUsedStopFacilities: true,
        routingWithCandidateDistance: true,
        scheduleFreespeedModes: 'artificial',
      },
      configurable_params: {
        lastIteration: val('sim-iterations', 100),
        'qsim.flowCapacityFactor': val('sim-flow_capacity', 0.1),
        'qsim.storageCapacityFactor': val('sim-storage_capacity', 0.12),
        'linkStats.averageLinkStatsOverIterations': val('sim-linkstats_avg', 5),
      },
    },
    counts: {
      enabled: val('counts-enabled', true),
      rebuild: false,
      fha: {
        year: val('counts-fha_year', 2024),
        month: parseInt(document.getElementById('counts-fha_month')?.value || '7'),
        weight: val('counts-fha_weight', 1),
      },
      custom: {
        enabled: false,
        weight: 0,
      },
    },
    evaluation: {
      run_evaluation: val('eval-run', true),
      ground_truth_data_dir: 'data/evaluation',
      generate_spatial_maps: val('eval-spatial_maps', true),
      generate_per_device_reports: val('eval-per_device', false),
    },
    nonwork_purposes: nonworkPurposes,
  };
}

async function exportConfig() {
  const formData = collectFormData();

  try {
    const resp = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    });
    const config = await resp.json();

    // Download as file
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'config.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error('Export failed:', err);
    alert('Export failed. Check the console for details.');
  }
}

// ==========================================================================
// Utilities
// ==========================================================================

function formatNumber(n) {
  if (n == null) return '0';
  return n.toLocaleString('en-US');
}

// ==========================================================================
// Init
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  renderStepProgress();
  setupNavigation();
  initMap();
  setupSurveys();
  setupModes();
  setupTransit();
  setupDemand();
  buildDurationTable();
  setupSimulation();
  setupSliders();
  setupCollapsibles();
  setupInfoPopups();

  // Export button
  document.getElementById('btnExport').addEventListener('click', exportConfig);
});
