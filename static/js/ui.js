import { state, notify } from './state.js';

// ── Helpers ──────────────────────────────────────────────────────────────

function el(id) { return document.getElementById(id); }
function sign(v) { return v >= 0 ? '+' : ''; }
function fmt1(v) { return v.toFixed(1); }

function featureLabel(name) {
    const cfg = state.featuresConfig.find(c => c.name === name);
    return cfg ? cfg.label : name;
}

// ── Mode Toggle ──────────────────────────────────────────────────────────

export function renderModeToggle() {
    const teamBtn = el('mode-team');
    const indBtn = el('mode-individual');
    teamBtn.classList.toggle('active', state.mode === 'team');
    indBtn.classList.toggle('active', state.mode === 'individual');

    el('team-upload-section').classList.toggle('hidden', state.mode !== 'team');
    el('btn-max-all').classList.toggle('hidden', state.mode !== 'team');

    // Update button labels
    if (state.mode === 'team') {
        el('btn-reset').textContent = 'Reset to Baseline';
        el('btn-baseline').classList.add('hidden');
        el('sliders-caption').textContent = 'Move sliders to simulate uniform interventions across the team.';
    } else {
        el('btn-reset').textContent = 'Reset Defaults';
        el('btn-baseline').classList.remove('hidden');
        el('sliders-caption').textContent = 'Move sliders to explore different scenarios.';
    }
}

// ── Status Badges ────────────────────────────────────────────────────────

export function renderStatusBadges() {
    const container = el('status-badges');
    const currentSource = state.modelSources.find(s => s.id === state.modelSource);
    const mhqOk = currentSource?.mhq_loaded ?? false;
    const unprodOk = currentSource?.unprod_loaded ?? false;

    let html = '';
    html += badge('MHQ Model', mhqOk ? 'Loaded' : 'Missing', mhqOk ? 'loaded' : 'missing');
    html += badge('Productivity Model', unprodOk ? 'Loaded' : 'Missing', unprodOk ? 'loaded' : 'missing');

    if (state.mode === 'team') {
        const teamLoaded = state.teamData !== null;
        html += badge('Team Data',
            teamLoaded ? `${state.teamData.length} responses` : 'Not loaded',
            teamLoaded ? 'loaded' : 'neutral');
    } else {
        const baselineSet = state.baseline !== null;
        html += badge('Baseline', baselineSet ? 'Set' : 'Not set', baselineSet ? 'loaded' : 'neutral');
    }

    container.innerHTML = html;
}

function badge(label, text, type) {
    return `<div class="status-badge ${type}"><span class="badge-text">${text}</span><span class="text-gray-700 text-xs">${label}</span></div>`;
}

// ── Help Text ────────────────────────────────────────────────────────────

export function renderHelpText() {
    const container = el('help-text');
    if (state.mode === 'team') {
        container.innerHTML = `
            <p><strong>Step 1:</strong> Upload a CSV with one row per team member (11 features).</p>
            <p><strong>Step 2:</strong> Adjust sliders to simulate uniform interventions.</p>
            <p><strong>Step 3:</strong> Click MHQ, Productivity, or Balanced to optimize.</p>`;
    } else {
        container.innerHTML = `
            <p><strong>Step 1:</strong> Adjust sliders to reflect your current situation.</p>
            <p><strong>Step 2:</strong> Click "Set Baseline" to save your reference point.</p>
            <p><strong>Step 3:</strong> Move sliders to explore changes.</p>
            <p><strong>Step 4:</strong> Click an optimization button to find best improvements.</p>`;
    }
}

// ── Team Upload ──────────────────────────────────────────────────────────

export function renderTeamUpload() {
    const uploadArea = el('team-upload-area');
    const loadedInfo = el('team-loaded-info');

    if (state.teamData !== null) {
        uploadArea.classList.add('hidden');
        loadedInfo.classList.remove('hidden');
        el('team-loaded-text').textContent = `Team data loaded — ${state.teamData.length} complete responses`;
    } else {
        uploadArea.classList.remove('hidden');
        loadedInfo.classList.add('hidden');
    }
}

// ── Outcome Banner ───────────────────────────────────────────────────────

export function renderOutcomeBanner() {
    const isTeam = state.mode === 'team';
    const hasBaseline = isTeam ? state.teamBaseline !== null : state.baseline !== null;

    // Labels
    el('mhq-label').textContent = isTeam ? 'Avg. Mental Health Quotient (MHQ)' : 'Mental Health Quotient (MHQ)';
    el('unprod-label').textContent = isTeam ? 'Avg. Productive Days (per month)' : 'Productive Days (per month)';

    if (isTeam && state.teamData) {
        el('banner-subtitle').textContent = `Team of ${state.teamData.length} members — simulating uniform interventions.`;
    } else {
        el('banner-subtitle').textContent = 'Explore how work environment and lifestyle factors impact mental health and productivity.';
    }

    if (!hasBaseline) {
        el('mhq-value').innerHTML = '&mdash;';
        el('mhq-pct').textContent = '';
        el('mhq-range').textContent = '';
        el('unprod-value').innerHTML = '&mdash;';
        el('unprod-pct').textContent = '';
        el('unprod-range').textContent = '';
        el('banner-caption').textContent = isTeam
            ? 'Upload a CSV to set the team baseline.'
            : 'Set a baseline to see how changes in work factors impact mental health and productivity.';
        return;
    }

    let baseMhq, baseUnprod, curMhq, curUnprod;

    if (isTeam) {
        baseMhq = state.teamBaseline.mhq;
        baseUnprod = state.teamBaseline.unprod;
        curMhq = state.teamPrediction?.avg_mhq ?? baseMhq;
        curUnprod = state.teamPrediction?.avg_unproductive_days ?? baseUnprod;
    } else {
        baseMhq = state.baseline.mhq;
        baseUnprod = state.baseline.unproductive_days;
        curMhq = state.currentPrediction?.mhq ?? baseMhq;
        curUnprod = state.currentPrediction?.unproductive_days ?? baseUnprod;
    }

    const dMhq = curMhq - baseMhq;
    const dUnprod = curUnprod - baseUnprod;
    const pctMhq = baseMhq !== 0 ? (dMhq / baseMhq * 100) : 0;
    const pctUnprod = baseUnprod !== 0 ? (dUnprod / baseUnprod * 100) : 0;

    el('mhq-value').textContent = `${sign(dMhq)}${fmt1(dMhq)}`;
    el('mhq-pct').textContent = `(${sign(pctMhq)}${fmt1(pctMhq)}%)`;
    const dProd = -dUnprod;
    const pctProd = -pctUnprod;
    el('unprod-value').textContent = `${sign(dProd)}${fmt1(dProd)}`;
    el('unprod-pct').textContent = `(${sign(pctProd)}${fmt1(pctProd)}%)`;

    // Unproductive days card color
    const unprodCard = el('unprod-card');
    unprodCard.classList.toggle('warning', curUnprod > 3);

    // Team min/max ranges
    if (isTeam && state.teamPrediction && state.teamBaseline.individual_mhq) {
        const baseMhqInd = state.teamBaseline.individual_mhq;
        const baseUnprodInd = state.teamBaseline.individual_unprod;
        const curMhqInd = state.teamPrediction.individual_mhq;
        const curUnprodInd = state.teamPrediction.individual_unproductive_days;

        const mhqDeltas = curMhqInd.map((v, i) => v - baseMhqInd[i]);
        const prodDeltas = curUnprodInd.map((v, i) => -(v - baseUnprodInd[i]));

        el('mhq-range').textContent = `Range: ${sign(Math.min(...mhqDeltas))}${fmt1(Math.min(...mhqDeltas))} to ${sign(Math.max(...mhqDeltas))}${fmt1(Math.max(...mhqDeltas))}`;
        el('unprod-range').textContent = `Range: ${sign(Math.min(...prodDeltas))}${fmt1(Math.min(...prodDeltas))} to ${sign(Math.max(...prodDeltas))}${fmt1(Math.max(...prodDeltas))}`;
    } else {
        el('mhq-range').textContent = '';
        el('unprod-range').textContent = '';
    }

    if (isTeam) {
        el('banner-caption').textContent = `Team of ${state.teamData.length} members. Showing average change from baseline.`;
    } else {
        el('banner-caption').textContent = 'Showing change from your baseline. Positive = improvement for both MHQ and Productive Days.';
    }
}

// ── Sliders ──────────────────────────────────────────────────────────────

export function renderSliders(onSliderChange) {
    renderSliderGroup('work', el('work-sliders'), onSliderChange);
    renderSliderGroup('lifestyle', el('lifestyle-sliders'), onSliderChange);
}

function renderSliderGroup(category, container, onSliderChange) {
    const features = state.featuresConfig.filter(c => c.category === category);
    container.innerHTML = '';

    for (const cfg of features) {
        const isHighlighted = state.highlightedLevers.has(cfg.name);
        const value = state.sliderValues[cfg.name] ?? cfg.default;
        const rawAvg = state.teamRawAverages?.[cfg.name];

        const group = document.createElement('div');
        group.className = `slider-group ${isHighlighted ? 'highlighted' : ''}`;

        if (cfg.name === 'exercise_freq_ord' || cfg.name === 'UPF_freq_ord') {
            const labels = cfg.name === 'exercise_freq_ord' ? state.exerciseLabels : state.upfLabels;
            const captionText = cfg.name === 'exercise_freq_ord'
                ? (rawAvg != null ? `Baseline avg: ${rawAvg.toFixed(1)} · Higher = more frequent exercise` : 'Higher = more frequent exercise')
                : (rawAvg != null ? `Baseline avg: ${rawAvg.toFixed(1)} · Higher = healthier (less UPF)` : 'Higher = healthier (less UPF)');

            group.className += ' select-slider';
            group.innerHTML = `
                ${isHighlighted ? `<p class="text-sm font-bold text-green-600 mb-0">&#9679; ${cfg.label}</p>` : ''}
                <label class="block text-sm font-semibold text-gray-700 ${isHighlighted ? 'hidden' : ''}">${cfg.label}</label>
                <select data-feature="${cfg.name}" class="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm">
                    ${labels.map((lbl, i) => `<option value="${i + 1}" ${(i + 1) === value ? 'selected' : ''}>${lbl}</option>`).join('')}
                </select>
                <p class="slider-caption">${captionText}</p>`;

            group.querySelector('select').addEventListener('change', (e) => {
                state.sliderValues[cfg.name] = parseInt(e.target.value);
                onSliderChange();
            });
        } else {
            const captionText = rawAvg != null ? `Baseline avg: ${rawAvg.toFixed(1)}` : '';
            group.innerHTML = `
                ${isHighlighted ? `<p class="text-sm font-bold text-green-600 mb-0">&#9679; ${cfg.label}</p>` : ''}
                <div class="flex items-center justify-between">
                    <label class="text-sm font-semibold text-gray-700 ${isHighlighted ? 'hidden' : ''}">${cfg.label}</label>
                    <span class="slider-value">${value}</span>
                </div>
                <input type="range" data-feature="${cfg.name}" min="${cfg.min}" max="${cfg.max}" step="${cfg.step}" value="${value}">
                ${captionText ? `<p class="slider-caption">${captionText}</p>` : ''}`;

            const slider = group.querySelector('input[type="range"]');
            const valSpan = group.querySelector('.slider-value');
            slider.addEventListener('input', (e) => {
                const v = parseInt(e.target.value);
                state.sliderValues[cfg.name] = v;
                valSpan.textContent = v;
                onSliderChange();
            });
        }

        container.appendChild(group);
    }
}

// ── Optimization Result ──────────────────────────────────────────────────

export function renderOptimizationResult() {
    const container = el('optimization-result');

    if (!state.optimizationResult || !state.optimizationGoal) {
        container.classList.add('hidden');
        container.innerHTML = '';
        return;
    }

    const result = state.optimizationResult;
    const goal = state.optimizationGoal;
    const isTeam = state.mode === 'team';
    const prefix = isTeam ? 'Avg. ' : '';

    const goalClass = `goal-${goal}`;
    const goalColor = { mhq: '#8b5cf6', productivity: '#f59e0b', balanced: '#10b981' }[goal] || '#10b981';
    const goalLabel = { mhq: 'MHQ', productivity: 'Productivity', balanced: 'Balanced' }[goal] || goal;

    const leverLabels = result.levers.map(l => `<strong>${featureLabel(l)}</strong>`).join(', ');

    let improvements = [];
    if (goal === 'mhq' || goal === 'balanced') {
        improvements.push(`<span style="color:#10b981;font-weight:600">${prefix}MHQ ${sign(result.delta_mhq)}${fmt1(result.delta_mhq)}</span>`);
    }
    if (goal === 'productivity' || goal === 'balanced') {
        improvements.push(`<span style="color:#f59e0b;font-weight:600">${prefix}Days ${sign(result.delta_unprod)}${fmt1(result.delta_unprod)}</span>`);
    }

    container.classList.remove('hidden');
    container.innerHTML = `
        <div class="opt-result ${goalClass}">
            <div class="mb-2">
                <span style="color:${goalColor};font-weight:700;font-size:1.1rem;">Optimized for ${goalLabel}</span>
            </div>
            <div class="mb-2">
                <span class="text-sm font-semibold" style="color:#065f46">Focus on: </span>
                <span class="text-sm" style="color:#047857">${leverLabels}</span>
            </div>
            <div>
                <span class="text-sm font-semibold" style="color:#065f46">Expected: </span>
                <span class="text-sm">${improvements.join(' &bull; ')}</span>
            </div>
        </div>`;
}

// ── Action Bar State ─────────────────────────────────────────────────────

export function renderActionBar() {
    const canOptimize = state.mode === 'team' ? state.teamData !== null : state.baseline !== null;

    el('opt-mhq').disabled = !canOptimize;
    el('opt-productivity').disabled = !canOptimize;
    el('opt-balanced').disabled = !canOptimize;

    if (!canOptimize) {
        el('optimize-hint').textContent = state.mode === 'team'
            ? 'Upload a team CSV to enable optimization'
            : 'Set a baseline first to enable optimization';
        el('optimize-hint').classList.remove('hidden');
    } else {
        el('optimize-hint').classList.add('hidden');
    }

    // Baseline button
    if (state.mode === 'individual') {
        const baselineSet = state.baseline !== null;
        el('btn-baseline').textContent = baselineSet ? 'Clear Baseline' : 'Set Baseline';
    }
}

// ── Model Source Dropdown ────────────────────────────────────────────────

export function renderModelDropdown() {
    const select = el('model-source');
    select.innerHTML = state.modelSources.map(s =>
        `<option value="${s.id}" ${s.id === state.modelSource ? 'selected' : ''}>${s.label}</option>`
    ).join('');
}

// ── Spinner ──────────────────────────────────────────────────────────────

export function showSpinner(text = 'Optimizing...') {
    el('spinner').classList.remove('hidden');
    el('spinner-text').textContent = text;
}

export function hideSpinner() {
    el('spinner').classList.add('hidden');
}

// ── Render All ───────────────────────────────────────────────────────────

export function renderAll(onSliderChange) {
    renderModeToggle();
    renderModelDropdown();
    renderStatusBadges();
    renderHelpText();
    renderTeamUpload();
    renderOutcomeBanner();
    renderActionBar();
    renderOptimizationResult();
    renderSliders(onSliderChange);
}
