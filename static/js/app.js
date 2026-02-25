import { fetchFeaturesConfig, fetchModels, predict, predictBatch, uploadCsv, optimize, fetchSensitivity } from './api.js';
import { state, resetSliders, clearTeamData, clearBaseline } from './state.js';
import {
    renderAll, renderOutcomeBanner, renderSliders, renderStatusBadges,
    renderModeToggle, renderHelpText, renderTeamUpload, renderActionBar,
    renderOptimizationResult, renderSensitivityResult, renderModelDropdown, showSpinner, hideSpinner,
} from './ui.js';
import { generateReport } from './report.js';

// ── Debounce helper ──────────────────────────────────────────────────────

let _debounceTimer = null;
function debounce(fn, ms = 80) {
    clearTimeout(_debounceTimer);
    _debounceTimer = setTimeout(fn, ms);
}

// ── Prediction ───────────────────────────────────────────────────────────

async function runPrediction() {
    if (state.mode === 'team' && state.teamData && state.teamAverages) {
        const result = await predictBatch(
            state.teamData, state.featureNames,
            state.sliderValues, state.teamAverages,
            state.modelSource,
        );
        state.teamPrediction = result;
    } else if (state.mode === 'individual') {
        const result = await predict(state.sliderValues, state.modelSource);
        state.currentPrediction = result;
    }
    renderOutcomeBanner();
    renderTeamUpload();
}

function onSliderChange() {
    debounce(() => runPrediction());
}

// ── Initialization ───────────────────────────────────────────────────────

async function init() {
    // Load config from API
    const configData = await fetchFeaturesConfig();
    state.featuresConfig = configData.features;
    state.exerciseLabels = configData.exercise_labels;
    state.upfLabels = configData.upf_labels;
    state.categories = configData.categories;

    const modelsData = await fetchModels();
    state.modelSources = modelsData.sources;
    state.modelSource = modelsData.default;

    // Initialize slider defaults
    resetSliders();

    // Initial render
    renderAll(onSliderChange);

    // Initial prediction
    await runPrediction();

    // Wire events
    wireEvents();
}

// ── Event Wiring ─────────────────────────────────────────────────────────

function wireEvents() {
    // Mode toggle
    document.getElementById('mode-team').addEventListener('click', () => {
        if (state.mode === 'team') return;
        state.mode = 'team';
        state.baseline = null;
        state.currentPrediction = null;
        state.optimizationGoal = null;
        state.optimizationResult = null;
        state.sensitivityResult = null;
        state.highlightedLevers = new Set();
        resetSliders();
        renderAll(onSliderChange);
        runPrediction();
    });

    document.getElementById('mode-individual').addEventListener('click', () => {
        if (state.mode === 'individual') return;
        state.mode = 'individual';
        state.optimizationGoal = null;
        state.optimizationResult = null;
        state.sensitivityResult = null;
        state.highlightedLevers = new Set();
        resetSliders();
        renderAll(onSliderChange);
        runPrediction();
    });

    // Model source
    document.getElementById('model-source').addEventListener('change', async (e) => {
        state.modelSource = e.target.value;
        renderStatusBadges();

        // Re-compute team baseline if team data loaded
        if (state.mode === 'team' && state.teamData) {
            // Re-upload to get new baseline with new model
            // For simplicity, just re-predict
            await runPrediction();
        } else {
            await runPrediction();
        }
    });

    // CSV upload — shared handler
    async function handleCsvFile(file) {
        if (!file) return;

        const filenameEl = document.getElementById('csv-filename');
        if (filenameEl) filenameEl.textContent = file.name;

        showSpinner('Processing CSV...');
        const result = await uploadCsv(file, state.modelSource);
        hideSpinner();

        if (result.error) {
            alert(result.error);
            return;
        }

        state.teamData = result.team_data;
        state.featureNames = result.feature_names;
        state.teamAverages = result.team_averages;
        state.teamRawAverages = result.team_raw_averages;
        state.teamBaseline = {
            mhq: result.baseline_mhq,
            unprod: result.baseline_unproductive_days,
            teamq: result.baseline_teamq,
            teamp: result.baseline_teamp,
            individual_mhq: result.baseline_individual_mhq,
            individual_unprod: result.baseline_individual_unproductive_days,
        };

        for (const name of state.featureNames) {
            state.sliderValues[name] = state.teamAverages[name];
        }

        state.optimizationGoal = null;
        state.optimizationResult = null;
        state.highlightedLevers = new Set();

        renderAll(onSliderChange);
        await runPrediction();
    }

    document.getElementById('csv-upload').addEventListener('change', (e) => {
        handleCsvFile(e.target.files[0]);
    });

    // Drag-and-drop on upload area
    const dropZone = document.getElementById('team-upload-area');
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-blue-500', 'bg-blue-50');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-blue-500', 'bg-blue-50');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-blue-500', 'bg-blue-50');
        const file = e.dataTransfer.files[0];
        if (!file) return;
        const name = file.name.toLowerCase();
        if (name.endsWith('.jpg') || name.endsWith('.jpeg') || name.endsWith('.png') || name.endsWith('.gif') || name.endsWith('.pdf')) {
            alert(`"${file.name}" is not a CSV file. Please drop a .csv file.`);
            return;
        }
        handleCsvFile(file);
    });

    // Clear team
    document.getElementById('clear-team-btn').addEventListener('click', () => {
        clearTeamData();
        document.getElementById('csv-upload').value = '';
        renderAll(onSliderChange);
        renderOutcomeBanner();
    });

    // Generate report
    document.getElementById('btn-report').addEventListener('click', () => {
        generateReport();
    });

    // Reset
    document.getElementById('btn-reset').addEventListener('click', () => {
        if (state.mode === 'team' && state.teamAverages) {
            for (const name of state.featureNames) {
                state.sliderValues[name] = state.teamAverages[name];
            }
        } else {
            resetSliders();
        }
        state.optimizationGoal = null;
        state.optimizationResult = null;
        state.sensitivityResult = null;
        state.highlightedLevers = new Set();
        renderAll(onSliderChange);
        runPrediction();
    });

    // Baseline (individual mode)
    document.getElementById('btn-baseline').addEventListener('click', async () => {
        if (state.baseline !== null) {
            clearBaseline();
        } else {
            const pred = state.currentPrediction || await predict(state.sliderValues, state.modelSource);
            state.baseline = {
                features: { ...state.sliderValues },
                mhq: pred.mhq,
                unproductive_days: pred.unproductive_days,
            };
        }
        renderAll(onSliderChange);
        renderOutcomeBanner();
    });

    // Max All (team mode)
    document.getElementById('btn-max-all').addEventListener('click', () => {
        for (const cfg of state.featuresConfig) {
            state.sliderValues[cfg.name] = cfg.max;
        }
        renderSliders(onSliderChange);
        runPrediction();
    });

    // Optimization buttons
    for (const [btnId, goal] of [['opt-mhq', 'mhq'], ['opt-productivity', 'productivity'], ['opt-balanced', 'balanced']]) {
        document.getElementById(btnId).addEventListener('click', async () => {
            const k = parseInt(document.getElementById('optimize-k').value);

            // Block if all features are already at maximum — nothing to improve
            const allAtMax = state.featuresConfig.every(cfg => state.sliderValues[cfg.name] >= cfg.max);
            if (allAtMax) {
                const hint = document.getElementById('optimize-hint');
                hint.textContent = 'All factors are at maximum — no room to optimize. Reset or adjust sliders first.';
                hint.classList.remove('hidden');
                return;
            }

            state.isOptimizing = true;
            showSpinner(`Searching for best ${k} inputs to optimize for ${goal}...`);

            const params = {
                mode: state.mode,
                current_inputs: state.sliderValues,
                model_source: state.modelSource,
                k: k,
                goal: goal,
                team_data: state.mode === 'team' ? state.teamData : null,
                team_averages: state.mode === 'team' ? state.teamAverages : null,
            };

            const result = await optimize(params);
            hideSpinner();
            state.isOptimizing = false;

            if (result.top_results && result.top_results.length > 0) {
                const best = result.top_results[0];
                state.optimizationGoal = goal;
                state.optimizationResult = best;
                state.highlightedLevers = new Set(best.levers);
            } else {
                state.optimizationGoal = null;
                state.optimizationResult = null;
                state.highlightedLevers = new Set();
            }

            renderOptimizationResult();
            renderSliders(onSliderChange);
            renderActionBar();
        });
    }

    // Sensitivity button
    document.getElementById('btn-sensitivity').addEventListener('click', async () => {
        showSpinner('Computing sensitivity analysis...');

        const params = {
            mode: state.mode,
            current_inputs: state.sliderValues,
            model_source: state.modelSource,
            team_data: state.mode === 'team' ? state.teamData : null,
            team_averages: state.mode === 'team' ? state.teamAverages : null,
        };

        const result = await fetchSensitivity(params);
        hideSpinner();

        state.sensitivityResult = result;
        renderSensitivityResult();
    });
}

// ── Start ────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', init);
