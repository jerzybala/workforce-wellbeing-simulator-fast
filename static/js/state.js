// Reactive state store with subscriber pattern
const _subscribers = [];

export const state = {
    mode: 'team',  // 'individual' | 'team'
    modelSource: 'models_west',

    // Config (loaded from API)
    featuresConfig: [],
    exerciseLabels: [],
    upfLabels: [],
    categories: {},
    modelSources: [],

    // Slider values (current)
    sliderValues: {},

    // Individual mode
    baseline: null,  // {features, mhq, unproductive_days}
    currentPrediction: null,  // {mhq, unproductive_days}

    // Team mode
    teamData: null,  // 2D array
    featureNames: null,
    teamAverages: null,
    teamRawAverages: null,
    teamBaseline: null,  // {mhq, unprod, teamq, individual_mhq, individual_unprod}
    teamPrediction: null,  // {avg_mhq, avg_unproductive_days, teamq, individual_mhq, individual_unproductive_days}

    // Optimization
    optimizationGoal: null,
    optimizationResult: null,
    highlightedLevers: new Set(),
    optimizationK: 2,
    isOptimizing: false,

    // Sensitivity analysis
    sensitivityResult: null,  // {features: [{name, current, curve, slope_mhq, ...}]}
    sensitivityMetric: 'mhq',  // 'mhq' | 'unprod'
};

export function subscribe(fn) {
    _subscribers.push(fn);
}

export function notify() {
    for (const fn of _subscribers) fn(state);
}

export function resetSliders() {
    state.sliderValues = {};
    for (const cfg of state.featuresConfig) {
        const defaultVal = state.mode === 'team' && state.teamAverages
            ? state.teamAverages[cfg.name]
            : cfg.default;
        state.sliderValues[cfg.name] = defaultVal;
    }
    state.optimizationGoal = null;
    state.optimizationResult = null;
    state.highlightedLevers = new Set();
}

export function clearTeamData() {
    state.teamData = null;
    state.featureNames = null;
    state.teamAverages = null;
    state.teamRawAverages = null;
    state.teamBaseline = null;
    state.teamPrediction = null;
    state.optimizationGoal = null;
    state.optimizationResult = null;
    state.sensitivityResult = null;
    state.highlightedLevers = new Set();
    resetSliders();
}

export function clearBaseline() {
    state.baseline = null;
    state.optimizationGoal = null;
    state.optimizationResult = null;
    state.sensitivityResult = null;
    state.highlightedLevers = new Set();
}
