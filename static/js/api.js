const API = '/api';

export async function fetchFeaturesConfig() {
    const res = await fetch(`${API}/features-config`);
    return res.json();
}

export async function fetchModels() {
    const res = await fetch(`${API}/models`);
    return res.json();
}

export async function predict(features, modelSource) {
    const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features, model_source: modelSource }),
    });
    return res.json();
}

export async function predictBatch(teamData, featureNames, sliderValues, teamAverages, modelSource) {
    const res = await fetch(`${API}/predict-batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            team_data: teamData,
            feature_names: featureNames,
            slider_values: sliderValues,
            team_averages: teamAverages,
            model_source: modelSource,
        }),
    });
    return res.json();
}

export async function uploadCsv(file, modelSource) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API}/upload-csv?model_source=${modelSource}`, {
        method: 'POST',
        body: formData,
    });
    return res.json();
}

export async function optimize(params) {
    const res = await fetch(`${API}/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    });
    return res.json();
}
