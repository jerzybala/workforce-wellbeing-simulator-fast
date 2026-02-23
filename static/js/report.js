import { state } from './state.js';
import { fetchShapWeights, fetchSensitivity } from './api.js';

function fmt1(v) { return v.toFixed(1); }
function sign(v) { return v >= 0 ? '+' : ''; }

function featureLabel(name) {
    const cfg = state.featuresConfig.find(c => c.name === name);
    return cfg ? cfg.label : name;
}

function categoryLabel(cat) {
    return state.categories[cat] || cat;
}

function median(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export async function generateReport() {
    const [shapData, sensData] = await Promise.all([
        fetchShapWeights(state.modelSource),
        fetchSensitivity({
            mode: state.mode,
            current_inputs: state.sliderValues,
            model_source: state.modelSource,
            team_data: state.mode === 'team' ? state.teamData : null,
            team_averages: state.mode === 'team' ? state.teamAverages : null,
        }),
    ]);
    const shapWeights = shapData.weights || {};
    const sensFeatures = sensData.features || [];

    const now = new Date();
    const dateStr = now.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
    const fileDate = now.toISOString().slice(0, 10);

    const modelLabel = state.modelSources.find(s => s.id === state.modelSource)?.label || state.modelSource;
    const teamSize = state.teamData?.length || 0;

    const baseTeamq = state.teamBaseline?.teamq ?? 0;
    const baseMhq = state.teamBaseline?.mhq ?? 0;
    const baseUnprod = state.teamBaseline?.unprod ?? 0;

    // Check if sliders have changed from baseline
    const changedFeatures = [];
    for (const cfg of state.featuresConfig) {
        const cur = state.sliderValues[cfg.name];
        const base = state.teamAverages?.[cfg.name];
        if (cur !== base) {
            changedFeatures.push({ name: cfg.name, label: cfg.label, from: base, to: cur, category: cfg.category });
        }
    }
    const hasScenario = changedFeatures.length > 0 && state.teamPrediction;
    const hasOptimization = state.optimizationResult && state.optimizationGoal;

    // Sort features by SHAP weight descending
    const sortedFeatures = [...state.featuresConfig].sort((a, b) => {
        return (shapWeights[b.name] || 0) - (shapWeights[a.name] || 0);
    });
    const totalShap = Object.values(shapWeights).reduce((s, v) => s + v, 0);

    // Individual distributions
    const indMhq = state.teamBaseline?.individual_mhq || [];
    const indUnprod = state.teamBaseline?.individual_unprod || [];

    let html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Team Wellbeing Report - ${fileDate}</title>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #1a1a1a; background: #fff; padding: 48px; line-height: 1.65; font-size: 14px; }
    .container { max-width: 760px; margin: 0 auto; }
    .header { border-bottom: 2px solid #1a1a1a; padding-bottom: 20px; margin-bottom: 32px; }
    .header h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; color: #1a1a1a; }
    .header-meta { font-size: 0.8rem; color: #666; margin-top: 6px; }
    .header-meta span { margin-right: 16px; }
    .section { margin-bottom: 32px; }
    .section h2 { font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #1a1a1a; margin-bottom: 14px; padding-bottom: 6px; border-bottom: 1px solid #d4d4d4; }
    .overview-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: #d4d4d4; border: 1px solid #d4d4d4; }
    .metric-card { background: #fff; padding: 18px 16px; text-align: center; }
    .metric-card .label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #888; margin-bottom: 6px; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1a1a1a; }
    .metric-card .sub { font-size: 0.75rem; color: #888; margin-top: 4px; }
    .teamq-bar-wrap { margin-top: 8px; width: 100%; height: 6px; background: #e5e5e5; border-radius: 3px; overflow: hidden; }
    .teamq-bar-inner { height: 100%; border-radius: 3px; background: #555; }
    table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    thead th { text-align: left; font-weight: 600; color: #888; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; padding: 6px 10px; border-bottom: 1px solid #d4d4d4; }
    tbody td { padding: 8px 10px; border-bottom: 1px solid #eee; color: #333; }
    tbody tr:last-child td { border-bottom: none; }
    .bar-cell { width: 100px; }
    .bar-track { width: 100%; height: 5px; background: #e5e5e5; border-radius: 2px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 2px; background: #888; }
    .importance-pct { font-size: 0.78rem; color: #555; }
    .cat-label { font-size: 0.7rem; color: #666; }
    .delta { font-weight: 600; }
    .delta.positive { color: #1a7a3a; }
    .delta.negative { color: #b91c1c; }
    .rec-box { border: 1px solid #d4d4d4; border-left: 3px solid #1a1a1a; padding: 16px 20px; margin-top: 4px; }
    .rec-box .goal { font-weight: 700; font-size: 0.9rem; color: #1a1a1a; margin-bottom: 8px; }
    .sens-row { display: flex; align-items: center; gap: 14px; padding: 10px 0; border-bottom: 1px solid #eee; }
    .sens-row:last-child { border-bottom: none; }
    .sens-rank { font-size: 0.82rem; font-weight: 700; color: #555; min-width: 22px; }
    .sens-info { flex: 1; min-width: 0; }
    .sens-info strong { font-size: 0.82rem; color: #1a1a1a; }
    .sens-info .cat { font-size: 0.7rem; color: #888; }
    .sens-chart { flex-shrink: 0; }
    .sens-vals { text-align: right; min-width: 140px; }
    .sens-vals .slope { font-size: 0.82rem; font-weight: 700; }
    .sens-vals .slope.pos { color: #1a7a3a; }
    .sens-vals .slope.neg { color: #b91c1c; }
    .sens-vals .totals { font-size: 0.72rem; color: #888; }
    .sens-vals .totals .pos { color: #1a7a3a; font-weight: 600; }
    .sens-vals .totals .neg { color: #b91c1c; font-weight: 600; }
    .footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid #d4d4d4; text-align: center; font-size: 0.72rem; color: #999; }
    @media print {
        body { padding: 24px; }
    }
</style>
</head>
<body>
<div class="container">

<!-- Header -->
<div class="header">
    <h1>Team Wellbeing &amp; Productivity Report</h1>
    <div class="header-meta">
        <span>${dateStr}</span>
        <span>Model: ${esc(modelLabel)}</span>
        <span>${teamSize} team members</span>
    </div>
</div>

<!-- Team Overview -->
<div class="section">
    <h2>Team Overview</h2>
    <div class="overview-grid">
        <div class="metric-card">
            <div class="label">TeamQ Score</div>
            <div class="value">${fmt1(baseTeamq)}</div>
            <div class="sub">out of 100</div>
            <div class="teamq-bar-wrap"><div class="teamq-bar-inner" style="width:${Math.max(0, Math.min(100, baseTeamq))}%"></div></div>
        </div>
        <div class="metric-card">
            <div class="label">Baseline Avg. MHQ</div>
            <div class="value">${fmt1(baseMhq)}</div>
            <div class="sub">Mental Health Quotient</div>
        </div>
        <div class="metric-card">
            <div class="label">Baseline Avg. Productive Days</div>
            <div class="value">${fmt1(22 - baseUnprod)}</div>
            <div class="sub">per month (of 22)</div>
        </div>
    </div>
</div>

<!-- Individual Distribution -->
${indMhq.length > 0 ? `
<div class="section">
    <h2>Individual Distribution</h2>
    <table>
        <thead><tr><th>Metric</th><th>Min</th><th>Median</th><th>Max</th><th>Range</th></tr></thead>
        <tbody>
        <tr>
            <td><strong>MHQ</strong></td>
            <td>${fmt1(Math.min(...indMhq))}</td>
            <td>${fmt1(median(indMhq))}</td>
            <td>${fmt1(Math.max(...indMhq))}</td>
            <td>${fmt1(Math.max(...indMhq) - Math.min(...indMhq))}</td>
        </tr>
        <tr>
            <td><strong>Productive Days</strong></td>
            <td>${fmt1(22 - Math.max(...indUnprod))}</td>
            <td>${fmt1(22 - median(indUnprod))}</td>
            <td>${fmt1(22 - Math.min(...indUnprod))}</td>
            <td>${fmt1(Math.max(...indUnprod) - Math.min(...indUnprod))}</td>
        </tr>
        </tbody>
    </table>
</div>` : ''}

<!-- Factor Breakdown -->
<div class="section">
    <h2>Factor Breakdown</h2>
    <table>
        <thead><tr><th>#</th><th>Factor</th><th>Category</th><th>Team Avg</th><th>Scale</th><th class="bar-cell">Level</th><th>Importance</th></tr></thead>
        <tbody>
        ${sortedFeatures.map((cfg, i) => {
            const avg = state.teamRawAverages?.[cfg.name] ?? cfg.default;
            const pct = ((avg - cfg.min) / (cfg.max - cfg.min)) * 100;
            const w = shapWeights[cfg.name] || 0;
            const impPct = totalShap > 0 ? (w / totalShap * 100) : 0;
            const cat = cfg.category;
            return `<tr>
                <td>${i + 1}</td>
                <td><strong>${esc(cfg.label)}</strong></td>
                <td><span class="cat-label">${categoryLabel(cat)}</span></td>
                <td>${fmt1(avg)}</td>
                <td>${cfg.min}&ndash;${cfg.max}</td>
                <td class="bar-cell"><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div></td>
                <td><span class="importance-pct">${fmt1(impPct)}%</span></td>
            </tr>`;
        }).join('')}
        </tbody>
    </table>
</div>`;

    // Current Scenario (conditional)
    if (hasScenario) {
        const curMhq = state.teamPrediction.avg_mhq;
        const curUnprod = state.teamPrediction.avg_unproductive_days;
        const curTeamq = state.teamPrediction.teamq;
        const dMhq = curMhq - baseMhq;
        const dUnprod = curUnprod - baseUnprod;
        const dTeamq = curTeamq - baseTeamq;

        html += `
<!-- Simulated Scenario -->
<div class="section">
    <h2>Simulated Scenario</h2>
    <p style="font-size:0.82rem;color:#666;margin-bottom:12px;">${changedFeatures.length} factor(s) adjusted from baseline.</p>
    <table>
        <thead><tr><th>Factor</th><th>Baseline</th><th>Adjusted</th><th>Change</th></tr></thead>
        <tbody>
        ${changedFeatures.map(f => {
            const delta = f.to - f.from;
            const cls = delta > 0 ? 'positive' : delta < 0 ? 'negative' : '';
            return `<tr>
                <td><strong>${esc(f.label)}</strong></td>
                <td>${f.from}</td>
                <td>${f.to}</td>
                <td><span class="delta ${cls}">${sign(delta)}${delta}</span></td>
            </tr>`;
        }).join('')}
        </tbody>
    </table>
    <div style="margin-top:16px;">
        <div class="overview-grid">
            <div class="metric-card">
                <div class="label">TeamQ</div>
                <div class="value">${fmt1(curTeamq)}</div>
                <div class="sub"><span class="delta ${dTeamq >= 0 ? 'positive' : 'negative'}">${sign(dTeamq)}${fmt1(dTeamq)}</span> from baseline</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg. MHQ</div>
                <div class="value">${fmt1(curMhq)}</div>
                <div class="sub"><span class="delta ${dMhq >= 0 ? 'positive' : 'negative'}">${sign(dMhq)}${fmt1(dMhq)}</span> from baseline</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg. Productive Days</div>
                <div class="value">${fmt1(22 - curUnprod)}</div>
                <div class="sub"><span class="delta ${-dUnprod >= 0 ? 'positive' : 'negative'}">${sign(-dUnprod)}${fmt1(-dUnprod)}</span> from baseline</div>
            </div>
        </div>
    </div>
</div>`;
    }

    // Optimization Recommendation (conditional)
    if (hasOptimization) {
        const opt = state.optimizationResult;
        const goalLabel = { mhq: 'MHQ (Mental Health)', productivity: 'Productivity', balanced: 'Balanced' }[state.optimizationGoal] || state.optimizationGoal;

        html += `
<!-- Optimization Recommendation -->
<div class="section">
    <h2>Optimization Recommendation</h2>
    <div class="rec-box">
        <div class="goal">Optimized for ${esc(goalLabel)} (with max factors)</div>
        <p style="font-size:0.82rem;margin-bottom:6px;"><strong>Focus on:</strong> ${opt.levers.map(l => esc(featureLabel(l))).join(', ')}</p>
        <p style="font-size:0.82rem;"><strong>Expected improvement:</strong>
            MHQ <span class="delta ${opt.delta_mhq >= 0 ? 'positive' : 'negative'}">${sign(opt.delta_mhq)}${fmt1(opt.delta_mhq)}</span>
            &middot;
            Productive Days <span class="delta ${-opt.delta_unprod >= 0 ? 'positive' : 'negative'}">${sign(-opt.delta_unprod)}${fmt1(-opt.delta_unprod)}</span>
        </p>
    </div>
</div>`;
    }

    // Sensitivity Analysis section
    if (sensFeatures.length > 0) {
        html += `
<!-- Factor Sensitivity Analysis -->
<div class="section">
    <h2>Factor Sensitivity Analysis</h2>
    <p style="font-size:0.82rem;color:#666;margin-bottom:14px;">Each factor swept independently from minimum to maximum. Ranked by MHQ improvement per unit increase from current position.</p>
    ${sensFeatures.map((f, i) => {
        const label = featureLabel(f.name);
        const cfg = state.featuresConfig.find(c => c.name === f.name);
        const cat = cfg ? categoryLabel(cfg.category) : '';
        const svg = buildReportSparkline(f.curve, f.current, cfg);
        const slopeCls = f.slope_mhq > 0 ? 'pos' : f.slope_mhq < 0 ? 'neg' : '';
        const totalMhqCls = f.total_delta_mhq > 0 ? 'pos' : f.total_delta_mhq < 0 ? 'neg' : '';
        const prodDelta = -f.total_delta_unprod;
        const prodCls = prodDelta > 0 ? 'pos' : prodDelta < 0 ? 'neg' : '';
        return `<div class="sens-row">
            <div class="sens-rank">${i + 1}</div>
            <div class="sens-info">
                <strong>${esc(label)}</strong><br>
                <span class="cat">${esc(cat)} &middot; Current: ${fmt1(f.current)}</span>
            </div>
            <div class="sens-chart">${svg}</div>
            <div class="sens-vals">
                <div class="slope ${slopeCls}">${sign(f.slope_mhq)}${fmt1(f.slope_mhq)} MHQ/unit</div>
                <div class="totals">Total: <span class="${totalMhqCls}">${sign(f.total_delta_mhq)}${fmt1(f.total_delta_mhq)} MHQ</span> &middot; <span class="${prodCls}">${sign(prodDelta)}${fmt1(prodDelta)} days</span></div>
            </div>
        </div>`;
    }).join('')}
</div>`;
    }

    html += `
<!-- Footer -->
<div class="footer">
    Generated by Workforce Wellbeing &amp; Productivity Simulator &middot; ${dateStr}
</div>

</div>
</body>
</html>`;

    // Trigger download
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `team-report-${fileDate}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function buildReportSparkline(curve, current, cfg) {
    const W = 200, H = 50;
    const pad = 8;
    const mhqVals = curve.map(p => p.mhq);
    const minMhq = Math.min(...mhqVals);
    const maxMhq = Math.max(...mhqVals);
    const range = maxMhq - minMhq || 1;

    const xVals = curve.map(p => p.value);
    const xMin = Math.min(...xVals);
    const xMax = Math.max(...xVals);
    const xRange = xMax - xMin || 1;

    const points = curve.map(p => {
        const x = pad + ((p.value - xMin) / xRange) * (W - 2 * pad);
        const y = (H - pad) - ((p.mhq - minMhq) / range) * (H - 2 * pad);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');

    const cx = pad + ((current - xMin) / xRange) * (W - 2 * pad);
    const curPt = curve.reduce((best, p) => Math.abs(p.value - current) < Math.abs(best.value - current) ? p : best, curve[0]);
    const cy = (H - pad) - ((curPt.mhq - minMhq) / range) * (H - 2 * pad);

    return `<svg width="${W}" height="${H}" style="display:block;background:#fafafa;border-radius:4px;overflow:visible">
        <line x1="${cx.toFixed(1)}" y1="${pad}" x2="${cx.toFixed(1)}" y2="${H - pad}" stroke="#d4d4d4" stroke-width="1" stroke-dasharray="2,2"/>
        <polyline points="${points}" fill="none" stroke="#555" stroke-width="1.5" stroke-linejoin="round"/>
        <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="5" fill="#f87171" stroke="#fff" stroke-width="1.5"/>
    </svg>`;
}

function esc(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
