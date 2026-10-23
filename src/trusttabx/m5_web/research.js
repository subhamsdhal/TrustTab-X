/* Read-only charts of packaged, verified aggregate research results. */
const TrustTabXResearch = (() => {
  const metricNames = {brier:'Brier score', log_loss:'Log loss', ece_10_bins:'10-bin expected calibration error'};
  const modelNames = {logistic_regression:'Logistic regression',hist_gradient_boosting:'Histogram gradient boosting',random_forest:'Random forest',xgboost:'XGBoost',catboost_ohe:'CatBoost + one-hot'};
  const policyNames = {static:'Fixed expert',uniform:'Uniform ensemble',weighted:'Weighted ensemble',uniform_ensemble:'Uniform ensemble',weighted_ensemble:'Weighted ensemble',metadata_router:'Metadata router',simple_router:'Simplified router',confidence_expert:'Confidence expert',reliability_router:'Reliability router',calibrated_static:'Calibrated fixed expert'};
  const svgNS = 'http://www.w3.org/2000/svg';
  let data, datasetIndex = 0, metric = 'brier', comparisonIndex = 0;

  function chartGeometry(rows, key) {
    const bars = rows.filter(row => typeof row[key] === 'number' && Number.isFinite(row[key]) && row[key] >= 0)
      .map(row => ({label:row.label, value:row[key], kind:row.kind || 'default'}));
    const highest = Math.max(0, ...bars.map(bar => bar.value));
    const scale = highest ? 10 ** Math.floor(Math.log10(highest)) : 1;
    const axisMax = highest ? Math.ceil(highest / scale * 1.05 * 10) / 10 * scale : 1;
    return {bars:bars.map(bar => ({...bar, fraction:bar.value / axisMax})), axisMax};
  }

  function node(name, attrs = {}, value = '') {
    const element = document.createElementNS(svgNS, name);
    Object.entries(attrs).forEach(([key, content]) => element.setAttribute(key, String(content)));
    if (value !== '') element.textContent = value;
    return element;
  }

  function color(kind) {
    if (kind === 'router') return '#e6bd84';
    if (kind === 'ensemble') return '#8acbd0';
    if (kind === 'selected') return '#a7eed1';
    return '#79b89d';
  }

  function drawBars(target, rows, key, heading) {
    target.replaceChildren();
    const geometry = chartGeometry(rows, key);
    if (!geometry.bars.length) {
      const empty = document.createElement('div'); empty.className = 'chart-empty';
      empty.textContent = `${metricNames[key] || key} was not reported for this comparison.`;
      target.append(empty); target.setAttribute('aria-label', empty.textContent); return;
    }
    const width = 560, left = 183, plotWidth = 285, top = 28, rowHeight = 37;
    const height = top + geometry.bars.length * rowHeight + 40;
    const svg = node('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':heading});
    const axisX = left + plotWidth;
    for (let tick = 0; tick <= 2; tick++) {
      const x = left + plotWidth * tick / 2;
      svg.append(node('line',{x1:x,y1:top-7,x2:x,y2:height-34,stroke:'#35524c','stroke-width':1}));
      svg.append(node('text',{x,y:height-12,fill:'#8fb5a7','font-size':11,'text-anchor':tick===0?'start':tick===2?'end':'middle'},(geometry.axisMax*tick/2).toFixed(3)));
    }
    geometry.bars.forEach((bar,index) => {
      const y = top + index*rowHeight;
      svg.append(node('text',{x:left-9,y:y+15,fill:'#d4e5dc','font-size':12,'text-anchor':'end'},bar.label));
      const rect = node('rect',{x:left,y,width:Math.max(1,bar.fraction*plotWidth),height:20,rx:4,fill:color(bar.kind)});
      rect.append(node('title',{},`${bar.label}: ${metricNames[key] || key} ${bar.value.toFixed(6)}. Lower is better.`));
      svg.append(rect);
      svg.append(node('text',{x:axisX+12,y:y+15,fill:'#e7f4ed','font-size':12},bar.value.toFixed(4)));
    });
    target.append(svg);
    target.setAttribute('aria-label',`${heading}. ${geometry.bars.map(bar=>`${bar.label} ${bar.value.toFixed(4)}`).join('; ')}. Lower is better.`);
  }

  function policyKind(policy) {
    if (policy.includes('router')) return 'router';
    if (policy === 'uniform' || policy === 'weighted' || policy.endsWith('_ensemble')) return 'ensemble';
    return 'selected';
  }

  function text(id, value) { document.getElementById(id).textContent = value || ''; }

  function setView(name) {
    document.getElementById('research-view').hidden = name !== 'research';
    document.getElementById('demo-view').hidden = name !== 'demo';
    document.querySelectorAll('[data-view]').forEach(button => {
      const active = button.dataset.view === name;
      button.classList.toggle('active',active);
      if (active) button.setAttribute('aria-current','page'); else button.removeAttribute('aria-current');
    });
  }

  function render() {
    const dataset = data.datasets[datasetIndex];
    document.querySelectorAll('#dataset-tabs button').forEach((button,index) => {
      button.classList.toggle('active', index === datasetIndex);
      button.setAttribute('aria-selected', String(index === datasetIndex));
    });
    text('dataset-type',dataset.shift_kind); text('dataset-title',dataset.title);
    text('dataset-provenance',dataset.provenance); text('selected-model',modelNames[dataset.selected_model] || dataset.selected_model);
    const sourceLink = document.getElementById('dataset-source'); sourceLink.hidden = !dataset.source_url;
    if (dataset.source_url) sourceLink.href = dataset.source_url;
    text('dataset-protocol',dataset.protocol); text('shift-scope',dataset.shift_kind);
    text('shift-note',dataset.shift_kind === 'ID / OOD' ? 'The same source-selected expert is evaluated on ID and held-out OOD test rows.' :
      dataset.shift_kind === 'Forward time' ? 'Both values are later chronological windows; neither is an IID test.' :
      'Held-out results are from a separate cohort or shifted month; no IID comparison is available.');
    text('comparison-note',dataset.comparison_note); text('dataset-limits',dataset.limitations.join(' '));
    text('source-hash-note',dataset.result_files.map(path=>`${path} · SHA-256 ${data.source_files_sha256[path].slice(0,12)}…`).join(' | '));
    const options = document.getElementById('comparison-window'); options.replaceChildren();
    dataset.comparisons.forEach((group,index) => { const option = document.createElement('option'); option.value = String(index); option.textContent = group.label; options.append(option); });
    comparisonIndex = Math.min(comparisonIndex,dataset.comparisons.length-1); options.value = String(comparisonIndex);
    updateCharts();
  }

  function updateCharts() {
    const dataset = data.datasets[datasetIndex]; const group = dataset.comparisons[comparisonIndex];
    document.querySelectorAll('[data-metric]').forEach(button => button.classList.toggle('active',button.dataset.metric === metric));
    text('metric-help',metric === 'ece_10_bins' ? 'ECE uses 10 bins where reported. Missing calibration estimates remain unavailable.' :
      `${metricNames[metric]} is lower-is-better; charts start at zero and show exact values on hover.`);
    drawBars(document.getElementById('shift-chart'),dataset.shift.map(row=>({label:row.label,kind:'selected',...row})),metric,
      `${dataset.title} ${dataset.shift_kind} ${metricNames[metric]}`);
    drawBars(document.getElementById('policy-chart'),group.rows.map(row=>({label:policyNames[row.policy] || row.policy,kind:policyKind(row.policy),...row})),metric,
      `${dataset.title} ${group.label} policy ${metricNames[metric]}`);
    drawBars(document.getElementById('model-chart'),data.models.map(name=>({label:modelNames[name] || name,
      brier:dataset.selection_brier[name],kind:name===dataset.selected_model?'selected':'default'})), 'brier',
      `${dataset.title} source-selection model Brier score`);
  }

  async function boot() {
    document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>setView(button.dataset.view)));
    document.querySelectorAll('[data-metric]').forEach(button=>button.addEventListener('click',()=>{metric=button.dataset.metric;updateCharts();}));
    document.getElementById('comparison-window').addEventListener('change',event=>{comparisonIndex=Number(event.target.value);updateCharts();});
    try {
      const response = await fetch('/assets/research_results.json',{cache:'no-store'});
      if (!response.ok) throw new Error('Result bundle unavailable');
      data = await response.json();
      if (data.schema_version !== 1 || data.decision !== 'C. NO ADAPTIVE ADVANTAGE DEMONSTRATED' || data.release_classification !== 'Local Demo Only' || data.datasets.length !== 5) throw new Error('Unexpected result bundle');
      const tabs = document.getElementById('dataset-tabs');
      data.datasets.forEach((dataset,index)=>{const button=document.createElement('button');button.type='button';button.setAttribute('role','tab');button.textContent=dataset.title;button.addEventListener('click',()=>{datasetIndex=index;comparisonIndex=0;render();});tabs.append(button);});
      text('decision-explanation',data.decision_explanation); text('research-status','Verified aggregate results loaded'); render();
    } catch (error) { text('research-status','Research results unavailable; see local manuscript.'); }
  }

  return {chartGeometry,boot};
})();

if (typeof module !== 'undefined' && module.exports) module.exports = TrustTabXResearch;
if (typeof document !== 'undefined') TrustTabXResearch.boot();
