const $ = (s) => document.querySelector(s);
let data = {experiments: [], figures: [], tables: [], runs: {}};
let accuracyChart, lossChart, selectedKind = 'baseline';
const hosted = !['127.0.0.1','localhost'].includes(location.hostname);
const rawRoot = 'https://raw.githubusercontent.com/PaulEbenezer454/backdoordefence/main/';

function pct(value) { return value == null ? '—' : `${(value * 100).toFixed(2)}%`; }
function num(value, digits=2) { return value == null ? '—' : Number(value).toFixed(digits); }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function toast(message, error=false) { const el=$('#toast'); el.textContent=message; el.className=`toast show${error?' error':''}`; clearTimeout(toast.timer); toast.timer=setTimeout(()=>el.className='toast',4200); }

function renderStats() {
  const runs=data.experiments, attacks=runs.filter(r=>r.attack && r.attack!=='none'), benign=runs.filter(r=>!r.attack||r.attack==='none');
  const best=benign.filter(r=>r.accuracy!=null).sort((a,b)=>b.accuracy-a.accuracy)[0];
  const last=benign[0];
  const items=[['RECORDED RUNS',runs.length,'Saved locally','▤'],['BEST BENIGN ACCURACY',best?pct(best.accuracy):'—',best?best.id:'No benign run saved','↗'],['ATTACK RUNS',attacks.length,'LSA / conventional','⌁'],['AVAILABLE FIGURES',data.figures.length,'PNG charts from experiments','▧']];
  $('#stats').innerHTML=items.map(x=>`<article class="stat"><div class="stat-top">${x[0]}<span class="stat-icon">${x[3]}</span></div><div class="stat-value">${x[1]}</div><div class="stat-foot">${escapeHtml(x[2])}</div></article>`).join('');
}
function renderRuns() {
  const query=$('#search').value.toLowerCase(), filter=$('#filter').value;
  const rows=data.experiments.filter(r=>`${r.id} ${r.attack} ${r.partition}`.toLowerCase().includes(query)).filter(r=>filter==='all'||(filter==='attack'?(r.attack&&r.attack!=='none'):(!r.attack||r.attack==='none')));
  $('#run-rows').innerHTML=rows.slice(0,100).map(r=>{const attack=r.attack&&r.attack!=='none'; return `<tr><td class="run-name" title="${escapeHtml(r.id)}">${escapeHtml(r.id)}</td><td><span class="pill ${attack?'attack':'clean'}">${attack?escapeHtml(r.attack):'FedAvg'}</span></td><td>${r.rounds}</td><td class="metric">${pct(r.accuracy)}</td><td>${pct(r.asr)}</td><td>${r.seed??'—'}</td><td><button class="view-link" data-chart="${escapeHtml(r.id)}">Inspect →</button></td></tr>`}).join('')||'<tr><td colspan="7">No matching experiments.</td></tr>';
  $('#chart-run').innerHTML=data.experiments.map(r=>`<option value="${escapeHtml(r.id)}">${escapeHtml(r.id)}</option>`).join('');
  if(data.experiments.length) drawCharts(data.experiments[0].id);
}
function drawCharts(id) {
  const run=data.experiments.find(r=>r.id===id); if(!run)return;
  const rounds=run.history.map(p=>p.round), accuracy=run.history.map(p=>p.accuracy==null?null:p.accuracy*100), loss=run.history.map(p=>p.loss);
  const common={responsive:true,maintainAspectRatio:false,interaction:{intersect:false,mode:'index'},plugins:{legend:{display:false},tooltip:{backgroundColor:'#1f2a39',padding:10,titleFont:{family:'DM Sans'},bodyFont:{family:'DM Sans'}}},scales:{x:{grid:{display:false},ticks:{color:'#96a0ac',font:{size:9}},title:{display:true,text:'Federated round',color:'#929ca8',font:{size:9}}},y:{grid:{color:'#f0f2f5'},ticks:{color:'#96a0ac',font:{size:9}},border:{display:false}}}};
  const a=$('#accuracy-chart').getContext('2d'), l=$('#loss-chart').getContext('2d');
  if(accuracyChart)accuracyChart.destroy(); if(lossChart)lossChart.destroy();
  accuracyChart=new Chart(a,{type:'line',data:{labels:rounds,datasets:[{data:accuracy,borderColor:'#3975f6',backgroundColor:'#3975f61a',fill:true,borderWidth:2,pointRadius:0,pointHoverRadius:4,tension:.25}]},options:{...common,scales:{...common.scales,y:{...common.scales.y,min:0,max:100,ticks:{...common.scales.y.ticks,callback:v=>`${v}%`}}}}});
  lossChart=new Chart(l,{type:'line',data:{labels:rounds,datasets:[{data:loss,borderColor:'#17a17a',backgroundColor:'#17a17a18',fill:true,borderWidth:2,pointRadius:0,pointHoverRadius:4,tension:.25}]},options:common});
}
function fileUrl(file){return hosted?rawRoot+file:'/files/'+file.split('/').map(encodeURIComponent).join('/')}
function renderFigures(){ $('#figures').innerHTML=data.figures.slice(0,30).map(file=>`<article class="figure-card" data-image="${escapeHtml(file)}"><img loading="lazy" src="${fileUrl(file)}" alt="${escapeHtml(file.split('/').pop())}"><div>${escapeHtml(file.split('/').pop().replaceAll('_',' '))}</div></article>`).join('')||'<p>No saved figures found.</p>'; }
function renderTables(){ $('#tables').innerHTML=data.tables.map(file=>`<a class="table-link" href="${fileUrl(file)}" ${hosted?'target="_blank" rel="noreferrer"':'download'}><span>CSV</span>${escapeHtml(file.split('/').pop())} ↓</a>`).join('')||'<p>No generated tables found.</p>'; }
async function load(){try{const r=await fetch(hosted?'./data.json':'/api/data');data=await r.json();renderStats();renderRuns();renderFigures();renderTables();if(hosted){document.querySelectorAll('[data-run]').forEach(el=>el.hidden=true);$('#refresh').hidden=true;$('#dialog-description').textContent='GitHub Pages displays recorded experiment results. To launch training, run the local dashboard from Run-Demo.bat.';}}catch(e){toast('Could not load project results. Refresh the page or check the deployment.',true)}}

document.addEventListener('click', async e=>{
  const runBtn=e.target.closest('[data-run]'); if(runBtn){selectedKind=runBtn.dataset.run;const titles={baseline:'Run benign FedAvg',lsa:'Run LSA-inspired experiment',conventional:'Run conventional backdoor control'};$('#dialog-title').textContent=titles[selectedKind];$('#dialog-description').textContent='This runs on your computer and can take a few minutes. Results will appear here when saved.';$('#config').value=selectedKind==='baseline'?'mnist.yaml':selectedKind==='conventional'?'mnist_conventional_late.yaml':'mnist_lsa_late.yaml';$('#run-dialog').showModal();}
  if(e.target.closest('#cancel-run'))$('#run-dialog').close();
  const inspect=e.target.closest('[data-chart]'); if(inspect){$('#chart-run').value=inspect.dataset.chart;drawCharts(inspect.dataset.chart);location.hash='#charts';}
  const image=e.target.closest('[data-image]');if(image)window.open(fileUrl(image.dataset.image),'_blank','noopener');
});
$('#confirm-run').addEventListener('click',async e=>{e.preventDefault();const btn=e.currentTarget;btn.disabled=true;btn.textContent='Starting…';try{const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:selectedKind,config:$('#config').value})});const result=await response.json();if(!response.ok)throw new Error(result.error||'Could not start the run');$('#run-dialog').close();toast(`Started ${result.experiment_id}. Refresh results while it runs.`);setTimeout(load,1500);const timer=setInterval(async()=>{await load();const status=data.runs?.[result.experiment_id];if(status&&status.status!=='running'){clearInterval(timer);toast(status.status==='completed'?`Run ${result.experiment_id} finished.`:`Run ${result.experiment_id} failed. Check the terminal output.` ,status.status!=='completed');}},5000);}catch(err){toast(err.message,true)}finally{btn.disabled=false;btn.textContent='Start run';}});
$('#refresh').addEventListener('click',load);$('#search').addEventListener('input',renderRuns);$('#filter').addEventListener('change',renderRuns);$('#chart-run').addEventListener('change',e=>drawCharts(e.target.value));load();

