"""Build a self-contained interactive dashboard (single .html, no deps) from the ablation logs.

    python -m ablation.build_dashboard --log_dir ablation/logs --out ablation/dashboard.html
    # or from an existing results.json:
    python -m ablation.build_dashboard --results ablation/results.json --out ablation/dashboard.html

The HTML embeds the parsed data + a vanilla-JS UI: per-axis filters, a per-metric leaderboard
that drills down to the full config, and a val-loss-vs-wall-clock canvas plot for selected runs.
Opens offline on any machine.
"""
import argparse
import json
import os

from ablation.config_schema import EVAL_LENGTHS, expand_grid, ATTN_TYPES, REG_MODES
from ablation.parse_logs import parse_log
import glob


def metric_catalog():
    cat = [{"name": "final_val_loss", "label": "final val loss", "dir": "lower"}]
    for L in EVAL_LENGTHS:
        cat.append({"name": f"clean_ppl_{L}", "label": f"clean ppl @{L} (nats)", "dir": "lower"})
    for L in EVAL_LENGTHS:
        cat.append({"name": f"junk_ppl_{L}", "label": f"junk ppl @{L} (nats)", "dir": "lower"})
    for d in EVAL_LENGTHS:
        cat.append({"name": f"needle_acc_{d}", "label": f"needle acc @{d} (%)", "dir": "higher"})
    cat.append({"name": "mfu_final", "label": "MFU (%)", "dir": "higher"})
    cat.append({"name": "train_elapsed_s", "label": "train wall-clock (s)", "dir": "lower"})
    return cat


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Atma ablation dashboard</title>
<style>
 body{font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#d7dae0}
 h1{font-size:16px;margin:0} header{padding:10px 16px;background:#161922;border-bottom:1px solid #262b36;
   display:flex;gap:18px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:5}
 .pill{background:#222836;border-radius:10px;padding:2px 8px;font-size:11px;color:#9aa3b2}
 main{display:grid;grid-template-columns:300px 1fr;gap:0}
 aside{padding:12px 14px;border-right:1px solid #262b36;background:#12151c;min-height:90vh}
 .axis{margin-bottom:14px} .axis b{display:block;margin-bottom:4px;color:#aeb6c4;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
 label.chk{display:inline-flex;gap:4px;align-items:center;margin:2px 8px 2px 0;cursor:pointer}
 section{padding:12px 16px}
 select{background:#1b212c;color:#d7dae0;border:1px solid #2c3340;border-radius:6px;padding:4px 8px}
 table{border-collapse:collapse;width:100%;margin-top:8px;font-size:12px}
 th,td{text-align:left;padding:4px 8px;border-bottom:1px solid #20242e;white-space:nowrap}
 th{color:#8b93a3;cursor:default;position:sticky;top:52px;background:#0f1115}
 tr.run:hover{background:#171b24} td.val{font-variant-numeric:tabular-nums;font-weight:600}
 .chip{display:inline-block;border-radius:4px;padding:0 5px;margin-right:3px;font-size:10px}
 .rope{background:#3a2d52;color:#cdb6ff}.nope{background:#1f3a4d;color:#9fd8ff}.polar{background:#173d2f;color:#9af0c6}
 .on{background:#243; color:#9af0c6}.off{background:#332; color:#e0c08a}
 .reg{background:#2a2f3a;color:#aeb6c4}
 .muted{color:#6b7384}.bad{color:#e0736b}.good{color:#7fe0a0}
 #detail{margin-top:10px;background:#12151c;border:1px solid #262b36;border-radius:8px;padding:10px;display:none}
 #detail pre{white-space:pre-wrap;color:#9aa3b2;font-size:11px;max-height:260px;overflow:auto}
 canvas{background:#0b0d12;border:1px solid #20242e;border-radius:8px;margin-top:6px}
 .legend span{display:inline-flex;gap:4px;align-items:center;margin-right:12px;font-size:11px}
 .sw{width:10px;height:10px;border-radius:2px;display:inline-block}
 button{background:#222836;color:#d7dae0;border:1px solid #2c3340;border-radius:6px;padding:3px 8px;cursor:pointer}
</style></head><body>
<header>
 <h1>Atma ablation</h1>
 <span class="pill" id="status"></span>
 <span style="margin-left:auto">metric&nbsp;<select id="metric"></select></span>
 <span>top&nbsp;<select id="topn"><option>10</option><option>20</option><option selected>all</option></select></span>
 <button id="clearcmp">clear plot</button>
</header>
<main>
 <aside id="filters"></aside>
 <section>
   <div class="legend" id="legend"></div>
   <canvas id="plot" width="900" height="280"></canvas>
   <table id="board"><thead></thead><tbody></tbody></table>
   <div id="detail"></div>
 </section>
</main>
<script id="data" type="application/json">/*DATA*/</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const RECORDS = D.records, CATALOG = D.catalog, AXES = D.axes, EXPECTED = D.expected;
const byId = Object.fromEntries(RECORDS.map(r=>[r.run_id,r]));
const state = {filters:{}, metric:CATALOG[0].name, topn:'all', cmp:new Set()};
AXES.forEach(a=> state.filters[a]=new Set(D.axis_values[a]));
const PAL = ['#7fb3ff','#9af0c6','#cdb6ff','#e0c08a','#e0736b','#6fd6d0','#f0a0d0','#b6e07f'];

function fmt(v){ if(v==null||isNaN(v)) return '—'; return (Math.abs(v)>=100?v.toFixed(1):v.toFixed(3)); }
function chip(ax,val){ const m={attn_type:val,reg_mode:'reg',distractor:val?'on':'off',memory:val?'on':'off',window:val?'on':'off'};
  const cls=ax=='attn_type'?val:(ax=='reg_mode'?'reg':(val?'on':'off'));
  const txt=ax=='attn_type'?val:(ax=='reg_mode'?val:(ax[0]+(val?'+':'-')));
  return `<span class="chip ${cls}">${txt}</span>`; }

function buildFilters(){
  const el=document.getElementById('filters'); el.innerHTML='';
  AXES.forEach(ax=>{
    const d=document.createElement('div'); d.className='axis';
    d.innerHTML=`<b>${ax}</b>`;
    D.axis_values[ax].forEach(v=>{
      const id=`f_${ax}_${v}`;
      const lab=document.createElement('label'); lab.className='chk';
      lab.innerHTML=`<input type=checkbox id="${id}" checked> ${chip(ax,v)}`;
      lab.querySelector('input').onchange=e=>{ e.target.checked?state.filters[ax].add(v):state.filters[ax].delete(v); render(); };
      d.appendChild(lab);
    });
    el.appendChild(d);
  });
}
function buildMetric(){
  const s=document.getElementById('metric');
  CATALOG.forEach(m=>{const o=document.createElement('option');o.value=m.name;o.textContent=m.label+' ('+m.dir+')';s.appendChild(o);});
  s.onchange=e=>{state.metric=e.target.value;render();};
  document.getElementById('topn').onchange=e=>{state.topn=e.target.value;render();};
  document.getElementById('clearcmp').onclick=()=>{state.cmp.clear();render();};
}
function passes(r){ return AXES.every(ax=> r[ax]==null || state.filters[ax].has(typeof r[ax]=='boolean'?(r[ax]?1:0):r[ax])); }

function render(){
  const cat=CATALOG.find(m=>m.name==state.metric); const dir=cat.dir;
  let rows=RECORDS.filter(passes);
  const val=r=> r.metrics? r.metrics[state.metric] : undefined;
  rows.sort((a,b)=>{ const x=val(a),y=val(b);
    if(x==null&&y==null)return 0; if(x==null)return 1; if(y==null)return -1;
    return dir=='lower'? x-y : y-x; });
  if(state.topn!='all') rows=rows.slice(0, +state.topn);

  // status
  const done=RECORDS.filter(r=>r.status=='done').length, run=RECORDS.filter(r=>r.status=='running').length,
        err=RECORDS.filter(r=>r.status=='error').length, miss=EXPECTED.length-RECORDS.length;
  document.getElementById('status').textContent=`${EXPECTED.length} cells · ${done} done · ${run} running · ${err} error · ${miss} missing`;

  // headline columns: the metric + a few standard ones
  const cols=['needle_acc_'+D.maxlen,'clean_ppl_'+D.maxlen,'junk_ppl_'+D.maxlen,'mfu_final','train_elapsed_s'];
  const thead=document.querySelector('#board thead'), tbody=document.querySelector('#board tbody');
  thead.innerHTML='<tr><th>plot</th><th>#</th><th>run</th><th>'+cat.label+'</th>'+cols.map(c=>'<th>'+c.replace(/_/g,' ')+'</th>').join('')+'</tr>';
  tbody.innerHTML='';
  rows.forEach((r,i)=>{
    const tr=document.createElement('tr'); tr.className='run';
    const chips=AXES.map(ax=>chip(ax, ax=='attn_type'||ax=='reg_mode'? r[ax] : r[ax])).join('');
    const checked=state.cmp.has(r.run_id)?'checked':'';
    tr.innerHTML=`<td><input type=checkbox ${checked}></td><td class=muted>${i+1}</td>`+
      `<td>${chips}<span class=muted>${r.run_id}</span></td>`+
      `<td class=val>${fmt(val(r))}</td>`+
      cols.map(c=>`<td>${fmt(r.metrics&&r.metrics[c])}</td>`).join('');
    tr.querySelector('input').onclick=e=>{e.stopPropagation(); e.target.checked?state.cmp.add(r.run_id):state.cmp.delete(r.run_id); drawPlot();};
    tr.onclick=()=>showDetail(r);
    tbody.appendChild(tr);
  });
  drawPlot();
}

function showDetail(r){
  const el=document.getElementById('detail'); el.style.display='block';
  const mlines=Object.entries(r.metrics||{}).map(([k,v])=>`${k.padEnd(20)} ${fmt(v)}`).join('\n');
  el.innerHTML=`<b>${r.run_id}</b> <span class=muted>[${r.status}]</span>`+
    `<pre>METRICS\n${mlines}\n\nCONFIG\n${JSON.stringify(r.config,null,1)}`+
    (r.error?`\n\nERROR\n${(r.error.error||'').slice(0,1500)}`:'')+`</pre>`;
}

function drawPlot(){
  const cv=document.getElementById('plot'), g=cv.getContext('2d'); g.clearRect(0,0,cv.width,cv.height);
  const ids=[...state.cmp]; const leg=document.getElementById('legend'); leg.innerHTML='';
  const series=ids.map(id=>byId[id]).filter(r=>r&&r.curve&&r.curve.length);
  if(!series.length){ g.fillStyle='#6b7384'; g.fillText('select runs (plot checkbox) to compare val-loss vs wall-clock',20,24); return; }
  let xmax=0,ymin=1e9,ymax=-1e9;
  series.forEach(r=>r.curve.forEach(p=>{xmax=Math.max(xmax,p.wall_s);ymin=Math.min(ymin,p.val_loss);ymax=Math.max(ymax,p.val_loss);}));
  const pad=40,W=cv.width-pad-10,H=cv.height-pad-20;
  const X=x=>pad+ (xmax? x/xmax:0)*W, Y=y=>10+ (1-(y-ymin)/((ymax-ymin)||1))*H;
  g.strokeStyle='#20242e'; g.fillStyle='#6b7384'; g.beginPath(); g.moveTo(pad,10);g.lineTo(pad,10+H);g.lineTo(pad+W,10+H); g.stroke();
  g.fillText(ymax.toFixed(2),4,16); g.fillText(ymin.toFixed(2),4,10+H); g.fillText((xmax).toFixed(0)+'s',pad+W-20,10+H+14); g.fillText('wall',pad,10+H+14);
  series.forEach((r,i)=>{ const c=PAL[i%PAL.length]; g.strokeStyle=c; g.beginPath();
    r.curve.forEach((p,j)=>{const xx=X(p.wall_s),yy=Y(p.val_loss); j?g.lineTo(xx,yy):g.moveTo(xx,yy);}); g.stroke();
    const s=document.createElement('span'); s.innerHTML=`<span class="sw" style="background:${c}"></span>${r.run_id}`; leg.appendChild(s); });
}

buildFilters(); buildMetric(); render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description="Build the static ablation dashboard.")
    ap.add_argument("--log_dir", default="ablation/logs")
    ap.add_argument("--results", default=None, help="use an existing results.json instead of parsing logs")
    ap.add_argument("--out", default="ablation/dashboard.html")
    args = ap.parse_args()

    if args.results:
        records = json.load(open(args.results))
    else:
        records = [parse_log(p) for p in sorted(glob.glob(os.path.join(args.log_dir, "*.log")))]

    expected = [c.run_id for c in expand_grid()]
    data = {
        "records": records,
        "catalog": metric_catalog(),
        "axes": ["attn_type", "reg_mode", "distractor", "memory", "window"],
        "axis_values": {"attn_type": ATTN_TYPES, "reg_mode": REG_MODES,
                        "distractor": [0, 1], "memory": [0, 1], "window": [0, 1]},
        "expected": expected,
        "maxlen": max(EVAL_LENGTHS),
    }
    html = HTML.replace("/*DATA*/", json.dumps(data))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    done = sum(1 for r in records if r["status"] == "done")
    print(f"wrote {args.out}  ({len(records)} runs parsed, {done} done, grid={len(expected)})")


if __name__ == "__main__":
    main()
