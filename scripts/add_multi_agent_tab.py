"""
Add (or refresh) the Multi-agent tab in dashboard/dist/index.html.

dist/ is gitignored build output, so this script is how the tab is reproduced:
it reads the result files in results/, embeds them as one JSON constant, and
inserts a nav button, a page, and a render function between marker comments.
Running it again replaces the previous insert.

    python scripts/add_multi_agent_tab.py
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "dashboard" / "dist" / "index.html"
R = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


def build_data() -> dict:
    three = load("mode_comparison_fixed_tools.json")
    v3 = load("mode_comparison_v3.json")
    planted = load("planted_errors.json")
    hard = load("planted_errors_hard.json")

    def by_cat(d: dict, mode: str) -> dict:
        t = defaultdict(list)
        for x in d["runs"][mode]["results"]:
            s = x["scores"].get("task_success_score")
            if s is not None:
                t[x["category"]].append(s)
        return {c: round(sum(v) / len(v), 3) for c, v in t.items()}

    rows = []
    for x in v3["runs"]["multi_agent"]["results"]:
        s, u = x["scores"], x["token_usage"]
        rows.append({
            "id": x["scenario_id"], "category": x["category"], "q": x["question"],
            "task": s.get("task_success_score"), "tokens": u["input_tokens"] + u["output_tokens"],
            "cost": s.get("cost_usd"), "latency_s": round(x["latency_ms"] / 1000, 1),
            "route": s.get("routing_used"), "expected": s.get("routing_expected"),
            "verifier": s.get("verifier_action"), "culprit": s.get("attribution_culprit_agent"),
            "failure": s.get("attribution_failure_type"),
        })
    culprits = defaultdict(int)
    for r in rows:
        if r["culprit"]:
            culprits[f"{r['culprit']} / {r['failure']}"] += 1

    planted_sum = {k: v for k, v in planted["summary"].items() if isinstance(v, dict)}
    hard_sum = {k: v for k, v in hard["summary"].items() if isinstance(v, dict)}
    return {
        "modes": {m: three["summary"][m] for m in ("rag", "agent", "multi_agent")},
        "multi_v3": v3["summary"]["multi_agent"],
        "categories": {m: by_cat(three, m) for m in ("rag", "agent", "multi_agent")},
        "categories_v3": by_cat(v3, "multi_agent"),
        "rows": rows,
        "culprits_v3": dict(culprits),
        "culprits_first": {"structured_agent / bad_tool_args": 3, "data / other": 1,
                           "analyst_agent / reasoning_error": 1, "supervisor / other": 1},
        "planted": {"easy": planted_sum, "hard": hard_sum,
                    "false_alarm": planted["summary"].get("false_alarm_rate")},
    }


CSS = """
.ma-sec{margin:28px 0 12px}
.ma-sec h2{font-size:16px;font-weight:700;letter-spacing:-.2px;margin:0 0 4px}
.ma-sec p{font-size:13px;color:var(--text2);margin:0}
.ma-table{width:100%;border-collapse:collapse;min-width:0}
.ma-table th,.ma-table td{padding:10px 14px;font-size:13px;text-align:left;border-bottom:1px solid var(--border)}
.ma-table thead th{cursor:default}
.ma-table td.n,.ma-table th.n{text-align:right;font-variant-numeric:tabular-nums}
.ma-table tbody tr{cursor:default}
.ma-best{color:var(--green);font-weight:700}
.ma-bars{display:grid;gap:10px;padding:16px 18px}
.ma-bar-row{display:grid;grid-template-columns:150px 1fr 56px;gap:12px;align-items:center;font-size:13px}
.ma-bar{height:10px;border-radius:5px;background:var(--surface2);overflow:hidden}
.ma-bar>i{display:block;height:100%;border-radius:5px}
.ma-grp{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}
.ma-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:16px 18px}
.ma-card h3{font-size:12px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;margin:0 0 12px}
.ma-note{background:var(--yellow-bg);border:1px solid rgba(234,179,8,.25);color:var(--text);
  border-radius:var(--radius);padding:12px 16px;font-size:13px;line-height:1.5;margin-top:24px}
.ma-tag{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600}
.ma-ok{background:var(--green-bg);color:var(--green)}.ma-mid{background:var(--yellow-bg);color:var(--yellow)}
.ma-bad{background:var(--red-bg);color:var(--red)}
@media(max-width:700px){.ma-bar-row{grid-template-columns:110px 1fr 48px}}
"""

PAGE = """
<div id="page-multi" class="page">
<div class="container">
  <div class="hero">
    <div class="hero-info">
      <div class="hero-title">Multi-agent <span>vs single agent</span></div>
      <div class="hero-sub">18 hard scenarios · RAG vs single agent vs supervisor + specialists + verifier</div>
    </div>
    <div class="hero-kpis" id="ma-kpis"></div>
  </div>

  <div class="ma-sec"><h2>Three-way comparison</h2>
    <p>Task success from the tool-fixed run. Green marks the best value in each row. Multi-agent costs about 3x the single agent for the same accuracy.</p></div>
  <div class="table-wrap"><div class="table-scroll" id="ma-compare"></div></div>

  <div class="ma-sec"><h2>Accuracy by category</h2><p>Mean task success. 4 to 5 scenarios per category, so one scenario moves a bar by 20 to 25 points.</p></div>
  <div class="ma-grp" id="ma-cats"></div>

  <div class="ma-sec"><h2>Who caused the failures?</h2>
    <p>LLM-as-judge attribution, wrong answers only. "data" means the test data or expected answer was at fault.</p></div>
  <div class="ma-grp" id="ma-culprits"></div>

  <div class="ma-sec"><h2>Does the verifier catch real errors?</h2>
    <p>Deterministic corruptions planted into correct drafts, then shown to the verifier with the original evidence.</p></div>
  <div class="table-wrap"><div class="ma-bars" id="ma-planted"></div></div>

  <div class="ma-sec"><h2>Per scenario (latest multi-agent run)</h2><p>Routes show the specialists used. S structured, R retrieval, A analyst.</p></div>
  <div class="table-wrap"><div class="table-scroll" id="ma-rows"></div></div>

  <div class="ma-note"><strong>Read with care.</strong> 18 scenarios, one run each. The same answer was scored 1.0 and 0.5 by the judge on different runs, and the token budget is only tested with mocks. See docs/multi-agent-findings.md.</div>
</div>
</div>
"""

JS = r"""
function renderMA() {
  const D = MA, M = D.modes, pct = v => v == null ? 'n/a' : (v * 100).toFixed(0) + '%';
  const f1 = v => v == null ? 'n/a' : v.toFixed(2);
  const money = v => '$' + v.toFixed(3);
  const el = id => document.getElementById(id);
  const hardAll = Object.values(D.planted.hard).reduce((a, b) => a + b.flagged, 0);
  const hardN = Object.values(D.planted.hard).reduce((a, b) => a + b.n, 0);
  const easyAll = Object.entries(D.planted.easy).filter(([k]) => k !== 'clean').reduce((a, [, b]) => a + b.flagged, 0);
  const easyN = Object.entries(D.planted.easy).filter(([k]) => k !== 'clean').reduce((a, [, b]) => a + b.n, 0);

  el('ma-kpis').innerHTML = [
    ['kpi-blue', f1(M.agent.accuracy), 'Single agent'],
    ['kpi-purple', f1(M.multi_agent.accuracy), 'Multi-agent'],
    ['kpi-yellow', (M.multi_agent.avg_cost_usd / M.agent.avg_cost_usd).toFixed(1) + 'x', 'Cost vs single'],
    ['kpi-green', easyAll + '/' + easyN, 'Obvious errors caught'],
    ['kpi-green', hardAll + '/' + hardN, 'Subtle errors caught'],
  ].map(k => `<div class="kpi ${k[0]}"><div class="kpi-val">${k[1]}</div><div class="kpi-lbl">${k[2]}</div></div>`).join('');

  const defs = [
    ['Accuracy (task success)', 'accuracy', f1, 1], ['Fully correct (of 18)', 'fully_correct', v => v, 1],
    ['Faithfulness', 'faithfulness', f1, 1], ['Hallucination (lower is better)', 'hallucination', f1, -1],
    ['LLM calls per query', 'avg_llm_calls', v => v.toFixed(1), -1],
    ['Tokens per query', 'avg_tokens', v => Math.round(v).toLocaleString(), -1],
    ['Latency (s)', 'avg_latency_ms', v => (v / 1000).toFixed(1), -1],
    ['Cost per query', 'avg_cost_usd', money, -1],
  ];
  const modes = ['rag', 'agent', 'multi_agent'];
  el('ma-compare').innerHTML = `<table class="ma-table"><thead><tr><th>Metric</th><th class="n">RAG</th>
    <th class="n">Single agent</th><th class="n">Multi-agent</th></tr></thead><tbody>` +
    defs.map(([label, key, fmt, dir]) => {
      const vals = modes.map(m => M[m][key]);
      const best = dir > 0 ? Math.max(...vals) : Math.min(...vals);
      return `<tr><td>${label}</td>` + vals.map(v =>
        `<td class="n ${v === best ? 'ma-best' : ''}">${fmt(v)}</td>`).join('') + '</tr>';
    }).join('') + '</tbody></table>';

  const cats = ['multi_field', 'aggregation', 'multi_hop', 'adversarial'];
  const colors = {rag: 'var(--orange)', agent: 'var(--blue)', multi_agent: 'var(--purple)'};
  const names = {rag: 'RAG', agent: 'Single agent', multi_agent: 'Multi-agent'};
  el('ma-cats').innerHTML = cats.map(c => `<div class="ma-card"><h3>${c.replace('_', ' ')}</h3>` +
    modes.map(m => { const v = D.categories[m][c]; return `<div class="ma-bar-row" style="grid-template-columns:96px 1fr 40px;margin-bottom:8px">
      <span>${names[m]}</span><div class="ma-bar"><i style="width:${v * 100}%;background:${colors[m]}"></i></div><b>${v.toFixed(2)}</b></div>`; }).join('') +
    '</div>').join('');

  const bars = (obj, color) => { const mx = Math.max(...Object.values(obj), 1);
    return Object.entries(obj).sort((a, b) => b[1] - a[1]).map(([k, v]) => `<div class="ma-bar-row" style="grid-template-columns:1fr 1fr 24px">
      <span>${k}</span><div class="ma-bar"><i style="width:${v / mx * 100}%;background:${color}"></i></div><b>${v}</b></div>`).join(''); };
  el('ma-culprits').innerHTML =
    `<div class="ma-card"><h3>First run (before the lookup fix)</h3>${bars(D.culprits_first, 'var(--orange)')}</div>` +
    `<div class="ma-card"><h3>Latest full run</h3>${bars(D.culprits_v3, 'var(--purple)')}</div>`;

  const lab = {swap_entity: 'Swapped director', change_number: 'Changed number', add_unsupported_claim: 'Unsupported claim',
    swap_years: 'Swapped years', near_miss_number: 'Near-miss number', drop_list_item: 'Dropped list item', clean: 'Clean drafts flagged (false alarms)'};
  const bar = (k, s) => { const r = s.flagged / s.n, bad = k === 'clean';
    const col = bad ? (r > 0.2 ? 'var(--red)' : 'var(--yellow)') : (r >= 0.9 ? 'var(--green)' : r >= 0.7 ? 'var(--yellow)' : 'var(--red)');
    return `<div class="ma-bar-row"><span>${lab[k] || k}</span><div class="ma-bar"><i style="width:${r * 100}%;background:${col}"></i></div><b>${s.flagged}/${s.n}</b></div>`; };
  el('ma-planted').innerHTML =
    Object.entries(D.planted.easy).filter(([k]) => k !== 'clean').map(([k, s]) => bar(k, s)).join('') +
    Object.entries(D.planted.hard).map(([k, s]) => bar(k, s)).join('') + bar('clean', D.planted.easy.clean);

  const short = r => (r || []).map(a => a[0].toUpperCase()).join('→') || '–';
  const tag = (v) => v == null ? '–' : `<span class="ma-tag ${v >= 1 ? 'ma-ok' : v >= 0.5 ? 'ma-mid' : 'ma-bad'}">${v.toFixed(1)}</span>`;
  el('ma-rows').innerHTML = `<table class="ma-table"><thead><tr><th>ID</th><th>Category</th><th>Task</th>
    <th class="n">Tokens</th><th class="n">Latency</th><th>Route (used / expected)</th><th>Verifier</th><th>Culprit</th></tr></thead><tbody>` +
    D.rows.map(r => `<tr><td>${r.id}</td><td>${r.category}</td><td>${tag(r.task)}</td>
      <td class="n">${r.tokens.toLocaleString()}</td><td class="n">${r.latency_s}s</td>
      <td>${short(r.route)} / ${short(r.expected)}</td><td>${r.verifier || '–'}</td>
      <td>${r.culprit ? r.culprit + ' / ' + r.failure : '–'}</td></tr>`).join('') + '</tbody></table>';
}
renderMA();
"""


def inject(html: str, data: dict) -> str:
    """Insert the CSS, nav button, page and script; any earlier insert is removed first."""
    html = re.sub(r"<!--MA-([A-Z]+)-->.*?<!--/MA-\1-->", "", html, flags=re.S)
    html = re.sub(r"/\*MA-CSS\*/.*?/\*/MA-CSS\*/", "", html, flags=re.S)

    def block(tag: str, body: str) -> str:
        return f"<!--MA-{tag}-->{body}<!--/MA-{tag}-->"

    html = html.replace("</style>", "/*MA-CSS*/" + CSS + "/*/MA-CSS*/</style>", 1)
    reliability = "<button class=\"nav-tab\" onclick=\"showPage('reliability')\">Reliability</button>"
    tab = "<button class=\"nav-tab\" onclick=\"showPage('multi')\">Multi-agent</button>"
    html = html.replace(reliability, reliability + block("NAV", tab), 1)
    html = html.replace("<script>", block("PAGE", PAGE) + "<script>", 1)
    script = f"<script>const MA = {json.dumps(data, separators=(',', ':'))};{JS}</script>"
    return html.replace("</body>", block("JS", script) + "</body>", 1)


def main() -> int:
    if not HTML.exists():
        print(f"{HTML} not found", file=sys.stderr)
        return 1
    HTML.write_text(inject(HTML.read_text(), build_data()))
    print(f"updated {HTML}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
