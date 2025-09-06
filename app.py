# app.py
import os
from flask import Flask, render_template, jsonify
from assess import get_report  # returns (payload, report_or_none, raw_text)

app = Flask(__name__, static_folder="static", template_folder="templates")

def build_meta(payload):
    proj = payload.get("proj", {}) or {}
    return {
        "league": payload.get("league", ""),
        "team": payload.get("team", ""),
        "opponent": payload.get("opponent", "TBD"),
        "week": payload.get("week", 1),
        "strategy": payload.get("strategy_bias", "floor"),
        "proj_me": float(proj.get("me") or 0.0),
        "proj_opp": float(proj.get("opp") or 0.0),
    }

@app.route("/healthz")
def health():
    return "ok", 200

@app.route("/json")
def as_json():
    payload, report, raw = get_report()
    return jsonify({
        "payload": payload,
        "report": report,
        "raw": raw
    })

@app.route("/")
def index():
    payload, report, raw = get_report()
    meta = build_meta(payload)

    # ensure report has expected keys even if OpenAI failed/rate-limited
    if report is None:
        report = {
            "summary": "AI summary unavailable (model limit or key missing). Showing raw payload-derived data.",
            "start_sit": [],
            "bench_ranked": payload.get("my_bench", [])[:10],
            "waivers": [],
            "trades": [],
            "risks": []
        }

    # provide convenience lists for the template
    starters = payload.get("my_starters", [])
    bench = payload.get("my_bench", [])
    fa_top = payload.get("free_agents_top", [])
    local_swaps = payload.get("local_swap_candidates", [])

    return render_template(
        "index.html",
        meta=meta,
        payload=payload,
        report=report,
        starters=starters,
        bench=bench,
        fa_top=fa_top,
        local_swaps=local_swaps,
        raw=raw
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
