# app.py
from flask import Flask, render_template, jsonify
from assess import get_report, START_SLOTS

app = Flask(__name__)

def _badge(status: str) -> str:
    s = (status or "").upper()
    if s in {"OUT", "IR"}:
        return "out"
    if s in {"QUESTIONABLE", "DOUBTFUL"}:
        return "questionable"
    if s in {"ACTIVE", ""}:
        return "active"
    return "normal"

def _by_slot_order(starters):
    slot_order = {s: i for i, s in enumerate(START_SLOTS)}
    return sorted(starters, key=lambda x: (slot_order.get(x["slot"], 99), -float(x["wk_proj"])))

def _swap_map(swaps):
    # map by (slot, sit_name) -> swap dict
    m = {}
    for s in swaps:
        key = (s["slot"], s["sit"]["name"])
        m[key] = s
    return m

def _safe_num(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def build_view_models(payload, ai):
    # meta/top
    meta = {
        "team": payload["team"],
        "opponent": payload["opponent"],
        "week": payload["week"],
        "proj_me": payload["proj"]["me"],
        "proj_opp": payload["proj"]["opp"],
        "strategy": payload["strategy_bias"],
        "scoring": payload["scoring"],
    }

    starters_sorted = _by_slot_order(payload["my_starters"])
    swap_by_key = _swap_map(payload.get("local_swap_candidates", []))

    # Build Start/Sit rows deterministically (no duplicates):
    start_sit_rows = []
    for s in starters_sorted:
        key = (s["slot"], s["name"])
        swap = swap_by_key.get(key)
        if swap:
            verdict = f"Bench for {swap['start']['name']}"
            reason = f"Higher projection ({swap['start']['wk_proj']:.2f} vs {s['wk_proj']:.2f})."
        else:
            verdict = "Start"
            reason = {
                "QB": "Highest projected QB on roster.",
                "RB": "Better projection than alternatives.",
                "WR": "Top WR projection among starters.",
                "TE": "Best TE option on roster.",
                "FLEX": "Best available option for FLEX.",
                "D/ST": "Only D/ST available or best matchup.",
                "K": "Highest projected kicker."
            }.get(s["slot"], "Good projection.")
        start_sit_rows.append({
            "slot": s["slot"],
            "name": s["name"],
            "pos": s["pos"],
            "team": s["team"],
            "bye": s.get("bye") or "None",
            "status": s.get("status") or "ACTIVE",
            "status_class": _badge(s.get("status")),
            "live": f"{_safe_num(s.get('wk_actual')):.2f}",
            "proj": f"{_safe_num(s.get('wk_proj')):.2f}",
            "verdict": verdict,
            "reason": reason,
        })

    # Bench table: just projections (already sorted in payload)
    bench_rows = []
    for b in payload["my_bench"]:
        bench_rows.append({
            "name": b["name"],
            "pos": b["pos"],
            "team": b["team"],
            "status": b.get("status") or "ACTIVE",
            "status_class": _badge(b.get("status")),
            "bye": b.get("bye") or "None",
            "live": f"{_safe_num(b.get('wk_actual')):.2f}",
            "proj": f"{_safe_num(b.get('wk_proj')):.2f}",
        })

    # Executive summary
    summary = (ai or {}).get("executive_summary") or "No executive summary yet (model fallback). Use the Start/Sit table below."

    # Waivers / trades / risks
    waiv = (ai or {}).get("waivers") or []
    trades = (ai or {}).get("trades") or []
    risks = (ai or {}).get("risks") or []

    # Bench-swap ideas (local)
    swaps = payload.get("local_swap_candidates", [])
    swap_rows = [{
        "slot": s["slot"],
        "sit": f"{s['sit']['name']} ({s['sit']['pos']}) — {s['sit']['wk_proj']:.2f}",
        "start": f"{s['start']['name']} ({s['start']['pos']}) — {s['start']['wk_proj']:.2f}",
        "delta": f"{s['delta']:.2f}"
    } for s in swaps]

    return {
        "meta": meta,
        "summary": summary,
        "start_sit_rows": start_sit_rows,
        "bench_rows": bench_rows,
        "waivers": waiv,
        "trades": trades,
        "risks": risks,
        "swaps": swap_rows,
        "raw_model": ai,  # for debugging if needed
    }

@app.route("/json")
def json_dump():
    payload, ai, raw = get_report()
    return jsonify({"payload": payload, "ai": ai, "raw": raw})

@app.route("/healthz")
def health():
    return "ok", 200

@app.route("/")
def index():
    payload, ai, raw = get_report()
    vm = build_view_models(payload, ai)
    return render_template("index.html", **vm)

if __name__ == "__main__":
    app.run(debug=True)
