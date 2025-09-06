# app.py
import os
from flask import Flask, render_template, jsonify
from assess import get_report  # must return (payload, report_or_none, raw_text)

app = Flask(__name__, static_folder="static", template_folder="templates")


def _safe_numbers(payload):
    """Pull projections safely from payload."""
    proj = payload.get("proj") or {}
    try:
        me = float(proj.get("me", 0) or 0)
    except Exception:
        me = 0.0
    try:
        opp = float(proj.get("opp", 0) or 0)
    except Exception:
        opp = 0.0
    return me, opp


@app.route("/")
def index():
    """
    Main dashboard.
    - Calls assess.get_report()
    - Builds `meta` (team/opponent/league/week/projections/strategy)
    - Renders templates/index.html
    """
    try:
        payload, ai_report, raw = get_report()
    except Exception as e:
        # Failsafe: keep the page up even if ESPN/OpenAI hiccup
        payload, ai_report, raw = (
            {
                "team": "—",
                "opponent": "—",
                "league": "—",
                "week": 0,
                "proj": {"me": 0, "opp": 0},
                "strategy_bias": "floor",
                "my_starters": [],
                "my_bench": [],
                "free_agents_top": [],
                "local_swap_candidates": [],
            },
            None,
            f"ERROR calling get_report(): {e}",
        )

    me_proj, opp_proj = _safe_numbers(payload)

    meta = {
        "team": payload.get("team", "—"),
        "opponent": payload.get("opponent", "—"),
        "league": payload.get("league", "—"),
        "week": payload.get("week", 0),
        "proj_me": me_proj,
        "proj_opp": opp_proj,
        "strategy": payload.get("strategy_bias", "floor"),
    }

    return render_template(
        "index.html",
        payload=payload,
        report=ai_report,  # may be None if OpenAI was skipped or rate-limited
        raw=raw,           # raw AI text (for the collapsible debug)
        meta=meta,         # <-- template expects this
    )


@app.route("/json")
def json_dump():
    """Raw JSON for debugging / API-like usage."""
    try:
        payload, ai_report, raw = get_report()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(
        {
            "meta": {
                "team": payload.get("team", "—"),
                "opponent": payload.get("opponent", "—"),
                "league": payload.get("league", "—"),
                "week": payload.get("week", 0),
                "proj": payload.get("proj", {"me": 0, "opp": 0}),
                "strategy": payload.get("strategy_bias", "floor"),
            },
            "payload": payload,
            "report": ai_report,  # can be None
            "raw": raw,
        }
    )


@app.route("/healthz")
def health():
    return "ok", 200


if __name__ == "__main__":
    # Local dev server; Render will use gunicorn per your start command
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
