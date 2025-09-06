# app.py
from flask import Flask, render_template, jsonify, redirect, url_for
from assess import get_report
import json, time, os

app = Flask(__name__)

CACHE = {"ts": 0, "payload": None, "report": None, "raw": None}
CACHE_TTL = 60  # seconds


def deep_convert(obj):
    """Recursively convert non-JSON types (e.g., set) to JSON-safe ones."""
    if isinstance(obj, dict):
        return {k: deep_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [deep_convert(x) for x in obj]
    if isinstance(obj, set):
        return [deep_convert(x) for x in obj]
    return obj


def refresh_cache(force=False):
    now = time.time()
    if (not force) and CACHE["payload"] and (now - CACHE["ts"] < CACHE_TTL):
        return CACHE["payload"], CACHE["report"], CACHE["raw"]

    payload, report, raw = get_report()
    CACHE["ts"] = now
    CACHE["payload"] = payload
    CACHE["report"] = report
    CACHE["raw"] = raw
    return payload, report, raw


@app.route("/healthz")
def health():
    return "OK", 200


@app.route("/json")
def json_api():
    payload, report, raw = refresh_cache()
    safe_payload = deep_convert(payload)
    if report is None:
        return jsonify({"error": "model returned non-JSON", "raw": raw, "payload": safe_payload}), 200
    return jsonify({"payload": safe_payload, "report": deep_convert(report)}), 200


@app.route("/refresh")
def refresh():
    refresh_cache(force=True)
    return redirect(url_for("index"))


@app.route("/cron")
def cron():
    payload, report, raw = refresh_cache(force=True)
    # write a snapshot (Render health/cron-friendly)
    try:
        with open("/tmp/report.json", "w", encoding="utf-8") as f:
            json.dump({"payload": payload, "report": report, "raw": raw}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return "OK", 200


@app.route("/")
def index():
    payload, data, raw = refresh_cache()

    # name → float projection
    wk_by_name = {p["name"]: p["wk_proj"] for p in (payload["my_starters"] + payload["my_bench"])}
    # name → player dict
    players_by_name = {p["name"]: p for p in (payload["my_starters"] + payload["my_bench"])}

    # For hybrid hint: map "sit name" → swap object (from local optimizer)
    local_hints = {sw["sit"]["name"]: sw for sw in payload.get("local_swap_candidates", [])}

    if data is None:
        return render_template(
            "index.html",
            payload=payload,
            report=None,
            raw_text=raw,
            wk_by_name=wk_by_name,
            players_by_name=players_by_name,
            start_sit_by_slot={},
            local_hints=local_hints,
        ), 200

    start_sit_by_slot = {row["slot"]: row for row in data.get("start_sit", [])}

    return render_template(
        "index.html",
        payload=payload,
        report=data,
        raw_text=None,
        wk_by_name=wk_by_name,
        players_by_name=players_by_name,
        start_sit_by_slot=start_sit_by_slot,
        local_hints=local_hints,
    ), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
