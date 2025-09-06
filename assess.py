# assess.py
# ESPN → payload → OpenAI JSON advice
# Returns: (payload: dict, report: dict|None, raw_text_if_fail: str)

import os, json
from dotenv import load_dotenv
from espn_api.football import League
from openai import OpenAI


# ---------------- ENV ----------------
def load_env():
    load_dotenv(override=True)
    required = ["LEAGUE_ID", "ESPN_S2", "SWID", "OPENAI_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing env: {', '.join(missing)}")
    return {
        "LEAGUE_ID": int(os.environ["LEAGUE_ID"]),
        "YEAR": int(os.environ.get("YEAR", "2025")),
        "ESPN_S2": os.environ["ESPN_S2"],
        "SWID": os.environ["SWID"],
        "TEAM_NAME": os.environ.get("TEAM_NAME", "AI Replacements"),
        "MODEL": os.environ.get("OPENAI_MODEL", "gpt-4-turbo"),
    }


# ------------- HELPERS ---------------
START_SLOTS = ("QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "D/ST", "K")  # order for display

def normalize_slot(slot: str) -> str:
    if not slot:
        return ""
    s = str(slot).upper().replace(" ", "")
    if s in {"RB/WR/TE", "W/R/T", "WR/RB/TE", "RBWRTE"}:
        return "FLEX"
    if s in {"DST", "DEF"}:
        return "D/ST"
    return slot

def safe_team(val):
    try:
        return str(val.name) if hasattr(val, "name") else str(val)
    except Exception:
        return str(val)

def get_slot(player) -> str:
    raw = getattr(player, "lineupSlot", None) or getattr(player, "slot_position", None) or ""
    return normalize_slot(str(raw))

def week_proj(player, week: int) -> float:
    v = getattr(player, "projected_points", None)
    if isinstance(v, (int, float)) and v >= 0:
        return float(v)
    stats = getattr(player, "stats", None)
    if isinstance(stats, dict):
        wk = stats.get(week) or stats.get(str(week)) or {}
        v = wk.get("projected_points") or wk.get("projected_total_points")
        if isinstance(v, (int, float)) and v >= 0:
            return float(v)
    season = getattr(player, "projected_total_points", None)
    if isinstance(season, (int, float)) and season > 0:
        return float(season) / 17.0
    return 0.0

def pack_player(player, week: int) -> dict:
    return {
        "name": player.name,
        "pos": player.position,
        "team": safe_team(getattr(player, "proTeam", "")),
        "slot": get_slot(player),
        "wk_proj": round(week_proj(player, week), 2),
        "status": getattr(player, "injuryStatus", "") or "",
    }

def sum_proj(players) -> float:
    return round(sum(p["wk_proj"] for p in players), 2)

def bench_swaps(starters: list, bench: list) -> list:
    swaps = []
    bench_by_pos = {
        "QB":  [b for b in bench if b["pos"] == "QB"],
        "RB":  [b for b in bench if b["pos"] == "RB"],
        "WR":  [b for b in bench if b["pos"] == "WR"],
        "TE":  [b for b in bench if b["pos"] == "TE"],
        "D/ST":[b for b in bench if b["pos"] == "D/ST"],
        "K":   [b for b in bench if b["pos"] == "K"],
        "FLEX":[b for b in bench if b["pos"] in {"RB", "WR", "TE"}],
    }
    for s in starters:
        pool = bench_by_pos["FLEX"] if s["slot"] == "FLEX" else bench_by_pos.get(s["pos"], [])
        if not pool:
            continue
        best = max(pool, key=lambda x: x["wk_proj"], default=None)
        if best and best["wk_proj"] > s["wk_proj"]:
            swaps.append({
                "slot": s["slot"],
                "sit": {"name": s["name"], "pos": s["pos"], "wk_proj": s["wk_proj"]},
                "start": {"name": best["name"], "pos": best["pos"], "wk_proj": best["wk_proj"]},
                "delta": round(best["wk_proj"] - s["wk_proj"], 2),
            })
    swaps.sort(key=lambda x: x["delta"], reverse=True)
    return swaps


# -------------- CORE -----------------
def build_payload() -> dict:
    cfg = load_env()
    league = League(
        league_id=cfg["LEAGUE_ID"],
        year=cfg["YEAR"],
        espn_s2=cfg["ESPN_S2"],
        swid=cfg["SWID"],
    )
    week = league.current_week

    me = next(t for t in league.teams if t.team_name == cfg["TEAM_NAME"])

    sb = league.scoreboard(week)
    my_match = next(m for m in sb if me in (m.home_team, m.away_team))
    opp = my_match.away_team if my_match.home_team == me else my_match.home_team

    my_all = [pack_player(p, week) for p in me.roster]
    start_labels = set(START_SLOTS)
    my_starters = [p for p in my_all if p["slot"] in start_labels]
    my_bench = [p for p in my_all if p["slot"] not in start_labels]

    slot_rank = {slot: i for i, slot in enumerate(START_SLOTS)}
    my_starters_sorted = sorted(my_starters, key=lambda x: slot_rank.get(x["slot"], 999))

    opp_all = [pack_player(p, week) for p in opp.roster]
    opp_starters = [p for p in opp_all if p["slot"] in start_labels]

    my_proj_raw = my_match.home_score if my_match.home_team == me else my_match.away_score
    opp_proj_raw = my_match.away_score if my_match.home_team == me else my_match.home_score
    my_proj = round(float(my_proj_raw or 0.0), 2)
    opp_proj = round(float(opp_proj_raw or 0.0), 2)
    if my_proj == 0.0 and opp_proj == 0.0:
        my_proj, opp_proj = sum_proj(my_starters_sorted), sum_proj(opp_starters)

    underdog = my_proj < opp_proj

    fa = [
        pack_player(p, week) for p in league.free_agents(size=80)
        if getattr(p, "position", "") in {"QB", "RB", "WR", "TE", "K", "D/ST"}
    ]
    fa.sort(key=lambda x: x["wk_proj"], reverse=True)
    fa_top = fa[:40]

    local_swaps = bench_swaps(my_starters_sorted, my_bench)

    payload = {
        "week": int(week),
        "league": str(league.settings.name),
        "team": str(me.team_name),
        "opponent": str(opp.team_name),
        "proj": {"me": float(my_proj), "opp": float(opp_proj)},
        "strategy_bias": "ceiling" if underdog else "floor",
        "scoring": "0.5 PPR; ESPN defaults.",
        "my_starters": my_starters_sorted,
        "my_bench": sorted(my_bench, key=lambda x: x["wk_proj"], reverse=True),
        "free_agents_top": fa_top,
        "local_swap_candidates": local_swaps,
        "_model": cfg["MODEL"],
    }
    return payload


def get_report():
    """
    Return (payload, report_dict, raw_text_fallback).
    """
    payload = build_payload()
    client = OpenAI()
    model = payload["_model"]

    schema = {
        "type": "object",
        "properties": {
            "start_sit": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "slot": {"type": "string"},
                        "starter": {"type": "string"},
                        "verdict": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["slot", "starter", "verdict", "reason"],
                },
            },
            "bench_ranked": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "pos": {"type": "string"},
                        "wk_proj": {"type": "number"},
                    },
                    "required": ["name", "pos", "wk_proj"],
                },
            },
            "waivers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "add": {"type": "string"},
                        "drop": {"type": "string"},
                        "why": {"type": "string"},
                    },
                    "required": ["add", "drop", "why"],
                },
            },
            "trades": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "why": {"type": "string"},
                    },
                    "required": ["target", "why"],
                },
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["start_sit", "bench_ranked", "waivers", "trades", "risks"],
    }

    system_msg = (
        "You are an expert fantasy football analyst. "
        "Return concise WEEKLY decisions. "
        "If underdog: bias toward ceiling; if favorite: bias toward floor. "
        "Output ONLY a single valid JSON object — no prose, no code fences."
    )

    user_prompt = f"""
Output STRICT JSON (no prose; no backticks) that validates against this schema:

Schema:
{json.dumps(schema)}

Guidelines:
- For each starter slot in my_starters, produce one row in start_sit.
- 'verdict' must be 'Start' or 'Bench for <bench player name>'.
- bench_ranked: return my_bench sorted by highest wk_proj.
- waivers: pick the 3 best adds from free_agents_top and pair each with a real bench drop.
- trades: 1–2 realistic targets with short reasons.
- risks: 3 short bullets.

League JSON:
{json.dumps(payload)}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or ""

    def strip_code_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = "\n".join(t.splitlines()[1:])
        if t.endswith("```"):
            t = "\n".join(t.splitlines()[:-1])
        t = t.strip()
        if "{" in t and "}" in t:
            start = t.find("{")
            end = t.rfind("}")
            t = t[start : end + 1]
        return t

    cleaned = strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
        return payload, data, ""
    except Exception:
        return payload, None, raw
