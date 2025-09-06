# assess.py
# ESPN → normalized payload (starters/bench/FA with live & projections)
# OpenAI → executive summary + waivers/trades/risks (no start/sit from AI)
#
# Returns: (payload: dict, ai: dict|None, raw_text_if_fail: str)

import os, json
from typing import Any, Dict, List, Tuple, Optional
from dotenv import load_dotenv
from espn_api.football import League
from openai import OpenAI

# ---------------- ENV ----------------
def _env() -> Dict[str, Any]:
    load_dotenv(override=True)
    need = ["LEAGUE_ID", "ESPN_S2", "SWID", "OPENAI_API_KEY"]
    miss = [k for k in need if not os.environ.get(k)]
    if miss:
        raise RuntimeError(f"Missing env: {', '.join(miss)}")
    return {
        "LEAGUE_ID": int(os.environ["LEAGUE_ID"]),
        "YEAR": int(os.environ.get("YEAR", "2025")),
        "ESPN_S2": os.environ["ESPN_S2"],
        "SWID": os.environ["SWID"],
        "TEAM_NAME": os.environ.get("TEAM_NAME", "AI Replacements"),
        "MODEL": os.environ.get("OPENAI_MODEL", "gpt-4-turbo"),
    }

# ------------- HELPERS ---------------
# Order we want to display the starting slots
START_SLOTS = ("QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "D/ST", "K")

def _normalize_slot(s: str) -> str:
    s = (s or "").upper().replace(" ", "")
    if s in {"RB/WR/TE", "W/R/T", "WR/RB/TE", "RBWRTE"}:
        return "FLEX"
    if s in {"DST", "DEF"}:
        return "D/ST"
    return s

def _slot_from_player(p) -> str:
    raw = getattr(p, "lineupSlot", None) or getattr(p, "slot_position", None) or ""
    return _normalize_slot(str(raw))

def _team_str(val) -> str:
    try:
        if hasattr(val, "name"):
            return str(val.name)
        return str(val)
    except Exception:
        return str(val)

def _float(x, default=0.0) -> float:
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            return float(x)
    except Exception:
        pass
    return float(default)

def _week_proj(p, week: int) -> float:
    # 1) direct projected_points if present
    v = getattr(p, "projected_points", None)
    if isinstance(v, (int, float)):
        return float(v)

    # 2) stats dict
    stats = getattr(p, "stats", None)
    if isinstance(stats, dict):
        wk = stats.get(week) or stats.get(str(week)) or {}
        v = wk.get("projected_points") or wk.get("projected_total_points") or wk.get("projected")
        if isinstance(v, (int, float)):
            return float(v)

    # 3) season projected / 17
    season = getattr(p, "projected_total_points", None)
    if isinstance(season, (int, float)) and season > 0:
        return float(season) / 17.0

    return 0.0

def _week_actual(p, week: int) -> float:
    # try a few common keys ESPN lib uses
    stats = getattr(p, "stats", None)
    if isinstance(stats, dict):
        wk = stats.get(week) or stats.get(str(week)) or {}
        for k in ("points", "total_points", "applied_total", "actual_points", "scoring_period_points"):
            if k in wk and isinstance(wk[k], (int, float)):
                return float(wk[k])
    # sometimes `points` is directly on the object (rare)
    v = getattr(p, "points", None)
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0

def _pack_player(p, week: int) -> Dict[str, Any]:
    return {
        "name": str(p.name),
        "pos": str(p.position),
        "team": _team_str(getattr(p, "proTeam", "")),
        "slot": _slot_from_player(p),
        "wk_proj": round(_week_proj(p, week), 2),
        "wk_actual": round(_week_actual(p, week), 2),
        "status": getattr(p, "injuryStatus", "") or "",
        "bye": None,  # ESPN lib doesn’t always expose bye per player; we show "None" consistently in UI
    }

def _sum_proj(players: List[Dict[str, Any]]) -> float:
    return round(sum(_float(p.get("wk_proj", 0.0)) for p in players), 2)

def _bench_swaps(starters: List[Dict[str, Any]], bench: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find simple local upgrades: if any bench player outprojects current starter in that slot."""
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
        best = max(pool, key=lambda x: _float(x["wk_proj"]), default=None)
        if best and _float(best["wk_proj"]) > _float(s["wk_proj"]):
            swaps.append({
                "slot": s["slot"],
                "sit": {"name": s["name"], "pos": s["pos"], "wk_proj": _float(s["wk_proj"])},
                "start": {"name": best["name"], "pos": best["pos"], "wk_proj": _float(best["wk_proj"])},
                "delta": round(_float(best["wk_proj"]) - _float(s["wk_proj"]), 2),
            })
    swaps.sort(key=lambda x: x["delta"], reverse=True)
    return swaps

# -------------- CORE -----------------
def build_payload() -> Dict[str, Any]:
    cfg = _env()
    league = League(
        league_id=cfg["LEAGUE_ID"],
        year=cfg["YEAR"],
        espn_s2=cfg["ESPN_S2"],
        swid=cfg["SWID"],
    )
    week = int(league.current_week)

    # My team & opponent
    me = next(t for t in league.teams if t.team_name == cfg["TEAM_NAME"])
    sb = league.scoreboard(week)
    my_match = next(m for m in sb if me in (m.home_team, m.away_team))
    opp = my_match.away_team if my_match.home_team == me else my_match.home_team

    # Pack rosters
    my_all = [_pack_player(p, week) for p in me.roster]
    opp_all = [_pack_player(p, week) for p in opp.roster]

    # Starters/bench using actual lineupSlot (prevents duplicates & respects two RB/two WR)
    start_label_set = {"QB", "RB", "WR", "TE", "FLEX", "D/ST", "K"}
    my_starters = [p for p in my_all if p["slot"] in start_label_set]
    my_bench    = [p for p in my_all if p["slot"] not in start_label_set]

    # Sort starters into the fixed slot order
    order_index = {slot: i for i, slot in enumerate(START_SLOTS)}
    # stable sort: by slot index then by projected points desc (so the higher RB/WR is listed first)
    my_starters_sorted = sorted(
        my_starters, key=lambda x: (order_index.get(x["slot"], 99), -_float(x["wk_proj"]))
    )

    opp_starters = [p for p in opp_all if p["slot"] in start_label_set]

    # Projections: ESPN sometimes returns 0.0 until closer to kickoff; fallback to our sum if 0–0
    my_proj_raw  = my_match.home_score if my_match.home_team == me else my_match.away_score
    opp_proj_raw = my_match.away_score if my_match.home_team == me else my_match.home_score
    my_proj = round(_float(my_proj_raw), 2)
    opp_proj = round(_float(opp_proj_raw), 2)
    if my_proj == 0.0 and opp_proj == 0.0:
        my_proj, opp_proj = _sum_proj(my_starters_sorted), _sum_proj(opp_starters)

    underdog = my_proj < opp_proj

    # Free agents
    fas = [
        _pack_player(p, week) for p in league.free_agents(size=80)
        if getattr(p, "position", "") in {"QB", "RB", "WR", "TE", "K", "D/ST"}
    ]
    fas.sort(key=lambda x: _float(x["wk_proj"]), reverse=True)
    fas_top = fas[:40]

    swaps = _bench_swaps(my_starters_sorted, my_bench)

    return {
        "week": week,
        "league": str(league.settings.name),
        "team": str(me.team_name),
        "opponent": str(opp.team_name),
        "proj": {"me": my_proj, "opp": opp_proj},
        "strategy_bias": "ceiling" if underdog else "floor",
        "scoring": "0.5 PPR; ESPN defaults.",
        "my_starters": my_starters_sorted,
        "my_bench": sorted(my_bench, key=lambda x: _float(x["wk_proj"]), reverse=True),
        "free_agents_top": fas_top,
        "local_swap_candidates": swaps,
        "_model": _env()["MODEL"],
    }

def get_report() -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], str]:
    """
    Return (payload, ai_dict, raw_text_if_fail).
    We ask OpenAI only for: executive_summary, waivers, trades, risks.
    Start/Sit is computed locally to avoid duplicates / illegal lineups.
    """
    payload = build_payload()
    client = OpenAI()
    model = payload["_model"]

    schema = {
        "type": "object",
        "properties": {
            "executive_summary": {"type": "string"},
            "waivers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "add": {"type": "string"},
                        "drop": {"type": "string"},
                        "why": {"type": "string"}
                    },
                    "required": ["add", "drop", "why"]
                }
            },
            "trades": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string"},
                        "why": {"type": "string"}
                    },
                    "required": ["target", "why"]
                }
            },
            "risks": {"type": "array", "items": {"type": "string"}}
        }
    }

    system = (
        "You are a precise fantasy football analyst. "
        "Respond with a SINGLE JSON object (no prose, no code fences)."
    )
    user = f"""
Return STRICT JSON that validates against this schema:

{json.dumps(schema)}

Write:
- executive_summary: 2–4 sentences tailored to the matchup and 'strategy_bias'.
  Mention swing spots (e.g., bench upgrades from local_swap_candidates), injuries/status,
  and where the edge likely comes from.
- waivers: pick 3 smart adds from free_agents_top and pair with realistic drops from my_bench.
- trades: 1–2 realistic targets with short reasons.
- risks: 2–4 bullets highlighting lineup/volatility/weather/injury concerns.

League JSON (for context):
{json.dumps(payload)}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or ""

    def strip_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = "\n".join(t.splitlines()[1:])
        if t.endswith("```"):
            t = "\n".join(t.splitlines()[:-1])
        if "{" in t and "}" in t:
            t = t[t.find("{"): t.rfind("}") + 1]
        return t

    try:
        data = json.loads(strip_fences(raw))
        return payload, data, ""
    except Exception:
        return payload, None, raw
