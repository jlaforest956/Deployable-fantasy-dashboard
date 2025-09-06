# assess.py
# Build a payload from ESPN → ask OpenAI for JSON advice → return (payload, report, raw_text)

import os
import json
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from espn_api.football import League

# OpenAI is optional; we’ll skip AI if the key/model aren’t present
try:
    from openai import OpenAI  # new SDK
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


# ---------------- ENV ----------------
def load_env() -> Dict[str, Any]:
    load_dotenv(override=True)
    # ESPN creds must exist
    required = ["LEAGUE_ID", "ESPN_S2", "SWID"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing env: {', '.join(missing)}")

    cfg = {
        "LEAGUE_ID": int(os.environ["LEAGUE_ID"]),
        "YEAR": int(os.environ.get("YEAR", "2025")),
        "ESPN_S2": os.environ["ESPN_S2"],
        "SWID": os.environ["SWID"],
        "TEAM_NAME": os.environ.get("TEAM_NAME", "AI Replacements"),
        # OpenAI (optional)
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "MODEL": os.environ.get("OPENAI_MODEL", "gpt-4-turbo"),
        # Free-agent pool size
        "FA_SIZE": int(os.environ.get("FA_SIZE", "80")),
    }
    return cfg


# ------------- HELPERS ---------------
START_SLOTS: Tuple[str, ...] = ("QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "D/ST", "K")

def normalize_slot(slot: Any) -> str:
    s = (str(slot) if slot is not None else "").upper().replace(" ", "")
    if s in {"RB/WR/TE", "W/R/T", "WR/RB/TE", "RBWRTE"}:
        return "FLEX"
    if s in {"DST", "DEF"}:
        return "D/ST"
    # ESPN sometimes returns "Bench", "IR", "OP"
    if s == "BENCH": return "BE"
    if s == "IR": return "IR"
    return (str(slot) if slot else "")

def safe_team(val: Any) -> str:
    try:
        if hasattr(val, "name"):
            return str(val.name)
        return str(val)
    except Exception:
        return ""

def get_slot(player: Any) -> str:
    # espn_api Player has .lineupSlot most of the time
    raw = getattr(player, "lineupSlot", None) or getattr(player, "slot_position", None) or ""
    return normalize_slot(raw)

def get_bye(player: Any) -> Optional[int]:
    # Different espn_api versions expose bye differently. Try a few.
    for attr in ("bye_week", "byeWeek", "bye"):
        v = getattr(player, attr, None)
        if isinstance(v, int):
            return v
        # sometimes nested on proTeam?
    # last resort: None
    return None

def week_proj(player: Any, week: int) -> float:
    # Try explicit projected points for this week
    v = getattr(player, "projected_points", None)
    if isinstance(v, (int, float)) and v >= 0:
        return float(v)
    stats = getattr(player, "stats", None)
    if isinstance(stats, dict):
        wk = stats.get(week) or stats.get(str(week)) or {}
        v = wk.get("projected_points") or wk.get("projected_total_points")
        if isinstance(v, (int, float)) and v >= 0:
            return float(v)
    # fallback: season projection per-game estimate
    season = getattr(player, "projected_total_points", None)
    if isinstance(season, (int, float)) and season > 0:
        return float(season) / 17.0
    return 0.0

def week_actual(player: Any, week: int) -> float:
    # Pull actual points if game already played
    stats = getattr(player, "stats", None)
    if isinstance(stats, dict):
        wk = stats.get(week) or stats.get(str(week)) or {}
        for key in ("points", "applied_total", "total_points"):
            v = wk.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    v = getattr(player, "points", None)
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0

def pack_player(player: Any, week: int) -> Dict[str, Any]:
    return {
        "name": getattr(player, "name", ""),
        "pos": getattr(player, "position", ""),
        "team": safe_team(getattr(player, "proTeam", "")),
        "slot": get_slot(player),
        "wk_proj": round(week_proj(player, week), 2),
        "wk_actual": round(week_actual(player, week), 2),
        "status": getattr(player, "injuryStatus", "") or "",
        "bye": get_bye(player),  # may be None
    }

def sum_proj(players: List[Dict[str, Any]]) -> float:
    return round(sum(p.get("wk_proj", 0.0) for p in players), 2)

def bench_swaps(starters: List[Dict[str, Any]], bench: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    swaps: List[Dict[str, Any]] = []
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
def build_payload() -> Dict[str, Any]:
    cfg = load_env()
    league = League(
        league_id=cfg["LEAGUE_ID"],
        year=cfg["YEAR"],
        espn_s2=cfg["ESPN_S2"],
        swid=cfg["SWID"],
    )

    # Prefer league.current_week (works during season)
    week = int(getattr(league, "current_week", 1) or 1)

    # Me & Opponent
    me = next((t for t in league.teams if t.team_name == cfg["TEAM_NAME"]), None)
    if me is None:
        raise RuntimeError(f"TEAM_NAME '{cfg['TEAM_NAME']}' not found in league.")

    sb = league.scoreboard(week)
    my_match = next((m for m in sb if me in (m.home_team, m.away_team)), None)
    if my_match is None:
        # Offseason or schedule not ready; create a safe placeholder
        opp_name = "TBD"
        opp_roster = []
        raw_me = raw_opp = 0.0
    else:
        opp = my_match.away_team if my_match.home_team == me else my_match.home_team
        opp_name = opp.team_name
        opp_roster = [pack_player(p, week) for p in opp.roster]
        raw_me = my_match.home_score if my_match.home_team == me else my_match.away_score
        raw_opp = my_match.away_score if my_match.home_team == me else my_match.home_score

    # My roster
    my_all = [pack_player(p, week) for p in me.roster]

    start_labels = set(START_SLOTS)
    my_starters = [p for p in my_all if p["slot"] in start_labels]
    my_bench = [p for p in my_all if p["slot"] not in start_labels]

    # Sort starters by the display order
    slot_rank = {slot: i for i, slot in enumerate(START_SLOTS)}
    my_starters_sorted = sorted(my_starters, key=lambda x: slot_rank.get(x["slot"], 999))

    # Opponent starters (for context)
    opp_starters = [p for p in opp_roster if p["slot"] in start_labels]

    # Projections: use ESPN live projections if available; else sum ours
    try:
        my_proj = round(float(raw_me or 0.0), 2)
        opp_proj = round(float(raw_opp or 0.0), 2)
    except Exception:
        my_proj = opp_proj = 0.0

    if my_proj == 0.0 and opp_proj == 0.0:
        my_proj = sum_proj(my_starters_sorted)
        opp_proj = sum_proj(opp_starters)

    underdog = my_proj < opp_proj

    # Free agents
    fa = [
        pack_player(p, week)
        for p in league.free_agents(size=cfg["FA_SIZE"])
        if getattr(p, "position", "") in {"QB", "RB", "WR", "TE", "K", "D/ST"}
    ]
    fa.sort(key=lambda x: x["wk_proj"], reverse=True)
    fa_top = fa[:40]

    # Bench swap ideas
    local_swaps = bench_swaps(my_starters_sorted, my_bench)

    payload: Dict[str, Any] = {
        "week": week,
        "league": str(league.settings.name),
        "team": str(me.team_name),
        "opponent": opp_name,
        "proj": {"me": float(my_proj), "opp": float(opp_proj)},
        "strategy_bias": "ceiling" if underdog else "floor",
        "scoring": "0.5 PPR; ESPN defaults.",
        "my_starters": my_starters_sorted,
        "my_bench": sorted(my_bench, key=lambda x: x["wk_proj"], reverse=True),
        "free_agents_top": fa_top,
        "local_swap_candidates": local_swaps,
        "_model": os.environ.get("OPENAI_MODEL", "gpt-4-turbo"),
    }
    return payload


def _call_openai(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Returns (report_dict_or_None, raw_text).
    """
    if not _OPENAI_AVAILABLE:
        return None, "OpenAI SDK not available in environment."
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None, "OPENAI_API_KEY not set; skipping AI analysis."

    client = OpenAI(api_key=api_key)
    model = payload.get("_model", "gpt-4-turbo")

    # Schema + prompt with guardrails against duplicate starters
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "start_sit": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "slot": {"type": "string"},
                        "starter": {"type": "string"},
                        "verdict": {"type": "string"},   # "Start" or "Bench for <name>"
                        "reason": {"type": "string"},
                        "proj": {"type": "number"}
                    },
                    "required": ["slot", "starter", "verdict", "reason"]
                }
            },
            "bench_ranked": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "pos": {"type": "string"},
                        "wk_proj": {"type": "number"},
                        "bye": {"type": ["integer", "null"]},
                        "status": {"type": "string"}
                    },
                    "required": ["name", "pos", "wk_proj"]
                }
            },
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
            "risks": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["summary", "start_sit", "bench_ranked", "waivers", "trades", "risks"]
    }

    guidelines = """Rules:
- Produce one unique starter per starter slot. Do NOT repeat the same player in multiple slots.
- 'verdict' must be 'Start' or 'Bench for <bench player name>'.
- Prefer 'ceiling' if underdog; prefer 'floor' if favorite.
- Use the provided projections; include the chosen starter's proj in each row.
- Use only players actually present in my_starters or my_bench; for FLEX, eligible bench are RB/WR/TE.
- Waivers: pick 3 realistic adds from free_agents_top with real drops from my_bench (not from starters).
- Keep the 'summary' to 2–4 sentences max: matchup, risk, top moves, overall outlook.
- Output a single valid JSON object; no code fences, no prose outside JSON.
"""

    system = "You are an expert fantasy football analyst. Return concise weekly advice as JSON only."
    user = f"""
Schema (JSON Schema):
{json.dumps(schema)}

Guidelines:
{guidelines}

League Payload:
{json.dumps(payload)}
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()

        # Strip possible code fences just in case
        t = raw
        if t.startswith("```"):
            lines = t.splitlines()
            if len(lines) >= 2:
                t = "\n".join(lines[1:])
                if t.endswith("```"):
                    t = t[:-3]
        # Extract JSON block
        if "{" in t and "}" in t:
            t = t[t.find("{"): t.rfind("}") + 1]

        data = json.loads(t)
        return data, raw
    except Exception as e:
        return None, f"AI error: {e}"


def get_report() -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], str]:
    """
    Return (payload, report_dict_or_None, raw_text_or_error).
    """
    payload = build_payload()
    report, raw = _call_openai(payload)
    return payload, report, raw
