"""
Arbitrage detector - finds arb opportunities between Polymarket and Bovada.
"""

import hashlib
from models import Game, Market, Arb


def calculate_arb(odds_a: float, odds_b: float) -> tuple[float, float, float]:
    """
    Calculate arbitrage profit and optimal stakes.
    Returns (profit_pct, stake_a_pct, stake_b_pct) or (0, 0, 0) if no arb.
    """
    if odds_a <= 1 or odds_b <= 1:
        return 0, 0, 0

    implied_sum = (1 / odds_a) + (1 / odds_b)

    if implied_sum >= 1:
        return 0, 0, 0

    profit_pct = (1 - implied_sum) * 100
    stake_a_pct = (1 / odds_a) / implied_sum * 100
    stake_b_pct = (1 / odds_b) / implied_sum * 100

    return profit_pct, stake_a_pct, stake_b_pct


def make_arb_id(game_id: str, market_type: str, line: float | None, poly_side: str, tp_side: str) -> str:
    """Generate unique ID for an arb opportunity."""
    key = f"{game_id}:{market_type}:{line}:{poly_side}:{tp_side}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def find_arbs(poly_game: Game, stake_game: Game, is_swapped: bool = False, return_all: bool = False) -> list[Arb]:
    """
    Find all arbitrage opportunities between a matched game pair.
    Checks both directions for each market type.

    is_swapped: When True, Poly's team order is opposite of Bovada's:
      - Normal (False): Poly YES = away wins, compare vs Bovada HOME (home wins) - OPPOSITE
      - Swapped (True): Poly YES = away wins (but away = Bovada home), compare vs Bovada AWAY - OPPOSITE

    return_all: When True, return ALL comparisons (including non-profitable) for debugging.
                Default False = only return profitable arbs (existing behavior).
    """
    arbs = []
    game_display = f"{poly_game.away_team} @ {poly_game.home_team}"
    poly_url = poly_game.url
    stake_url = stake_game.url

    # When swapped, we need to compare against the OPPOSITE Bovada sides:
    # - Normal: Poly YES (away) vs Bovada HOME, Poly NO (home) vs Bovada AWAY
    # - Swapped: Poly YES (away=Bovada home) vs Bovada AWAY, Poly NO (home=Bovada away) vs Bovada HOME
    if is_swapped:
        stake_side_for_poly_yes = "away"  # Poly away = Bovada home, so compare to Bovada away for opposite
        stake_side_for_poly_no = "home"   # Poly home = Bovada away, so compare to Bovada home for opposite
    else:
        stake_side_for_poly_yes = "home"  # Normal: Poly away != Bovada home, so compare to Bovada home for opposite
        stake_side_for_poly_no = "away"   # Normal: Poly home != Bovada away, so compare to Bovada away for opposite

    # Moneyline
    poly_mls = poly_game.markets.get("moneyline", [])
    stake_mls = stake_game.markets.get("moneyline", [])

    if poly_mls and stake_mls:
        pm = poly_mls[0]
        sm = stake_mls[0]

        # Check 1: Poly YES vs appropriate Bovada side
        profit, stake_a, stake_b = calculate_arb(pm.outcomes["yes"], sm.outcomes[stake_side_for_poly_yes])
        if profit > 0 or return_all:
            arbs.append(Arb(
                id=make_arb_id(poly_game.id, "moneyline", None, "yes", stake_side_for_poly_yes),
                game_id=poly_game.id, game_display=game_display, sport=poly_game.sport,
                market_type="moneyline", line=None, profit_pct=profit,
                poly_side="yes", poly_odds=pm.outcomes["yes"], poly_stake_pct=stake_a,
                poly_size=pm.sizes.get("yes") if pm.sizes else None,
                tp_side=stake_side_for_poly_yes, tp_odds=sm.outcomes[stake_side_for_poly_yes], tp_stake_pct=stake_b,
                poly_url=poly_url, tp_url=stake_url,
            ))

        # Check 2: Poly NO vs appropriate Bovada side
        profit, stake_a, stake_b = calculate_arb(pm.outcomes["no"], sm.outcomes[stake_side_for_poly_no])
        if profit > 0 or return_all:
            arbs.append(Arb(
                id=make_arb_id(poly_game.id, "moneyline", None, "no", stake_side_for_poly_no),
                game_id=poly_game.id, game_display=game_display, sport=poly_game.sport,
                market_type="moneyline", line=None, profit_pct=profit,
                poly_side="no", poly_odds=pm.outcomes["no"], poly_stake_pct=stake_a,
                poly_size=pm.sizes.get("no") if pm.sizes else None,
                tp_side=stake_side_for_poly_no, tp_odds=sm.outcomes[stake_side_for_poly_no], tp_stake_pct=stake_b,
                poly_url=poly_url, tp_url=stake_url,
            ))

    # Spreads: Match by actual line value, considering which team each spread is for
    # Polymarket: side="home" means YES=home covers that line, NO=away covers opposite
    # Bovada: line is always from HOME perspective, home=home covers, away=away covers
    stake_spreads = {s.line: s for s in stake_game.markets.get("spreads", [])}

    for ps in poly_game.markets.get("spreads", []):
        poly_line = ps.line
        poly_side = ps.side  # "home" or "away"

        if not poly_side:
            # Skip spreads where we don't know which team it's for
            continue

        # Account for team swap: when swapped, Poly's home=Bovada's away and vice versa
        # effective_side is what poly_side means in Bovada's frame of reference
        if is_swapped:
            effective_side = "away" if poly_side == "home" else "home"
        else:
            effective_side = poly_side

        # Find matching Bovada spread
        # If spread is for Bovada HOME team at line X, Bovada will have HOME at line X
        # If spread is for Bovada AWAY team at line X, we need Bovada where AWAY has line X
        #   but Bovada stores from HOME perspective, so Bovada line = -X

        if effective_side == "home":
            # Spread is for Bovada home team
            # Bovada line X: home = X, away = -X
            stake_line = poly_line
        else:  # effective_side == "away"
            # Spread is for Bovada away team
            # Bovada stores from home perspective, so if away has X, home has -X
            # Bovada line -X: home = -X, away = X
            stake_line = -poly_line

        ss = stake_spreads.get(stake_line)
        if not ss:
            continue

        # Now match outcomes:
        # Poly YES = the specified team covers (effective_side in Bovada terms)
        # Poly NO = the other team covers

        if effective_side == "home":
            # Poly YES = Bovada home covers, Bovada home = home covers -> SAME (not arb pair)
            # Poly YES = Bovada home covers, Bovada away = away covers -> OPPOSITE (arb pair)
            # Poly NO = Bovada away covers, Bovada home = home covers -> OPPOSITE (arb pair)
            # Poly NO = Bovada away covers, Bovada away = away covers -> SAME (not arb pair)

            # Check: Poly YES (Bovada home covers) vs Bovada AWAY (away covers) - OPPOSITE
            profit, stake_a, stake_b = calculate_arb(ps.outcomes["yes"], ss.outcomes["away"])
            if profit > 0 or return_all:
                arbs.append(Arb(
                    id=make_arb_id(poly_game.id, "spread", poly_line, "yes", "away"),
                    game_id=poly_game.id, game_display=game_display, sport=poly_game.sport,
                    market_type="spread", line=poly_line, profit_pct=profit,
                    poly_side="yes", poly_odds=ps.outcomes["yes"], poly_stake_pct=stake_a,
                    poly_size=ps.sizes.get("yes") if ps.sizes else None,
                    tp_side="away", tp_odds=ss.outcomes["away"], tp_stake_pct=stake_b,
                    poly_url=poly_url, tp_url=stake_url,
                    tp_line=stake_line,
                ))

            # Check: Poly NO (Bovada away covers) vs Bovada HOME (home covers) - OPPOSITE
            profit, stake_a, stake_b = calculate_arb(ps.outcomes["no"], ss.outcomes["home"])
            if profit > 0 or return_all:
                arbs.append(Arb(
                    id=make_arb_id(poly_game.id, "spread", poly_line, "no", "home"),
                    game_id=poly_game.id, game_display=game_display, sport=poly_game.sport,
                    market_type="spread", line=poly_line, profit_pct=profit,
                    poly_side="no", poly_odds=ps.outcomes["no"], poly_stake_pct=stake_a,
                    poly_size=ps.sizes.get("no") if ps.sizes else None,
                    tp_side="home", tp_odds=ss.outcomes["home"], tp_stake_pct=stake_b,
                    poly_url=poly_url, tp_url=stake_url,
                    tp_line=stake_line,
                ))
        else:  # effective_side == "away"
            # Poly YES = Bovada away covers, Bovada away = away covers -> SAME (not arb pair)
            # Poly YES = Bovada away covers, Bovada home = home covers -> OPPOSITE (arb pair)
            # Poly NO = Bovada home covers, Bovada away = away covers -> OPPOSITE (arb pair)
            # Poly NO = Bovada home covers, Bovada home = home covers -> SAME (not arb pair)

            # Check: Poly YES (Bovada away covers) vs Bovada HOME (home covers) - OPPOSITE
            profit, stake_a, stake_b = calculate_arb(ps.outcomes["yes"], ss.outcomes["home"])
            if profit > 0 or return_all:
                arbs.append(Arb(
                    id=make_arb_id(poly_game.id, "spread", poly_line, "yes", "home"),
                    game_id=poly_game.id, game_display=game_display, sport=poly_game.sport,
                    market_type="spread", line=poly_line, profit_pct=profit,
                    poly_side="yes", poly_odds=ps.outcomes["yes"], poly_stake_pct=stake_a,
                    poly_size=ps.sizes.get("yes") if ps.sizes else None,
                    tp_side="home", tp_odds=ss.outcomes["home"], tp_stake_pct=stake_b,
                    poly_url=poly_url, tp_url=stake_url,
                    tp_line=stake_line,
                ))

            # Check: Poly NO (Bovada home covers) vs Bovada AWAY (away covers) - OPPOSITE
            profit, stake_a, stake_b = calculate_arb(ps.outcomes["no"], ss.outcomes["away"])
            if profit > 0 or return_all:
                arbs.append(Arb(
                    id=make_arb_id(poly_game.id, "spread", poly_line, "no", "away"),
                    game_id=poly_game.id, game_display=game_display, sport=poly_game.sport,
                    market_type="spread", line=poly_line, profit_pct=profit,
                    poly_side="no", poly_odds=ps.outcomes["no"], poly_stake_pct=stake_a,
                    poly_size=ps.sizes.get("no") if ps.sizes else None,
                    tp_side="away", tp_odds=ss.outcomes["away"], tp_stake_pct=stake_b,
                    poly_url=poly_url, tp_url=stake_url,
                    tp_line=stake_line,
                ))

    # Totals: match by line, check over/under crosses
    poly_totals = {t.line: t for t in poly_game.markets.get("totals", [])}
    stake_totals = {t.line: t for t in stake_game.markets.get("totals", [])}

    for line in set(poly_totals.keys()) & set(stake_totals.keys()):
        pt = poly_totals[line]
        st = stake_totals[line]

        # Check 1: Poly OVER (yes) vs Bovada UNDER
        profit, stake_a, stake_b = calculate_arb(pt.outcomes["yes"], st.outcomes["under"])
        if profit > 0 or return_all:
            arbs.append(Arb(
                id=make_arb_id(poly_game.id, "total", line, "yes", "under"),
                game_id=poly_game.id, game_display=game_display, sport=poly_game.sport,
                market_type="total", line=line, profit_pct=profit,
                poly_side="over", poly_odds=pt.outcomes["yes"], poly_stake_pct=stake_a,
                poly_size=pt.sizes.get("yes") if pt.sizes else None,
                tp_side="under", tp_odds=st.outcomes["under"], tp_stake_pct=stake_b,
                poly_url=poly_url, tp_url=stake_url,
            ))

        # Check 2: Poly UNDER (no) vs Bovada OVER
        profit, stake_a, stake_b = calculate_arb(pt.outcomes["no"], st.outcomes["over"])
        if profit > 0 or return_all:
            arbs.append(Arb(
                id=make_arb_id(poly_game.id, "total", line, "no", "over"),
                game_id=poly_game.id, game_display=game_display, sport=poly_game.sport,
                market_type="total", line=line, profit_pct=profit,
                poly_side="under", poly_odds=pt.outcomes["no"], poly_stake_pct=stake_a,
                poly_size=pt.sizes.get("no") if pt.sizes else None,
                tp_side="over", tp_odds=st.outcomes["over"], tp_stake_pct=stake_b,
                poly_url=poly_url, tp_url=stake_url,
            ))

    return arbs
