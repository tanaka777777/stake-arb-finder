"""
Game matcher - matches games across Polymarket and Stake.
"""

from models import Game


def match_games(poly_games: list[Game], stake_games: list[Game], adapter) -> list[tuple[Game, Game, bool]]:
    """
    Match Polymarket games to Stake games using normalized team names.
    Returns list of (poly_game, stake_game, is_swapped) tuples.

    is_swapped is True when Poly's team order is opposite of Stake's:
    - Normal: Poly away = Stake away, Poly home = Stake home
    - Swapped: Poly away = Stake home, Poly home = Stake away
    """
    matched = []
    used_stake = set()

    for pg in poly_games:
        poly_away = adapter.normalize_team(pg.away_team)
        poly_home = adapter.normalize_team(pg.home_team)

        for i, sg in enumerate(stake_games):
            if i in used_stake:
                continue

            stake_away = adapter.normalize_team(sg.away_team)
            stake_home = adapter.normalize_team(sg.home_team)

            # Check normal order: Poly away = Stake away, Poly home = Stake home
            if poly_away == stake_away and poly_home == stake_home:
                matched.append((pg, sg, False))  # Not swapped
                used_stake.add(i)
                break

            # Check swapped order: Poly away = Stake home, Poly home = Stake away
            if poly_away == stake_home and poly_home == stake_away:
                matched.append((pg, sg, True))  # Swapped
                used_stake.add(i)
                break

    return matched
