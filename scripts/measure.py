"""Absolute Stärke eines trainierten Checkpoints messen (Schritt 2.5, Ergänzung).

Der Trainings-Loop loggt nur die Quote gegen **Greedy** – und Greedy ist bei
Othello so schwach, dass ein brauchbares Netz dort sofort an die Decke stößt
(100 %), also kein Signal mehr liefert. Dieses Skript stellt einen *nicht
sättigenden* Maßstab daneben: das Netz gegen **reines MCTS** (ohne Netz) bei
mehreren Simulationsbudgets, plus optional gegen ein **früheres Ich** (anderer
Checkpoint). Damit lässt sich sagen, *wie* stark das Modell wirklich ist und ob
es über die Iterationen zugelegt hat.

Beispiele:
    python scripts/measure.py                                   # best.pt vs Random/Greedy/MCTS
    python scripts/measure.py --checkpoint checkpoints/best.pt --games 60
    python scripts/measure.py --mcts-sims 50,150,400 --sims 64
    python scripts/measure.py --vs-checkpoint checkpoints/iter_005.pt  # Fortschritt ggü. früh

Die Netz-Seite wird gebündelt ausgewertet (parallele Arena), die Gegner ziehen
inline – ein Match von 60 Partien dauert daher nur Sekunden.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from az.arena_parallel import AgentPlayer, NetPlayer, play_match_parallel  # noqa: E402
from az.checkpoint import load_checkpoint  # noqa: E402
from config import DEFAULT_RUN, MCTSConfig  # noqa: E402

from agents.mcts import MCTSAgent  # noqa: E402
from agents.simple import GreedyAgent, RandomAgent  # noqa: E402


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _net_player(net, name, sims, temperature_moves, device, seed):
    """Netz-Spieler mit explorativer Eröffnung (sonst wären alle Partien identisch)."""
    return NetPlayer(
        net, name, MCTSConfig(n_simulations=sims),
        temperature=1.0, temperature_moves=temperature_moves, device=device, seed=seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Stärke eines Othello-Checkpoints messen")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Pfad zum zu messenden Netz (.pt)")
    parser.add_argument("--games", type=int, default=40, help="Partien pro Match (gerade Zahl)")
    parser.add_argument("--sims", type=int, default=DEFAULT_RUN.mcts.n_simulations,
                        help="MCTS-Simulationen des Netz-Spielers pro Zug")
    parser.add_argument("--mcts-sims", type=str, default="50,150,400",
                        help="Sim-Budgets der reinen-MCTS-Gegner (kommagetrennt)")
    parser.add_argument("--temperature-moves", type=int, default=10,
                        help="explorative Eröffnungszüge (Partievielfalt)")
    parser.add_argument("--vs-checkpoint", type=str, default=None,
                        help="optional: zweiter Checkpoint als Gegner (Fortschritt ggü. früh)")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (Default: auto)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"FEHLER: Checkpoint nicht gefunden: {ckpt_path.resolve()}")
        print("Tipp: aus dem Verzeichnis starten, in dem 'checkpoints/' liegt "
              "(oder --checkpoint mit vollem Pfad).")
        return 1

    net, extra = load_checkpoint(ckpt_path, device)
    size = net.board_size
    it = extra.get("iteration", "?")
    print(f"Modell: {ckpt_path} | Brett {size}x{size} | iter={it} | "
          f"Netz-Sims={args.sims} | {args.games} Partien/Match | Device={device}\n")

    net_player = _net_player(net, "net", args.sims, args.temperature_moves, device, args.seed)

    # Gegner-Riege: von trivial bis ernst. Reines MCTS ist der eigentliche Maßstab.
    opponents: list[tuple[str, object]] = [
        ("Random", AgentPlayer(RandomAgent(seed=args.seed))),
        ("Greedy", AgentPlayer(GreedyAgent(seed=args.seed))),
    ]
    for s in _parse_int_list(args.mcts_sims):
        opponents.append((f"MCTS({s})", AgentPlayer(MCTSAgent(n_simulations=s, seed=args.seed))))

    if args.vs_checkpoint is not None:
        vs_net, vs_extra = load_checkpoint(args.vs_checkpoint, device)
        vs_it = vs_extra.get("iteration", "?")
        vs_player = _net_player(vs_net, f"ckpt(iter={vs_it})", args.sims,
                                args.temperature_moves, device, args.seed + 1)
        opponents.append((f"Netz@iter={vs_it}", vs_player))

    print(f"{'Gegner':<18}{'Quote':>8}   {'W/L/D':>10}")
    print("-" * 42)
    for label, opp in opponents:
        result = play_match_parallel(net_player, opp, args.games, size, device=device)
        wld = f"{result.wins}/{result.losses}/{result.draws}"
        print(f"{label:<18}{result.win_rate:>7.1%}   {wld:>10}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
