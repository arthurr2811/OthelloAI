"""Misst stärke des aktuiellen checkpoints gegen einreines MCTS

Performance improvment: Die Netz-Seite wird gebündelt ausgewertet (parallele Arena), die Gegner ziehen
inline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from az.arena_parallel import AgentPlayer, NetPlayer, play_match_parallel  # noqa: E402
from az.checkpoint import load_checkpoint  # noqa: E402
from config import DEFAULT_CHECKPOINT, DEFAULT_RUN, MCTSConfig, project_path  # noqa: E402

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
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Pfad zum zu messenden Netz (.pt); relativ = ab Projektwurzel")
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

    ckpt_path = project_path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"FEHLER: Checkpoint nicht gefunden: {ckpt_path}")
        print("Tipp: --checkpoint mit einem existierenden Pfad angeben "
              "(relativ zur Projektwurzel oder absolut).")
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
        vs_net, vs_extra = load_checkpoint(project_path(args.vs_checkpoint), device)
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
