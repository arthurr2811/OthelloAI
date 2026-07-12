"""Multiprocessing-Self-Play (Hebel C aus dem README): Partien über CPU-Kerne verteilen.

Der gebündelte Self-Play (:mod:`az.selfplay_parallel`) ist Single-Core: MCTS-Baum
und Engine laufen in *einem* Python-Prozess, und genau diese CPU-Arbeit ist der
Engpass – nicht die GPU. Hier teilen sich ``n_workers`` Prozesse die Partien
einer Iteration; jeder Prozess fährt intern denselben gebündelten Scheduler mit
eigener GPU-Inferenz (eigener CUDA-Kontext). Die GPU verkraftet mehrere kleine
Inferenz-Ströme problemlos.

Der Pool ist **persistent** (ein Prozess-Start pro Trainingslauf, nicht pro
Iteration), weil jeder Spawn auf Windows torch + CUDA neu initialisiert
(mehrere Sekunden). Die aktuellen Netz-Gewichte wandern pro Auftrag als
CPU-``state_dict`` zu den Workern.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import torch

from az.net import OthelloNet
from az.replay import ReplayBuffer, Sample
from az.selfplay_parallel import generate_games_parallel
from config import MCTSConfig, SelfPlayConfig


def _play_chunk(args) -> list[Sample]:
    """Ein Worker-Auftrag: ``n_games`` Partien mit den übergebenen Gewichten spielen.

    Muss auf Modulebene liegen (picklebar für den Spawn auf Windows). Die
    Argumente kommen als ein Tupel, damit ``ProcessPoolExecutor.map`` reicht.
    """
    (state_dict, board_size, net_config, n_games,
     mcts_config, selfplay_config, device_str, seed) = args
    torch.set_num_threads(1)  # ein Kern pro Worker – kein Thread-Gerangel
    device = torch.device(device_str)
    net = OthelloNet(board_size, net_config)
    net.load_state_dict(state_dict)
    net.to(device).eval()
    return generate_games_parallel(
        net, board_size, n_games,
        mcts_config=mcts_config, config=selfplay_config, device=device, seed=seed,
    )


class SelfPlayPool:
    """Persistenter Prozess-Pool, der Self-Play-Partien auf CPU-Kerne verteilt."""

    def __init__(self, n_workers: int) -> None:
        if n_workers < 2:
            raise ValueError("SelfPlayPool lohnt erst ab 2 Workern (sonst direkt "
                             "generate_games_parallel benutzen)")
        self.n_workers = n_workers
        self._executor = ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp.get_context("spawn")
        )

    def generate(
        self,
        net: OthelloNet,
        size: int,
        n_games: int,
        *,
        mcts_config: MCTSConfig,
        config: SelfPlayConfig,
        device: str | torch.device,
        seed: int = 0,
        buffer: ReplayBuffer | None = None,
    ) -> list[Sample]:
        """Spielt ``n_games`` Partien verteilt und gibt die (augmentierten) Samples zurück.

        Schnittstelle wie :func:`az.selfplay_parallel.generate_games_parallel`,
        damit die Pipeline zwischen beiden umschalten kann.
        """
        state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}

        base, rest = divmod(n_games, self.n_workers)
        chunk_sizes = [base + (1 if i < rest else 0) for i in range(self.n_workers)]
        chunk_sizes = [n for n in chunk_sizes if n > 0]
        # config.n_parallel gilt für die Iteration insgesamt – auf die Worker aufteilen.
        per_worker_parallel = max(1, -(-config.n_parallel // len(chunk_sizes)))
        worker_config = replace(config, n_parallel=per_worker_parallel)

        tasks = [
            (state_dict, net.board_size, net.config, n,
             mcts_config, worker_config, str(device), seed + i * 1_000_003)
            for i, n in enumerate(chunk_sizes)
        ]
        samples: list[Sample] = []
        for chunk in self._executor.map(_play_chunk, tasks):
            samples.extend(chunk)

        if buffer is not None:
            buffer.add(samples)
        return samples

    def close(self) -> None:
        self._executor.shutdown()
