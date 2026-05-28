from __future__ import annotations

from env import DeliveryEnv
from solvers.common import OnlineHeuristicSolver, PolicyParams


class GreedyBFS(OnlineHeuristicSolver):
    """Greedy online solver using BFS shortest paths on the grid."""

    def __init__(self, env: DeliveryEnv):
        super().__init__(
            env,
            PolicyParams(
                name="GreedyBFS",
                pickup_weight=1.0,
                delivery_weight=1.0,
                priority_weight=7.0,
                urgency_weight=4.0,
                distance_weight=1.0,
                lateness_weight=2.0,
                reserve_next_cells=False,
                allow_batch_pickup=True,
            ),
        )
