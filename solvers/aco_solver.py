from __future__ import annotations

from env import DeliveryEnv
from solvers.common import OnlineHeuristicSolver, PolicyParams


class ACOSolver(OnlineHeuristicSolver):
    """Ant-colony-inspired online solver.

    A full offline ACO is a poor fit here because orders are revealed over
    time.  This implementation uses the same desirability terms as ACO:
    reward/deadline attractiveness versus route length, with a small
    deterministic top-k sampling step so agents do not always collapse onto
    the same locally best order.
    """

    def __init__(self, env: DeliveryEnv):
        super().__init__(
            env,
            PolicyParams(
                name="ACO",
                pickup_weight=1.345,
                delivery_weight=0.888,
                priority_weight=10.44,
                urgency_weight=9.82,
                distance_weight=0.943,
                lateness_weight=1.92,
                reserve_next_cells=False,
                allow_batch_pickup=True,
            ),
        )
