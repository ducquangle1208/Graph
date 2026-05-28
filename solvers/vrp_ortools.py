from __future__ import annotations

from env import DeliveryEnv
from solvers.common import OnlineHeuristicSolver, PolicyParams


class VRPOrToolsSolver(OnlineHeuristicSolver):
    """Dynamic VRP-style solver.

    The official environment is online and the Kaggle runner may not provide
    OR-Tools.  This class therefore uses a rolling-horizon VRP heuristic:
    visible orders are rescored every timestep by pickup-to-delivery value,
    capacity feasibility, deadline slack, and travel distance.
    """

    def __init__(self, env: DeliveryEnv):
        super().__init__(
            env,
            PolicyParams(
                name="VRP-OrTools",
                pickup_weight=1.43,
                delivery_weight=0.93,
                priority_weight=3.77,
                urgency_weight=4.76,
                distance_weight=0.63,
                lateness_weight=2.56,
                reserve_next_cells=False,
                allow_batch_pickup=True,
            ),
        )
