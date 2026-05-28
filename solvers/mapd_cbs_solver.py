from __future__ import annotations

from env import DeliveryEnv
from solvers.common import OnlineHeuristicSolver, PolicyParams


class MAPDCBSSolver(OnlineHeuristicSolver):
    """MAPD solver with prioritized conflict handling.

    The environment resolves collisions by shipper id.  This solver mirrors
    that rule during planning: lower-id agents reserve their next cell first,
    and later agents wait when their first BFS step would conflict.  It is a
    lightweight online CBS approximation suitable for the small Phase 1 maps.
    """

    def __init__(self, env: DeliveryEnv):
        super().__init__(
            env,
            PolicyParams(
                name="MAPD-CBS",
                pickup_weight=1.36,
                delivery_weight=0.87,
                priority_weight=5.0,
                urgency_weight=6.65,
                distance_weight=1.16,
                lateness_weight=2.65,
                reserve_next_cells=True,
                allow_batch_pickup=True,
            ),
        )
