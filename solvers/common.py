from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from env import ALPHA, BETA, DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


def base_reward(weight: float) -> float:
    if weight <= 0.2:
        return 4.0
    if weight <= 3.0:
        return 10.0
    if weight <= 10.0:
        return 15.0
    if weight <= 30.0:
        return 20.0
    return 30.0


def estimate_delivery_reward(order: Order, delivery_t: int, horizon: int) -> float:
    rb = base_reward(order.w)
    if delivery_t <= order.et:
        bonus = max(0.0, (order.et - delivery_t) / max(order.et, 1))
        return ALPHA[order.p] * rb * (1.0 + bonus)
    factor = max(0.0, 1.0 - (delivery_t - order.et) / max(horizon, 1))
    return BETA[order.p] * rb * factor


@dataclass(frozen=True)
class PolicyParams:
    name: str
    pickup_weight: float = 1.0
    delivery_weight: float = 1.0
    priority_weight: float = 7.0
    urgency_weight: float = 5.0
    distance_weight: float = 1.0
    lateness_weight: float = 2.0
    reserve_next_cells: bool = False
    allow_batch_pickup: bool = True
    stochastic_top_k: int = 1
    seed: int = 17


class OnlineHeuristicSolver(Solver):
    """Shared online planner for the Phase 1 solvers.

    The policy replans every timestep from the visible orders only.  It is
    intentionally dependency-free so it can run in the Kaggle grader without
    installing OR-Tools or other packages.
    """

    def __init__(self, env: DeliveryEnv, params: PolicyParams):
        super().__init__(env)
        self.params = params
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}
        self._rng = random.Random(params.seed)

    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _bfs_parents(
        self,
        start: Position,
        goal: Position,
    ) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None

        queue: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}

        while queue:
            cur = queue.popleft()
            if cur == goal:
                return parent
            for move, nxt in self._neighbors(cur):
                if nxt in parent:
                    continue
                parent[nxt] = (cur, move)
                queue.append(nxt)
        return None

    def distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        key = (start, goal)
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._distance_cache[key] = INF
            return INF

        dist = 0
        cur = goal
        while cur != start:
            prev, _ = parent[cur]
            if prev is None:
                self._distance_cache[key] = INF
                return INF
            cur = prev
            dist += 1

        self._distance_cache[key] = dist
        self._distance_cache[(goal, start)] = dist
        return dist

    def next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"
        key = (start, goal)
        cached = self._next_move_cache.get(key)
        if cached is not None:
            return cached

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._next_move_cache[key] = "S"
            return "S"

        cur = goal
        while True:
            prev, move = parent[cur]
            if prev is None:
                self._next_move_cache[key] = "S"
                return "S"
            if prev == start:
                self._next_move_cache[key] = move
                return move
            cur = prev

    def _carried_orders(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        return [orders[oid] for oid in shipper.bag if oid in orders and not orders[oid].delivered]

    def _can_pickup_at_cell(self, shipper: Shipper, orders: Dict[int, Order]) -> bool:
        return any(o for o in orders.values() if shipper.can_pickup(o, orders))

    def _best_delivery_here(self, shipper: Shipper, orders: Dict[int, Order]) -> bool:
        return any(o for o in self._carried_orders(shipper, orders) if (o.ex, o.ey) == shipper.position)

    def _delivery_score(self, shipper: Shipper, order: Order, t: int, horizon: int) -> float:
        dist = self.distance(shipper.position, (order.ex, order.ey))
        if dist >= INF:
            return -INF
        eta = t + dist
        reward = estimate_delivery_reward(order, eta, horizon)
        lateness = max(0, eta - order.et)
        slack = max(0, order.et - eta)
        urgency = self.params.urgency_weight / max(slack + 1, 1)
        return (
            self.params.delivery_weight * reward
            + self.params.priority_weight * order.p
            + urgency
            - self.params.distance_weight * dist
            - self.params.lateness_weight * lateness
        )

    def _pickup_score(self, shipper: Shipper, order: Order, orders: Dict[int, Order], t: int, horizon: int) -> float:
        if not shipper.can_carry(order, orders):
            return -INF
        to_pick = self.distance(shipper.position, (order.sx, order.sy))
        trip = self.distance((order.sx, order.sy), (order.ex, order.ey))
        if to_pick >= INF or trip >= INF:
            return -INF
        eta = t + to_pick + trip
        reward = estimate_delivery_reward(order, eta, horizon)
        lateness = max(0, eta - order.et)
        slack = max(0, order.et - eta)
        urgency = self.params.urgency_weight / max(slack + 1, 1)
        wait_penalty = max(0, t - order.appear_t) * 0.03
        return (
            self.params.pickup_weight * reward
            + self.params.priority_weight * order.p
            + urgency
            + wait_penalty
            - self.params.distance_weight * (to_pick + 0.65 * trip)
            - self.params.lateness_weight * lateness
        )

    def _choose_delivery(self, shipper: Shipper, orders: Dict[int, Order], t: int, horizon: int) -> Optional[Order]:
        candidates = self._carried_orders(shipper, orders)
        if not candidates:
            return None
        return max(candidates, key=lambda o: (self._delivery_score(shipper, o, t, horizon), -o.et, o.p))

    def _choose_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_orders: set[int],
        t: int,
        horizon: int,
    ) -> Optional[Order]:
        candidates = [
            o for o in orders.values()
            if o.id not in reserved_orders and not o.picked and not o.delivered
        ]
        scored = [
            (self._pickup_score(shipper, o, orders, t, horizon), o)
            for o in candidates
        ]
        scored = [(score, o) for score, o in scored if score > -INF]
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], item[1].p, -item[1].et), reverse=True)
        k = max(1, min(self.params.stochastic_top_k, len(scored)))
        if k == 1:
            return scored[0][1]
        weights = [max(0.01, scored[i][0] - scored[k - 1][0] + 1.0) for i in range(k)]
        return self._rng.choices([scored[i][1] for i in range(k)], weights=weights, k=1)[0]

    def _target_action(self, shipper: Shipper, goal: Position, op_at_goal: int) -> Action:
        move = self.next_move(shipper.position, goal)
        next_position = valid_next_pos(shipper.position, move, self.grid)
        return (move, op_at_goal) if next_position == goal else (move, 0)

    def _avoid_reserved_cell(
        self,
        shipper: Shipper,
        action: Action,
        reserved_cells: set[Position],
    ) -> Action:
        if not self.params.reserve_next_cells:
            return action
        move, op = action
        nxt = valid_next_pos(shipper.position, move, self.grid)
        if move == "S" or nxt not in reserved_cells:
            reserved_cells.add(nxt)
            return action
        reserved_cells.add(shipper.position)
        return ("S", 0)

    def decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs["t"])
        horizon = int(obs["T"])

        actions: Dict[int, Action] = {}
        reserved_orders: set[int] = set()
        reserved_cells: set[Position] = set()

        # Lower id shippers move first in the environment; planning in the same
        # order makes reservations match the simulator's collision priority.
        for shipper in sorted(shippers, key=lambda s: s.id):
            if self._best_delivery_here(shipper, orders):
                action = ("S", 2)
                actions[shipper.id] = self._avoid_reserved_cell(shipper, action, reserved_cells)
                continue

            carried = self._choose_delivery(shipper, orders, t, horizon)
            pickup = self._choose_pickup(shipper, orders, reserved_orders, t, horizon)

            chosen_action: Action = ("S", 0)
            chosen_pickup_id: Optional[int] = None

            if pickup is not None and self._can_pickup_at_cell(shipper, orders):
                # If already standing on useful cargo, pick it before moving on.
                chosen_action = ("S", 1)
                chosen_pickup_id = pickup.id if pickup.sx == shipper.r and pickup.sy == shipper.c else None
            elif carried is None and pickup is not None:
                chosen_action = self._target_action(shipper, (pickup.sx, pickup.sy), 1)
                chosen_pickup_id = pickup.id
            elif carried is not None and pickup is None:
                chosen_action = self._target_action(shipper, (carried.ex, carried.ey), 2)
            elif carried is not None and pickup is not None:
                deliver_score = self._delivery_score(shipper, carried, t, horizon)
                pickup_score = self._pickup_score(shipper, pickup, orders, t, horizon)
                delivery_dist = self.distance(shipper.position, (carried.ex, carried.ey))
                urgent_delivery = t + delivery_dist >= carried.et - max(2, delivery_dist // 2)

                if self.params.allow_batch_pickup and not urgent_delivery and pickup_score > deliver_score * 0.82:
                    chosen_action = self._target_action(shipper, (pickup.sx, pickup.sy), 1)
                    chosen_pickup_id = pickup.id
                else:
                    chosen_action = self._target_action(shipper, (carried.ex, carried.ey), 2)

            if chosen_pickup_id is not None:
                reserved_orders.add(chosen_pickup_id)
            actions[shipper.id] = self._avoid_reserved_cell(shipper, chosen_action, reserved_cells)

        return actions

    def run(self) -> dict:
        start = time.time()
        obs = self.env.reset()
        while not obs.get("done", False):
            actions = self.decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break
        return self.env.result(self.params.name, elapsed_sec=time.time() - start)
