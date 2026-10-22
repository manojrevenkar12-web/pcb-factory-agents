"""Replanning the factory: a constraint solver against a dispatch rule, through a day of disruptions.

Orders arrive through the day. Each runs paste printing -> placement (on any of three robots) ->
reflow -> inspection (two operators). A robot loaded with one design's feeders needs a 20-minute
changeover to place another design, so the order of work on each robot matters. Robots break down
without warning and a broken robot's current job starts over. Rush orders arrive with short
deadlines.

- **Dispatch rule**: when a resource is free it starts the waiting job with the highest
  ATCS priority (apparent tardiness cost with setups, the textbook heuristic for this problem).
- **CP-SAT replanning**: whenever something changes (an order arrives, a job finishes, a robot
  breaks or is repaired) the whole remaining plan is re-solved with OR-Tools CP-SAT, under a
  time limit, warm-started from the previous plan. Work already running is fixed. If the solver
  finds nothing in time, the previous plan stands.

Both see exactly the same day. The scores are weighted tardiness (rush orders weigh three times),
the share of orders on time, and how long each replan took.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

STAGES = ("print", "place", "reflow", "inspect")
ROBOTS = ("robot1", "robot2", "robot3")
DAY = 16 * 60  # minutes: two shifts


SETUP = 20  # minutes to change a robot's feeders to another design
DESIGNS = 6


@dataclass
class Order:
    id: int
    arrival: int
    due: int
    weight: int
    panels: int
    parts: int
    design: int = 0

    def duration(self, stage: str) -> int:
        if stage == "print":
            return 3 + self.panels
        if stage == "place":
            return 4 + math.ceil(self.panels * self.parts / 140)
        if stage == "reflow":
            return 6 + 2 * self.panels
        return 2 + 3 * self.panels  # inspection


@dataclass
class Breakdown:
    robot: str
    start: int
    end: int


@dataclass
class Day:
    orders: list[Order]
    breakdowns: list[Breakdown]


def make_day(seed: int, n_orders: int = 40) -> Day:
    rng = random.Random(seed)
    orders = []
    for i in range(n_orders):
        arrival = int(rng.uniform(0, DAY * 0.7))
        rush = rng.random() < 0.15
        panels = rng.randint(1, 8)
        parts = rng.randint(40, 400)
        work = 3 + panels + 4 + panels * parts / 140 + SETUP + 6 + 2 * panels + 2 + 3 * panels
        slack = rng.uniform(1.3, 2.2) if rush else rng.uniform(2.5, 6.0)
        orders.append(
            Order(
                i,
                arrival,
                int(arrival + work * slack + 30),
                3 if rush else 1,
                panels,
                parts,
                rng.randrange(DESIGNS),
            )
        )
    breakdowns = []
    for robot in ROBOTS:
        if rng.random() < 0.7:
            start = int(rng.uniform(60, DAY * 0.8))
            breakdowns.append(Breakdown(robot, start, start + int(rng.uniform(30, 120))))
    return Day(sorted(orders, key=lambda o: o.arrival), breakdowns)


# ---- the simulated factory ----------------------------------------------------------------------------------


@dataclass
class Running:
    order: int
    stage: str
    resource: str
    start: int
    end: int


@dataclass
class State:
    now: int = 0
    stage_done: dict[int, int] = field(default_factory=dict)  # order -> stages completed
    ready_at: dict[int, int] = field(default_factory=dict)  # order -> time its next stage may start
    finished: dict[int, int] = field(default_factory=dict)  # order -> completion time
    running: list[Running] = field(default_factory=list)
    broken: dict[str, int] = field(default_factory=dict)  # robot -> repaired at
    loaded: dict[str, int] = field(default_factory=dict)  # robot -> design its feeders hold
    setups: int = 0

    def busy(self, resource: str) -> bool:
        return any(r.resource == resource for r in self.running)

    def next_stage(self, oid: int) -> str | None:
        k = self.stage_done.get(oid, 0)
        return STAGES[k] if k < len(STAGES) else None


def setup_time(state: State, robot: str, o: Order) -> int:
    return 0 if state.loaded.get(robot) == o.design else SETUP


def resources_for(stage: str) -> list[str]:
    return {"print": ["printer"], "place": list(ROBOTS), "reflow": ["oven"], "inspect": ["op1", "op2"]}[stage]


# ---- policies ---------------------------------------------------------------------------------------------------


class DispatchRule:
    """Apparent tardiness cost (Vepsalainen & Morton): urgency weighted against work remaining."""

    name = "dispatch rule (ATCS)"

    def __init__(self, k: float = 2.0):
        self.k = k

    def plan(self, state: State, day: Day, arrived: list[Order]) -> None:
        pass

    def choose(self, state: State, resource: str, candidates: list[Order], stage: str) -> Order | None:
        if not candidates:
            return None
        mean_p = statistics.fmean(o.duration(stage) for o in candidates) or 1

        def atcs(o: Order) -> float:
            remaining = sum(o.duration(s) for s in STAGES[STAGES.index(stage) :])
            slack = max(0.0, o.due - remaining - state.now)
            setup = setup_time(state, resource, o) if stage == "place" else 0
            return (
                o.weight
                / (o.duration(stage) + setup)
                * math.exp(-slack / (self.k * mean_p))
                * math.exp(-setup / (0.5 * SETUP))
            )

        return max(candidates, key=atcs)


def forecast(
    state: State, orders: list[Order], rule: DispatchRule
) -> dict[tuple[int, str], tuple[int, int, str]]:
    """Runs the dispatch rule forward from now on the orders known so far: a complete, feasible
    schedule that the solver starts from, so its plan is never worse than the rule's."""
    st = State(
        state.now,
        dict(state.stage_done),
        dict(state.ready_at),
        dict(state.finished),
        [Running(r.order, r.stage, r.resource, r.start, r.end) for r in state.running],
        dict(state.broken),
        dict(state.loaded),
    )
    by_id = {o.id: o for o in orders}
    out = {(r.order, r.stage): (r.start, r.end, r.resource) for r in st.running}
    while any(o.id not in st.finished for o in orders):
        for r in [r for r in st.running if r.end <= st.now]:
            st.running.remove(r)
            st.stage_done[r.order] = st.stage_done.get(r.order, 0) + 1
            st.ready_at[r.order] = r.end
            if st.stage_done[r.order] == len(STAGES):
                st.finished[r.order] = r.end
        for robot, until in list(st.broken.items()):
            if until <= st.now:
                del st.broken[robot]
        for stage in STAGES:
            for res in resources_for(stage):
                if st.busy(res) or res in st.broken:
                    continue
                cands = [
                    by_id[oid]
                    for oid in by_id
                    if oid not in st.finished
                    and st.next_stage(oid) == stage
                    and st.ready_at.get(oid, by_id[oid].arrival) <= st.now
                    and not any(r.order == oid for r in st.running)
                ]
                pick = rule.choose(st, res, cands, stage)
                if pick is not None:
                    d = pick.duration(stage)
                    if stage == "place":
                        d += setup_time(st, res, pick)
                        st.loaded[res] = pick.design
                    st.running.append(Running(pick.id, stage, res, st.now, st.now + d))
                    out[(pick.id, stage)] = (st.now, st.now + d, res)
        later = [r.end for r in st.running] + list(st.broken.values())
        later = [t for t in later if t > st.now]
        st.now = min(later) if later else st.now + 1
    return out


class Replanner:
    """Re-solves the remaining schedule with CP-SAT whenever the factory changes.

    Each robot's work is a circuit through the placement jobs it is given, so the solver sees the
    20-minute changeover between designs and can group jobs of the same design.
    """

    name = "CP-SAT replanning"

    def __init__(self, time_limit_s: float = 1.0, workers: int = 4):
        self.time_limit_s, self.workers = time_limit_s, workers
        self.queue: dict[str, list[tuple[int, str]]] = {}  # resource -> planned (order, stage), in order
        self.solve_times: list[float] = []
        self.statuses: list[str] = []
        self.improved: list[float] = []  # objective gain over the dispatch rule's schedule, per replan
        self.rule = DispatchRule()

    def plan(self, state: State, day: Day, arrived: list[Order]) -> None:
        open_orders = [o for o in arrived if o.id not in state.finished]
        if not open_orders:
            self.queue = {}
            return
        start_from = forecast(state, open_orders, self.rule)
        m = cp_model.CpModel()
        horizon = state.now + sum(o.duration(s) + SETUP for o in open_orders for s in STAGES) + 600
        by_res: dict[str, list] = {r: [] for r in ["printer", "oven", "op1", "op2", *ROBOTS]}
        starts, ends, presences = {}, {}, {}
        place_ops: list[tuple[Order, tuple[int, str]]] = []
        tard_terms, end_terms = [], []
        running = {(r.order, r.stage): r for r in state.running}
        for o in open_orders:
            prev_end = None
            first = state.stage_done.get(o.id, 0)
            for stage in STAGES[first:]:
                key = (o.id, stage)
                d = o.duration(stage)
                if key in running:  # already started: fixed in time and place
                    r = running[key]
                    by_res[r.resource].append(m.NewIntervalVar(r.start, r.end - r.start, r.end, f"run_{key}"))
                    starts[key], end = m.NewConstant(r.start), m.NewConstant(r.end)
                else:
                    lo = max(
                        state.now,
                        state.ready_at.get(o.id, o.arrival) if stage == STAGES[first] else state.now,
                    )
                    s = m.NewIntVar(lo, horizon, f"s_{key}")
                    end = m.NewIntVar(lo + d, horizon + d, f"e_{key}")
                    m.Add(end == s + d)
                    starts[key] = s
                    options = resources_for(stage)
                    lits = []
                    f_start, f_end, f_res = start_from[key]
                    for res in options:
                        lit = m.NewBoolVar(f"on_{key}_{res}") if len(options) > 1 else m.NewConstant(1)
                        by_res[res].append(m.NewOptionalIntervalVar(s, d, end, lit, f"iv_{key}_{res}"))
                        lits.append((res, lit))
                        if len(options) > 1:
                            m.AddHint(lit, res == f_res)
                    if len(options) > 1:
                        m.AddExactlyOne(lit for _, lit in lits)
                    presences[key] = dict(lits)
                    m.AddHint(s, f_end - d)  # after any changeover the rule paid
                    m.AddHint(end, f_end)
                    if stage == "place":
                        place_ops.append((o, key))
                if prev_end is not None:
                    m.Add(starts[key] >= prev_end)
                prev_end = end
                ends[key] = end
            tard = m.NewIntVar(0, horizon, f"tard_{o.id}")
            m.Add(tard >= prev_end - o.due)
            m.AddHint(tard, max(0, start_from[(o.id, STAGES[-1])][1] - o.due))
            tard_terms.append(o.weight * tard)
            end_terms.append(prev_end)
        for res in ("printer", "oven", "op1", "op2"):
            if len(by_res[res]) > 1:
                m.AddNoOverlap(by_res[res])
        rule_cost = 0
        for o in open_orders:
            done_at = start_from[(o.id, STAGES[-1])][1]
            rule_cost += 1000 * o.weight * max(0, done_at - o.due) + done_at
        for robot in ROBOTS:  # a circuit per robot: order of jobs, with changeovers between designs
            seq = [
                k for _, k in sorted((start_from[k][0], k) for _, k in place_ops if start_from[k][2] == robot)
            ]
            nxt = {a: b for a, b in zip(seq, seq[1:], strict=False)}
            busy_until = max(
                [state.now, state.broken.get(robot, 0)]
                + [r.end for r in state.running if r.resource == robot]
            )
            arcs = []
            for i, (oi, ki) in enumerate(place_ops, start=1):
                on = presences[ki][robot]
                arcs.append((i, i, on.Not()))
                first_lit = m.NewBoolVar(f"first_{robot}_{i}")
                m.AddHint(first_lit, bool(seq) and seq[0] == ki)
                arcs.append((0, i, first_lit))
                setup0 = 0 if state.loaded.get(robot) == oi.design else SETUP
                m.Add(starts[ki] >= busy_until + setup0).OnlyEnforceIf(first_lit)
                last_lit = m.NewBoolVar(f"last_{robot}_{i}")
                m.AddHint(last_lit, bool(seq) and seq[-1] == ki)
                arcs.append((i, 0, last_lit))
                for j, (oj, kj) in enumerate(place_ops, start=1):
                    if i != j:
                        lit = m.NewBoolVar(f"arc_{robot}_{i}_{j}")
                        m.AddHint(lit, nxt.get(ki) == kj)
                        arcs.append((i, j, lit))
                        gap = 0 if oi.design == oj.design else SETUP
                        m.Add(starts[kj] >= ends[ki] + gap).OnlyEnforceIf(lit)
            idle = m.NewBoolVar(f"idle_{robot}")
            m.AddHint(idle, not seq)
            arcs.append((0, 0, idle))
            m.AddCircuit(arcs)
        m.Minimize(1000 * sum(tard_terms) + sum(end_terms))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        solver.parameters.num_workers = self.workers
        solver.parameters.random_seed = 0
        started = time.perf_counter()
        status = solver.Solve(m)
        self.solve_times.append(time.perf_counter() - started)
        self.statuses.append(solver.StatusName(status))
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE) or solver.ObjectiveValue() > rule_cost:
            # nothing better in time: follow the dispatch rule's own schedule
            plan = {r: [] for r in by_res}
            for key in presences:
                f_start, _, f_res = start_from[key]
                plan[f_res].append((f_start, key[0], key[1]))
            self.queue = {r: [(oid, st) for _, oid, st in sorted(v)] for r, v in plan.items()}
            self.improved.append(0.0)
            return
        self.improved.append((rule_cost - solver.ObjectiveValue()) / max(1.0, rule_cost))
        plan: dict[str, list[tuple[int, int, str]]] = {r: [] for r in by_res}
        for key, lits in presences.items():
            s = solver.Value(starts[key])
            res = next(r for r, lit in lits.items() if solver.Value(lit))
            plan[res].append((s, key[0], key[1]))
        self.queue = {r: [(oid, st) for _, oid, st in sorted(v)] for r, v in plan.items()}

    def choose(self, state: State, resource: str, candidates: list[Order], stage: str) -> Order | None:
        """Starts the earliest job in this resource's planned sequence that is ready now. A planned
        job that is not ready yet does not hold the resource idle (non-delay execution)."""
        planned = self.queue.get(resource, [])
        ready = {o.id: o for o in candidates}
        for i, (oid, st) in enumerate(planned):
            if st == stage and oid in ready:
                planned.pop(i)
                return ready[oid]
        return None


# ---- simulation -------------------------------------------------------------------------------------------------


@dataclass
class Result:
    policy: str
    weighted_tardiness: int
    on_time: float
    rush_on_time: float
    mean_flow_min: float
    changeovers: int = 0
    replans: int = 0
    solve_p95_s: float = 0.0
    solve_max_s: float = 0.0
    plan_kept: int = 0


def simulate(day: Day, policy) -> Result:
    state = State()
    orders = {o.id: o for o in day.orders}
    arrived: list[Order] = []
    pending_arrivals = list(day.orders)
    downs = sorted(day.breakdowns, key=lambda b: b.start)
    changed = True
    while len(state.finished) < len(orders):
        # events at the current time
        while pending_arrivals and pending_arrivals[0].arrival <= state.now:
            o = pending_arrivals.pop(0)
            arrived.append(o)
            state.ready_at[o.id] = o.arrival
            changed = True
        for r in [r for r in state.running if r.end <= state.now]:
            state.running.remove(r)
            state.stage_done[r.order] = state.stage_done.get(r.order, 0) + 1
            state.ready_at[r.order] = r.end
            if state.stage_done[r.order] == len(STAGES):
                state.finished[r.order] = r.end
        for b in [b for b in downs if b.start <= state.now]:
            downs.remove(b)
            state.broken[b.robot] = b.end
            for r in [r for r in state.running if r.resource == b.robot]:
                state.running.remove(r)  # the job on the robot starts over
            changed = True
        for robot, until in list(state.broken.items()):
            if until <= state.now:
                del state.broken[robot]
                changed = True
        if changed:
            policy.plan(state, day, arrived)
            changed = False
        # start work on every free resource
        for stage in STAGES:
            for res in resources_for(stage):
                if state.busy(res) or res in state.broken:
                    continue
                cands = [
                    orders[oid]
                    for oid in state.ready_at
                    if oid not in state.finished
                    and state.next_stage(oid) == stage
                    and state.ready_at[oid] <= state.now
                    and not any(r.order == oid for r in state.running)
                ]
                pick = policy.choose(state, res, cands, stage)
                if pick is not None:
                    d = pick.duration(stage)
                    if stage == "place":
                        setup = setup_time(state, res, pick)
                        state.setups += setup > 0
                        state.loaded[res] = pick.design
                        d += setup
                    state.running.append(Running(pick.id, stage, res, state.now, state.now + d))
        # advance to the next event
        nxt = (
            [r.end for r in state.running]
            + [p.arrival for p in pending_arrivals[:1]]
            + [b.start for b in downs[:1]]
            + list(state.broken.values())
        )
        later = [t for t in nxt if t > state.now]
        if not later:
            state.now += 1  # waiting on a planned start
            changed = True
            continue
        state.now = min(later) if not isinstance(policy, Replanner) else min(min(later), state.now + 2)
    tard = {o.id: max(0, state.finished[o.id] - o.due) for o in day.orders}
    rush = [o for o in day.orders if o.weight > 1]
    res = Result(
        policy.name,
        sum(o.weight * tard[o.id] for o in day.orders),
        sum(1 for o in day.orders if tard[o.id] == 0) / len(day.orders),
        sum(1 for o in rush if tard[o.id] == 0) / len(rush) if rush else 1.0,
        statistics.fmean(state.finished[o.id] - o.arrival for o in day.orders),
        state.setups,
    )
    if isinstance(policy, Replanner):
        times = sorted(policy.solve_times)
        res.replans = len(times)
        res.solve_p95_s = times[int(0.95 * (len(times) - 1))]
        res.solve_max_s = times[-1]
        res.plan_kept = sum(s not in ("OPTIMAL", "FEASIBLE") for s in policy.statuses)
    return res


def run(days: int = 3, orders: int = 70, time_limit_s: float = 1.0) -> dict:
    """The same simulated days under both policies."""
    rows = []
    for seed in range(1, days + 1):
        day = make_day(seed, orders)
        rule = simulate(day, DispatchRule())
        planner = Replanner(time_limit_s=time_limit_s)
        solved = simulate(day, planner)
        rows.append(
            {
                "day": seed,
                "rule": vars(rule),
                "cp_sat": vars(solved),
                "mean_plan_gain_over_rule": statistics.fmean(planner.improved) if planner.improved else 0.0,
                "replans_improving_on_rule": sum(g > 0 for g in planner.improved),
            }
        )

    def total(policy: str, field_: str) -> float:
        return sum(r[policy][field_] for r in rows)

    return {
        "days": days,
        "orders_per_day": orders,
        "time_limit_s": time_limit_s,
        "per_day": rows,
        "weighted_tardiness": {
            "rule": total("rule", "weighted_tardiness"),
            "cp_sat": total("cp_sat", "weighted_tardiness"),
        },
        "changeovers": {"rule": total("rule", "changeovers"), "cp_sat": total("cp_sat", "changeovers")},
        "on_time": {
            "rule": statistics.fmean(r["rule"]["on_time"] for r in rows),
            "cp_sat": statistics.fmean(r["cp_sat"]["on_time"] for r in rows),
        },
        "replans": sum(r["cp_sat"]["replans"] for r in rows),
        "solve_p95_s": max(r["cp_sat"]["solve_p95_s"] for r in rows),
        "mean_plan_gain_over_rule": statistics.fmean(r["mean_plan_gain_over_rule"] for r in rows),
    }
