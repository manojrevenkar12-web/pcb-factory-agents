from __future__ import annotations

from pfa.schedule.replan import STAGES, DispatchRule, Replanner, State, forecast, make_day, simulate


def test_every_order_is_built_under_both_policies():
    day = make_day(1, 25)
    for policy in (DispatchRule(), Replanner(time_limit_s=0.3)):
        result = simulate(day, policy)
        assert 0 <= result.on_time <= 1 and result.mean_flow_min > 0


def test_the_forecast_is_a_complete_feasible_schedule():
    day = make_day(2, 20)
    schedule = forecast(State(now=0), day.orders, DispatchRule())
    for o in day.orders:
        ends = [schedule[(o.id, s)] for s in STAGES]
        assert all(a[1] <= b[0] for a, b in zip(ends, ends[1:], strict=False))  # stages in order
    by_resource: dict[str, list] = {}
    for start, end, res in schedule.values():
        by_resource.setdefault(res, []).append((start, end))
    for spans in by_resource.values():  # no resource does two things at once
        spans.sort()
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))


def test_the_replanner_never_adopts_a_plan_worse_than_the_rule():
    day = make_day(3, 30)
    planner = Replanner(time_limit_s=0.3)
    simulate(day, planner)
    assert planner.improved and min(planner.improved) >= 0
    assert max(planner.solve_times) < 2.0
