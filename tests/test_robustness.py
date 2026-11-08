"""Hand-derived checks of the disruption engine on a 6-junction toy network.

   A -S1- B -S2- C -S6- F        S3: C->D (one-way)   S4: D-E (two-way)   S5: E->C (one-way)
   (C, D, E form a loop C->D->E->C)
Routes: R1 uses S1,S2,S3 (demand 100); R2 uses S4,S5 (50); R3 uses S6 (10). Total demand 160.
"""
import numpy as np
import pandas as pd
import pytest

from meteoric.robustness import (affected_routes, build_network, choose_segments, road_share, run_all_models,
                                 run_scenario)


@pytest.fixture
def net():
    segs = pd.DataFrame({
        "segment_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "from_node": ["A", "B", "C", "D", "E", "C"],
        "to_node":   ["B", "C", "D", "E", "C", "F"],
        "oneway":    [False, False, True, False, True, False],
    })
    rs = pd.DataFrame({"route": ["R1", "R1", "R1", "R2", "R2", "R3"],
                       "segment_id": ["S1", "S2", "S3", "S4", "S5", "S6"]})
    return build_network(segs, rs, pd.Series({"R1": 100.0, "R2": 50.0, "R3": 10.0}))


def idx(net, *ids):
    return np.array([list(net.segment_ids).index(i) for i in ids])


def test_intact_network_is_fully_connected_both_ways(net):
    nothing = np.array([], dtype=int)
    assert road_share(net, nothing, directed=False) == 1.0
    assert road_share(net, nothing, directed=True) == 1.0


def test_closing_a_one_way_link_splits_directed_but_not_undirected(net):
    r = idx(net, "S3")
    assert road_share(net, r, directed=False) == 1.0               # still one weak piece
    assert road_share(net, r, directed=True) == pytest.approx(4 / 6)   # {A,B,C,F} and {D,E}


def test_closing_a_dead_end_isolates_one_junction(net):
    r = idx(net, "S6")
    assert road_share(net, r, directed=False) == pytest.approx(5 / 6)
    assert road_share(net, r, directed=True) == pytest.approx(5 / 6)


def test_affected_routes(net):
    assert list(affected_routes(net, idx(net, "S3"))) == [True, False, False]
    assert list(affected_routes(net, idx(net, "S4", "S6"))) == [False, True, True]


def test_scenario_numbers_for_s3(net):
    m0 = run_scenario(net, "M0", "random", 0)
    assert m0["road_retained"] == 1.0 and m0["routes_affected_share"] == 0
    # force the removal of S3 via targeted n=1 on a network where S3 is uniquely top
    segs = pd.DataFrame({"segment_id": ["S3"], "from_node": ["C"], "to_node": ["D"], "oneway": [True]})
    one = build_network(segs, pd.DataFrame({"route": ["R"], "segment_id": ["S3"]}), pd.Series({"R": 5.0}))
    out = run_scenario(one, "M3", "targeted", 1)
    assert out["routes_affected_share"] == 1.0 and out["demand_affected_share"] == 1.0


def test_demand_share_and_model_differences(net):
    # remove S3 + S6 explicitly through the building blocks used by run_scenario
    r = idx(net, "S3", "S6")
    hit = affected_routes(net, r)
    assert net.route_demand[hit].sum() / net.route_demand.sum() == pytest.approx(110 / 160)
    assert road_share(net, r, directed=True) == pytest.approx(3 / 6)    # {A,B,C} | {D,E} | {F}
    assert road_share(net, r, directed=False) == pytest.approx(5 / 6)   # {A,B,C,D,E} joined via S5, S4; F alone


def test_targeted_uses_demand_for_m2_m3_and_route_count_for_m0_m1(net):
    # S1,S2,S3 carry demand 100 each; S4,S5 50; S6 10. Route counts are all 1 (ties -> first by order).
    for model in ("M2", "M3"):
        assert set(net.segment_ids[choose_segments(net, model, "targeted", 3)]) == {"S1", "S2", "S3"}
    assert set(net.segment_ids[choose_segments(net, "M0", "targeted", 2)]) == {"S1", "S2"}


def test_random_is_repeatable_and_identical_across_models(net):
    a = choose_segments(net, "M0", "random", 3, seed=7)
    b = choose_segments(net, "M3", "random", 3, seed=7)
    c = choose_segments(net, "M0", "random", 3, seed=8)
    assert list(a) == list(b) and len(set(a)) == 3
    assert len(c) == 3


def test_bad_inputs_fail_loudly(net):
    with pytest.raises(ValueError):
        choose_segments(net, "M0", "random", 99)
    with pytest.raises(ValueError):
        choose_segments(net, "M9", "random", 1)
    segs = pd.DataFrame({"segment_id": ["S1"], "from_node": ["A"], "to_node": ["B"], "oneway": [False]})
    with pytest.raises(ValueError):
        build_network(segs, pd.DataFrame({"route": ["R"], "segment_id": ["NOPE"]}))


def test_candidate_pool_restricts_both_strategies(net):
    allowed = np.array([False, False, False, True, True, True])        # only S4, S5, S6 may fail
    for seed in range(5):
        assert set(net.segment_ids[choose_segments(net, "M0", "random", 2, seed, allowed)]) <= {"S4", "S5", "S6"}
    # S1-S3 carry the most demand, but are not allowed, so targeted picks the best allowed ones
    assert set(net.segment_ids[choose_segments(net, "M2", "targeted", 2, 0, allowed)]) == {"S4", "S5"}
    with pytest.raises(ValueError):
        choose_segments(net, "M0", "random", 4, 0, allowed)             # only 3 candidates


def test_run_all_models_matches_single_runs(net):
    for strategy, n in (("random", 2), ("targeted", 2)):
        together = {r["model"]: r for r in run_all_models(net, strategy, n, seed=3)}
        for model in ("M0", "M1", "M2", "M3"):
            single = run_scenario(net, model, strategy, n, seed=3)
            for key in ("road_share_after", "road_retained", "routes_affected", "demand_affected_share"):
                assert together[model][key] == pytest.approx(single[key])