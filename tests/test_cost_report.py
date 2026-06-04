from spaniq.reporting.cost_report import estimate_llm_judge_cost


def test_cost_scales_with_cases():
    cost_100 = estimate_llm_judge_cost(100, 1)
    cost_200 = estimate_llm_judge_cost(200, 1)
    assert cost_200 > cost_100


def test_cost_scales_with_metrics():
    cost_1 = estimate_llm_judge_cost(100, 1)
    cost_3 = estimate_llm_judge_cost(100, 3)
    assert cost_3 > cost_1


def test_zero_cases_zero_cost():
    assert estimate_llm_judge_cost(0, 3) == 0.0


def test_cost_is_positive_for_nonzero_inputs():
    assert estimate_llm_judge_cost(10, 2) > 0
