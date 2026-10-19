import pytest
from pydantic import ValidationError
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def _setup_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")
    return session, mine_patterns_tool


def test_mine_patterns_known_answer_17_fpgrowth():
    """Known Answer #17 test for FPGrowth filtering and mining.

    - variance & term_frequency keep zero terms for both catA and catB across all algorithms.
    - tfidf keeps alpha for catA and gamma for catB.
    - catA -> {alpha} (support 3 at min_sup=1..3, 0 patterns at min_sup=4)
    - catB -> {gamma} (support 4 at min_sup=1..4)
    """
    session, mine_patterns_tool = _setup_session()

    # 1. Check variance and term_frequency filtering for both catA and catB (0 terms survived)
    for cat in ["catA", "catB"]:
        for fm in ["variance", "term_frequency"]:
            for alg in ["fpgrowth", "topk", "maxfpgrowth"]:
                kwargs = {"category_name": cat, "filtering_method": fm, "algorithm": alg}
                if alg == "topk":
                    kwargs["k"] = 1
                else:
                    kwargs["min_sup"] = 1

                res = mine_patterns_tool.invoke(kwargs)
                assert res["kept_terms_count"] == 0
                assert res["patterns"] == []
                assert res.get("note") == "no terms survived filtering"

                # Verify artifact storage for empty zero-term results
                art_key = f"patterns_{cat}_{fm}_{alg}"
                assert art_key in session.artifacts
                assert session.artifacts[art_key] == []

    # 2. Check tfidf filtering for catA with min_sup=1, 2, 3 (expect alpha support 3)
    for ms in [1, 2, 3]:
        res_a = mine_patterns_tool.invoke(
            {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": ms}
        )
        assert res_a["kept_terms_count"] == 1
        assert len(res_a["patterns"]) == 1
        assert res_a["patterns"][0]["pattern"] == ["alpha"]
        assert res_a["patterns"][0]["support"] == 3

    # catA with min_sup=4 -> 0 patterns found
    res_a_ms4 = mine_patterns_tool.invoke(
        {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 4}
    )
    assert res_a_ms4["kept_terms_count"] == 1
    assert res_a_ms4["patterns"] == []
    assert res_a_ms4.get("note") == "zero patterns found"

    # 3. Check tfidf filtering for catB with min_sup=1..4 (expect gamma support 4)
    for ms in [1, 2, 3, 4]:
        res_b = mine_patterns_tool.invoke(
            {"category_name": "catB", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": ms}
        )
        assert res_b["kept_terms_count"] == 1
        assert len(res_b["patterns"]) == 1
        assert res_b["patterns"][0]["pattern"] == ["gamma"]
        assert res_b["patterns"][0]["support"] == 4

    # Check artifact key in session
    art_key = "patterns_catB_tfidf_fpgrowth"
    assert art_key in session.artifacts
    assert len(session.artifacts[art_key]) == 1
    assert session.artifacts[art_key][0]["pattern"] == ["gamma"]


def test_mine_patterns_known_answer_20_topk_and_maxfpgrowth():
    """Known Answer #20 test for TopK (FAE) and MaxFPGrowth algorithms.

    - TopK (FAE) at k=1, 2, 3 returns 1 pattern for both categories.
    - MaxFPGrowth returns 0 patterns for both categories at min_sup=1 and 4.
    """
    session, mine_patterns_tool = _setup_session()

    # 1. TopK algorithm test
    for k_val in [1, 2, 3]:
        # catA -> {alpha} support 3
        res_a_topk = mine_patterns_tool.invoke(
            {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": k_val}
        )
        assert res_a_topk["kept_terms_count"] == 1
        assert len(res_a_topk["patterns"]) == 1
        assert res_a_topk["patterns"][0]["pattern"] == ["alpha"]
        assert res_a_topk["patterns"][0]["support"] == 3

        # catB -> {gamma} support 4
        res_b_topk = mine_patterns_tool.invoke(
            {"category_name": "catB", "filtering_method": "tfidf", "algorithm": "topk", "k": k_val}
        )
        assert res_b_topk["kept_terms_count"] == 1
        assert len(res_b_topk["patterns"]) == 1
        assert res_b_topk["patterns"][0]["pattern"] == ["gamma"]
        assert res_b_topk["patterns"][0]["support"] == 4

    # Check artifact key for topk
    assert "patterns_catA_tfidf_topk" in session.artifacts

    # 2. MaxFPGrowth algorithm test
    for ms in [1, 4]:
        res_a_max = mine_patterns_tool.invoke(
            {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "maxfpgrowth", "min_sup": ms}
        )
        assert res_a_max["kept_terms_count"] == 1
        assert res_a_max["patterns"] == []
        assert res_a_max.get("note") == "zero patterns found"

        res_b_max = mine_patterns_tool.invoke(
            {"category_name": "catB", "filtering_method": "tfidf", "algorithm": "maxfpgrowth", "min_sup": ms}
        )
        assert res_b_max["kept_terms_count"] == 1
        assert res_b_max["patterns"] == []
        assert res_b_max.get("note") == "zero patterns found"


def test_mine_patterns_tool_argument_validation():
    """Test validation of arguments and error handling in mine_patterns_tool."""
    session, mine_patterns_tool = _setup_session()

    # Invalid category
    res_inv_cat = mine_patterns_tool.invoke(
        {"category_name": "catX", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1}
    )
    assert "error" in res_inv_cat

    # Invalid filtering method
    res_inv_fm = mine_patterns_tool.invoke(
        {"category_name": "catA", "filtering_method": "invalid_method", "algorithm": "fpgrowth", "min_sup": 1}
    )
    assert "error" in res_inv_fm

    # Invalid algorithm
    res_inv_alg = mine_patterns_tool.invoke(
        {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "invalid_alg", "min_sup": 1}
    )
    assert "error" in res_inv_alg

    # Missing min_sup for fpgrowth
    res_miss_sup = mine_patterns_tool.invoke(
        {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth"}
    )
    assert "error" in res_miss_sup

    # Missing k for topk
    res_miss_k = mine_patterns_tool.invoke(
        {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk"}
    )
    assert "error" in res_miss_k

    # Invalid boolean values (min_sup=True, k=True) raise Pydantic ValidationError due to StrictInt
    with pytest.raises(ValidationError):
        mine_patterns_tool.invoke(
            {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": True}
        )

    with pytest.raises(ValidationError):
        mine_patterns_tool.invoke(
            {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": True}
        )

    # Non-positive integers (min_sup=0, k=0) return error dicts
    res_zero_sup = mine_patterns_tool.invoke(
        {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 0}
    )
    assert "error" in res_zero_sup

    res_zero_k = mine_patterns_tool.invoke(
        {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": 0}
    )
    assert "error" in res_zero_k
