import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
    # Instantiate session state exactly once
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM matrix first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Record feature_matrix shape and feature_names before filtering to verify immutability
    original_matrix_shape = session.feature_matrix.shape
    original_feature_names = list(session.feature_names)

    # Instantiate filtering tools
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Run variance_filter_tool with threshold=0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Known Answer #5 assertions from TEST_FIXTURE.md:
    variances = result["variances"]
    assert variances["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert variances["gamma"] == pytest.approx(1.109375, abs=1e-3)
    assert variances["delta"] == pytest.approx(0.25, abs=1e-3)
    assert variances["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert variances["always"] == pytest.approx(0.109375, abs=1e-3)

    assert result["retained_features_count"] == 4
    assert set(result["retained_features"]) == {"alpha", "beta", "delta", "gamma"}
    assert result["removed_features_count"] == 1
    assert result["removed_features"] == ["always"]

    assert session.feature_matrix.shape == original_matrix_shape
    assert session.feature_names == original_feature_names

    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]
    assert full_report.columns[0] == "term"
    assert full_report.columns[1] == "variance"
    assert full_report["variance"].is_monotonic_decreasing
    assert full_report.iloc[0]["term"] == "alpha"


def test_pearson_filter_tool():
    # Instantiate session state exactly once
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM matrix first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate filtering tools and verify tools are returned by make_tools
    filtering_tools = make_filtering_tools(session)
    tool_names = [t.name for t in filtering_tools]
    assert "variance_filter_tool" in tool_names
    assert "pearson_filter_tool" in tool_names

    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Run pearson_filter_tool with target_class="catB" and threshold=0.0
    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    # Known Answer #6 assertions from TEST_FIXTURE.md:
    pearson_scores = result["pearson_r"]
    assert pearson_scores["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert pearson_scores["always"] == pytest.approx(0.3780, abs=1e-3)
    assert pearson_scores["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert pearson_scores["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert pearson_scores["gamma"] == pytest.approx(0.8307, abs=1e-3)

    assert result["retained_features_count"] == 3
    assert set(result["retained_features"]) == {"delta", "gamma", "always"}
    assert result["removed_features_count"] == 2
    assert set(result["removed_features"]) == {"alpha", "beta"}

    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]
    assert full_report.columns[0] == "term"
    assert full_report.columns[1] == "pearson_r"
    assert full_report["pearson_r"].is_monotonic_decreasing
    assert full_report.iloc[0]["term"] == "delta"


def test_spearman_filter_tool():
    # Instantiate session state exactly once
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM matrix first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate filtering tools and verify all three tools are returned
    filtering_tools = make_filtering_tools(session)
    tool_names = [t.name for t in filtering_tools]
    assert "variance_filter_tool" in tool_names
    assert "pearson_filter_tool" in tool_names
    assert "spearman_filter_tool" in tool_names

    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Run spearman_filter_tool with target_class="catB" and threshold=0.0
    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    # Known Answer #6 assertions from TEST_FIXTURE.md:
    spearman_scores = result["spearman_r"]
    assert spearman_scores["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert spearman_scores["always"] == pytest.approx(0.3780, abs=1e-3)
    assert spearman_scores["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert spearman_scores["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert spearman_scores["gamma"] == pytest.approx(0.9363, abs=1e-3)

    # Retained and removed features at threshold=0.0
    assert result["retained_features_count"] == 3
    assert set(result["retained_features"]) == {"delta", "gamma", "always"}
    assert result["removed_features_count"] == 2
    assert set(result["removed_features"]) == {"alpha", "beta"}

    # Verify full_report column structure and descending order sorting
    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]
    assert full_report.columns[0] == "term"
    assert full_report.columns[1] == "spearman_r"
    assert full_report["spearman_r"].is_monotonic_decreasing
    assert full_report.iloc[0]["term"] == "delta"  # Highest correlation r = 1.0
