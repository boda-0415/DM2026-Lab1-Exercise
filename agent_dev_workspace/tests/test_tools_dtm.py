import matplotlib.figure
import matplotlib.pyplot as plt
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
    # Instantiate session state exactly once
    session = SessionState()

    # Load load_dataset_tool from premade_tools/
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student's tools_dtm tools
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # Run build_dtm_tool
    result = build_dtm_tool.invoke({})

    # Known Answer #4 assertions from TEST_FIXTURE.md:
    assert result["n_terms"] == 5
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert result["n_documents"] == 8
    assert result["total_elements"] == 40
    assert result["non_zero"] == 21
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool():
    # Instantiate session state exactly once
    session = SessionState()

    # Load load_dataset_tool from premade_tools/
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM matrix first
    tools = make_tools(session)
    tool_names = [t.name for t in tools]
    assert "build_dtm_tool" in tool_names
    assert "term_frequency_tool" in tool_names

    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})

    # Run term_frequency_tool with default parameters
    result = term_frequency_tool.invoke({})

    # Known Answer #14 assertions from TEST_FIXTURE.md:
    assert result["total_terms"] == 5
    assert result["total_word_count"] == 27

    freqs = result["term_frequencies"]
    assert freqs["alpha"] == 6
    assert freqs["always"] == 7
    assert freqs["beta"] == 3
    assert freqs["delta"] == 4
    assert freqs["gamma"] == 7

    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]
    assert full_report.columns[0] == "term"
    assert full_report.columns[1] == "frequency"
    assert len(full_report) == 5
    assert full_report.iloc[0]["term"] == "always"
    assert full_report.iloc[0]["frequency"] == 7
    assert full_report.iloc[1]["term"] == "gamma"
    assert full_report.iloc[1]["frequency"] == 7
    assert full_report.iloc[2]["term"] == "alpha"
    assert full_report.iloc[2]["frequency"] == 6
    assert full_report.iloc[3]["term"] == "delta"
    assert full_report.iloc[3]["frequency"] == 4
    assert full_report.iloc[4]["term"] == "beta"
    assert full_report.iloc[4]["frequency"] == 3

    res_top2 = term_frequency_tool.invoke({"top_n": 2})
    assert len(res_top2["top_terms"]) == 2
    assert res_top2["top_terms"][0] == {"term": "always", "frequency": 7}
    assert res_top2["top_terms"][1] == {"term": "gamma", "frequency": 7}

    stored_top2 = session.results_store[res_top2["result_id"]]
    assert len(stored_top2["full_report"]) == 5

    err_zero = term_frequency_tool.invoke({"top_n": 0})
    assert "error" in err_zero
    err_neg = term_frequency_tool.invoke({"top_n": -1})
    assert "error" in err_neg


def test_dtm_heatmap_tool():
    try:
        # Instantiate session state exactly once
        session = SessionState()

        # Load premade load_dataset_tool
        premade_tools, _ = load_workspace_tools("premade_tools", session)
        load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

        # Load fixture dataset into session
        load_dataset_tool.invoke({
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label"
        })

        # Instantiate tools and verify all three tools are registered
        tools = make_tools(session)
        tool_names = [t.name for t in tools]
        assert "build_dtm_tool" in tool_names
        assert "term_frequency_tool" in tool_names
        assert "dtm_heatmap_tool" in tool_names

        build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
        dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

        build_dtm_tool.invoke({})

        # 1. Default n_terms=20, n_documents=20 returns full 8x5 matrix matching session.feature_matrix
        result_default = dtm_heatmap_tool.invoke({})
        assert result_default["n_documents"] == 8
        assert result_default["n_terms"] == 5
        assert result_default["included_terms"] == ["alpha", "always", "beta", "delta", "gamma"]
        assert result_default["included_documents"] == [f"Doc_{i}" for i in range(8)]
        assert result_default["sliced_matrix"] == session.feature_matrix.toarray().tolist()

        # Check session.pending_figure is set to a matplotlib Figure
        assert session.pending_figure is not None
        assert isinstance(session.pending_figure, matplotlib.figure.Figure)

        # 2. Known Answer #21 independent assertion: n_documents=3, n_terms=2 returns [[3, 1], [2, 1], [1, 1]]
        result_sliced = dtm_heatmap_tool.invoke({"n_documents": 3, "n_terms": 2})
        assert result_sliced["n_documents"] == 3
        assert result_sliced["n_terms"] == 2
        assert result_sliced["included_terms"] == ["alpha", "always"]
        assert result_sliced["included_documents"] == ["Doc_0", "Doc_1", "Doc_2"]
        assert result_sliced["sliced_matrix"] == [[3, 1], [2, 1], [1, 1]]

        # 3. Invalid parameter validation
        err_docs = dtm_heatmap_tool.invoke({"n_documents": 0})
        assert "error" in err_docs
        err_terms = dtm_heatmap_tool.invoke({"n_terms": -5})
        assert "error" in err_terms
    finally:
        plt.close("all")
