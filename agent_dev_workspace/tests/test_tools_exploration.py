import pytest
import json
import numpy as np
import scipy.sparse as sp
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools
from tools_dtm import make_tools as make_dtm_tools


def setup_fixture_session():
    """
    Helper to initialize SessionState, load sample_fixture.csv via load_dataset_tool,
    and build the global DTM via build_dtm_tool.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM to populate session.feature_matrix and session.feature_names
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session


def test_cosine_similarity_tool_known_answer_16():
    """
    Tests cosine_similarity_tool against TEST_FIXTURE.md Known Answer #16:
    0 vs 1 (both catA, near-identical text) = 0.9847
    0 vs 3 (catA vs catB, share only "always") = 0.0909
    3 vs 4 (both catB, near-identical text) = 0.9847
    0 vs 7 (catA vs empty-text row) = 0.0
    """
    session = setup_fixture_session()
    exploration_tools = make_tools(session)
    cosine_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # 1. Doc 0 vs Doc 1 (0.9847)
    res_0_1 = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["status"] == "success"
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-4)
    assert res_0_1["result_id"].startswith("cosine_")
    assert res_0_1["result_id"] in session.results_store

    # 2. Doc 0 vs Doc 3 (0.0909)
    res_0_3 = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["status"] == "success"
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-4)

    # 3. Doc 3 vs Doc 4 (0.9847)
    res_3_4 = cosine_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["status"] == "success"
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-4)

    # 4. Doc 0 vs Doc 7 (empty text -> 0.0 exactly)
    res_0_7 = cosine_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["status"] == "success"
    assert res_0_7["similarity"] == 0.0


def test_cosine_similarity_tool_error_handling():
    """
    Tests error handling for missing DataFrame, missing DTM, invalid/boolean indices,
    out-of-bounds indices, and mismatched DataFrame/DTM row counts.
    """
    # Case 1: Missing DataFrame
    session1 = SessionState()
    exploration_tools1 = make_tools(session1)
    cosine_tool1 = next(t for t in exploration_tools1 if t.name == "cosine_similarity_tool")

    res_no_df = cosine_tool1.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_no_df["error"] == "No dataframe loaded in session."

    # Case 2: Missing DTM
    session2 = SessionState()
    session2.dataframe = pd.DataFrame({"text": ["a", "b"], "category_name": ["catA", "catB"]})
    exploration_tools2 = make_tools(session2)
    cosine_tool2 = next(t for t in exploration_tools2 if t.name == "cosine_similarity_tool")

    res_no_dtm = cosine_tool2.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert "No Document-Term Matrix found" in res_no_dtm["error"]

    # Case 3: Boolean or invalid type indices
    session3 = setup_fixture_session()
    exploration_tools3 = make_tools(session3)
    cosine_tool3 = next(t for t in exploration_tools3 if t.name == "cosine_similarity_tool")

    assert "error" in cosine_tool3.invoke({"doc_index_1": True, "doc_index_2": 1})
    assert "error" in cosine_tool3.invoke({"doc_index_1": 0, "doc_index_2": "invalid"})

    # Case 4: Out-of-bounds indices
    assert "out of bounds" in cosine_tool3.invoke({"doc_index_1": -1, "doc_index_2": 1})["error"]
    assert "out of bounds" in cosine_tool3.invoke({"doc_index_1": 0, "doc_index_2": 20})["error"]

    # Case 5: Mismatched row counts between session.dataframe and session.feature_matrix
    session4 = setup_fixture_session()
    # Shrink session.dataframe without rebuilding DTM
    session4.dataframe = session4.dataframe.head(4).copy()
    exploration_tools4 = make_tools(session4)
    cosine_tool4 = next(t for t in exploration_tools4 if t.name == "cosine_similarity_tool")

    res_mismatch = cosine_tool4.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert "row count" in res_mismatch["error"]


def test_feature_correlation_matrix_tool_known_answer_19():
    """
    Tests feature_correlation_matrix_tool against Known Answer #19 in TEST_FIXTURE.md.
    Sample fixture terms by variance descending: alpha, gamma, delta, beta, always.
    """
    session = setup_fixture_session()
    exploration_tools = make_tools(session)
    feat_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    result = feat_corr_tool.invoke({})

    assert result["status"] == "success"
    assert result["n_terms"] == 5

    # Verify result_id is in session.results_store
    assert "result_id" in result
    assert result["result_id"] in session.results_store

    # Verify JSON serializability
    json_str = json.dumps(result)
    assert isinstance(json_str, str)

    # Verify term order by descending variance: alpha, gamma, delta, beta, always
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert result["terms"] == expected_terms

    # Verify figure generated
    assert session.pending_figure is not None

    # Expected correlation matrix from TEST_FIXTURE.md #19:
    expected_matrix = [
        [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
        [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
        [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
        [0.2601, 0.3140, 0.3780, 0.2928, 1.0000],
    ]

    actual_matrix = np.array(result["correlation_matrix"])
    assert actual_matrix.shape == (5, 5)
    np.testing.assert_allclose(actual_matrix, expected_matrix, atol=1e-3)


def test_feature_correlation_matrix_edge_cases():
    """
    Tests edge cases:
    - Fewer than 2 documents returns an error
    - Single feature case
    - Zero-variance features (no 1.0 forced on zero variance diagonal)
    """
    # 1. Fewer than 2 documents
    session1 = SessionState()
    session1.feature_matrix = np.array([[1, 2, 3]])
    session1.feature_names = ["t1", "t2", "t3"]

    tools1 = make_tools(session1)
    tool1 = next(t for t in tools1 if t.name == "feature_correlation_matrix_tool")
    res1 = tool1.invoke({})
    assert "error" in res1
    assert "fewer than 2 documents" in res1["error"]

    # 2. Single feature
    session2 = SessionState()
    session2.feature_matrix = np.array([[1], [2], [3]])
    session2.feature_names = ["t1"]

    tools2 = make_tools(session2)
    tool2 = next(t for t in tools2 if t.name == "feature_correlation_matrix_tool")
    res2 = tool2.invoke({})
    assert res2["status"] == "success"
    assert res2["n_terms"] == 1
    assert res2["terms"] == ["t1"]
    assert res2["correlation_matrix"] == [[1.0]]

    # 3. Zero variance features
    session3 = SessionState()
    # Feature 0 is constant (zero variance), Feature 1 varies
    session3.feature_matrix = np.array([
        [5, 1],
        [5, 2],
        [5, 3]
    ])
    session3.feature_names = ["constant_term", "varying_term"]

    tools3 = make_tools(session3)
    tool3 = next(t for t in tools3 if t.name == "feature_correlation_matrix_tool")
    res3 = tool3.invoke({})
    assert res3["status"] == "success"
    # constant_term has variance 0, varying_term has variance > 0
    # varying_term should be ranked first due to higher variance
    assert res3["terms"] == ["varying_term", "constant_term"]

    matrix3 = res3["correlation_matrix"]
    # varying_term vs varying_term = 1.0
    assert matrix3[0][0] == pytest.approx(1.0)
    # constant_term vs constant_term (diagonal for zero variance) = 0.0 (not forced to 1.0)
    assert matrix3[1][1] == 0.0
    # constant_term vs varying_term = 0.0
    assert matrix3[0][1] == 0.0
    assert matrix3[1][0] == 0.0
