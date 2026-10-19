import pytest
import json
import numpy as np
import scipy.sparse as sp
import pandas as pd
from pydantic import ValidationError
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def test_reduce_dimensions_pca_known_answer_7():
    """
    Tests reduce_dimensions_tool method='pca' against Known Answer #7 in TEST_FIXTURE.md.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "pca", "n_components": 2})

    assert res["status"] == "success"
    assert res["method"] == "pca"
    assert res["n_components"] == 2

    # Verify session.results_store and JSON serializability
    assert "result_id" in res
    assert res["result_id"] in session.results_store
    assert isinstance(json.dumps(res), str)

    # Verify pending figure
    assert session.pending_figure is not None

    # Expected PCA coordinates and explained_variance_ratio from Known Answer #7
    expected_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ]
    expected_var_ratio = [0.7532, 0.1965]

    actual_coords = np.array(res["coordinates"])
    actual_var_ratio = np.array(res["explained_variance_ratio"])

    assert actual_coords.shape == (8, 2)
    assert np.all(np.isfinite(actual_coords))

    np.testing.assert_allclose(actual_coords, expected_coords, atol=1e-3)
    np.testing.assert_allclose(actual_var_ratio, expected_var_ratio, atol=1e-3)


def test_reduce_dimensions_tsne_known_answer_8():
    """
    Tests reduce_dimensions_tool method='tsne' against Known Answer #8 in TEST_FIXTURE.md.
    Calls method="tsne", perplexity=2, random_state=42.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42,
    })

    assert res["status"] == "success"
    assert res["method"] == "tsne"

    # Verify session.results_store and JSON serializability
    assert res["result_id"] in session.results_store
    assert isinstance(json.dumps(res), str)
    assert session.pending_figure is not None

    # Expected t-SNE coordinates from Known Answer #8
    expected_coords = [
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973],
    ]

    actual_coords = np.array(res["coordinates"])
    assert actual_coords.shape == (8, 2)
    assert np.all(np.isfinite(actual_coords))

    # Calculate and report numerical differences if any
    diff = np.abs(actual_coords - np.array(expected_coords))
    max_diff = np.max(diff)
    print(f"\nt-SNE max absolute coordinate difference from Known Answer #8: {max_diff:.6f}")

    np.testing.assert_allclose(actual_coords, expected_coords, atol=1e-3)


@pytest.mark.skipif(not HAS_UMAP, reason="umap-learn is not installed")
def test_reduce_dimensions_umap_known_answer_9():
    """
    Tests reduce_dimensions_tool method='umap' against Known Answer #9 in TEST_FIXTURE.md.
    Calls method="umap", n_neighbors=3, random_state=42.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42,
    })

    assert res["status"] == "success"
    assert res["method"] == "umap"

    # Verify session.results_store and JSON serializability
    assert res["result_id"] in session.results_store
    assert isinstance(json.dumps(res), str)
    assert session.pending_figure is not None

    # Expected UMAP coordinates from Known Answer #9
    expected_coords = [
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622],
    ]

    actual_coords = np.array(res["coordinates"])
    assert actual_coords.shape == (8, 2)
    assert np.all(np.isfinite(actual_coords))

    # Calculate and report numerical differences if any
    diff = np.abs(actual_coords - np.array(expected_coords))
    max_diff = np.max(diff)
    print(f"\nUMAP max absolute coordinate difference from Known Answer #9: {max_diff:.6f}")

    np.testing.assert_allclose(actual_coords, expected_coords, atol=1e-3)


def test_reduce_dimensions_error_handling_and_sparse():
    """
    Tests error handling for missing DTM, invalid n_components, perplexity out of bounds,
    n_neighbors out of bounds, invalid method validation, and verifies sparse DTM input handling.
    """
    # 1. Missing DTM
    session1 = SessionState()
    tools1 = make_tools(session1)
    reduce1 = next(t for t in tools1 if t.name == "reduce_dimensions_tool")
    res1 = reduce1.invoke({})
    assert "error" in res1
    assert "No Document-Term Matrix found" in res1["error"]

    # Load session for remaining checks
    session2 = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session2)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    tools2 = make_tools(session2)
    reduce2 = next(t for t in tools2 if t.name == "reduce_dimensions_tool")

    # 2. Invalid n_components != 2
    res_comp = reduce2.invoke({"n_components": 3})
    assert "error" in res_comp
    assert "n_components must be 2" in res_comp["error"]

    # 3. Invalid perplexity for t-SNE (perplexity >= n_samples or <= 0)
    res_perp1 = reduce2.invoke({"method": "tsne", "perplexity": 30})
    assert "error" in res_perp1
    assert "perplexity" in res_perp1["error"]

    res_perp2 = reduce2.invoke({"method": "tsne", "perplexity": 0})
    assert "error" in res_perp2
    assert "perplexity" in res_perp2["error"]

    # 4. Invalid n_neighbors for UMAP (n_neighbors > n_samples or < 2)
    if HAS_UMAP:
        res_neigh1 = reduce2.invoke({"method": "umap", "n_neighbors": 15})
        assert "error" in res_neigh1
        assert "n_neighbors" in res_neigh1["error"]

        res_neigh2 = reduce2.invoke({"method": "umap", "n_neighbors": 1})
        assert "error" in res_neigh2
        assert "n_neighbors" in res_neigh2["error"]

    # 5. Invalid method throws ValidationError due to Literal["pca", "tsne", "umap"] schema
    with pytest.raises(ValidationError):
        reduce2.invoke({"method": "invalid_method"})

    # 6. Verify scipy CSR sparse matrix handling
    session3 = SessionState()
    dense_data = np.array([
        [1.0, 2.0, 0.0],
        [0.0, 3.0, 4.0],
        [5.0, 0.0, 1.0],
        [2.0, 2.0, 2.0],
    ])
    session3.feature_matrix = sp.csr_matrix(dense_data)
    session3.feature_names = ["f1", "f2", "f3"]

    tools3 = make_tools(session3)
    reduce3 = next(t for t in tools3 if t.name == "reduce_dimensions_tool")
    res_sparse = reduce3.invoke({"method": "pca"})
    assert res_sparse["status"] == "success"
    assert np.array(res_sparse["coordinates"]).shape == (4, 2)
