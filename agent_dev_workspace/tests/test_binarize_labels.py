import pandas as pd
from agent_pipeline.session_state import SessionState
from tools_reduction import make_tools as make_reduction_tools


def setup_session_with_data(data_dict):
    """
    Helper function to create a SessionState instance populated with custom DataFrame data.
    """
    session = SessionState()
    session.dataframe = pd.DataFrame(data_dict)
    tools = make_reduction_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")
    return session, binarize_tool


def test_binarize_labels_three_class():
    session, binarize_tool = setup_session_with_data({
        "category_name": ["catA", "catB", "catC", "catA", "catB"]
    })
    
    res = binarize_tool.invoke({})
    assert res["status"] == "success"
    assert res["column"] == "category_name"
    assert res["classes"] == ["catA", "catB", "catC"]
    assert res["shape"] == [5, 3]
    assert res["class_counts"] == {"catA": 2, "catB": 2, "catC": 1}
    
    expected_matrix = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ]
    assert session.artifacts["binarized_labels"]["encoded_labels"] == expected_matrix
    assert session.artifacts["binarized_labels"]["classes"] == ["catA", "catB", "catC"]
    assert session.artifacts["binarized_labels"]["column"] == "category_name"


def test_binarize_labels_binary_class_produces_two_columns():
    session, binarize_tool = setup_session_with_data({
        "category_name": ["yes", "no", "yes", "no"]
    })
    
    res = binarize_tool.invoke({})
    assert res["status"] == "success"
    assert res["classes"] == ["no", "yes"]
    assert res["shape"] == [4, 2]
    
    # Check that binary labels produce 2 columns [is_no, is_yes]
    expected_matrix = [
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0]
    ]
    assert session.artifacts["binarized_labels"]["encoded_labels"] == expected_matrix


def test_binarize_labels_default_priority_and_fallback():
    # 1. Test priority when both category_name and category exist
    session, binarize_tool = setup_session_with_data({
        "category_name": ["catA", "catB"],
        "category": [0, 1]
    })
    res = binarize_tool.invoke({})
    assert res["status"] == "success"
    assert res["column"] == "category_name"
    assert res["classes"] == ["catA", "catB"]

    # 2. Test fallback to category when category_name is absent and category contains strings
    session_fb, binarize_tool_fb = setup_session_with_data({
        "category": ["group1", "group2"]
    })
    res_fb = binarize_tool_fb.invoke({})
    assert res_fb["status"] == "success"
    assert res_fb["column"] == "category"
    assert res_fb["classes"] == ["group1", "group2"]


def test_binarize_labels_explicit_column_selection():
    session, binarize_tool = setup_session_with_data({
        "category_name": ["catA", "catB"],
        "custom_label": ["labelX", "labelY"]
    })
    res = binarize_tool.invoke({"column": "custom_label"})
    assert res["status"] == "success"
    assert res["column"] == "custom_label"
    assert res["classes"] == ["labelX", "labelY"]


def test_binarize_labels_missing_dataframe_and_column():
    session = SessionState()
    tools = make_reduction_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")
    
    # Missing dataframe
    res = binarize_tool.invoke({})
    assert "error" in res
    assert "No dataframe found" in res["error"]
    
    # Missing column
    session.dataframe = pd.DataFrame({"text": ["doc1", "doc2"]})
    res2 = binarize_tool.invoke({"column": "non_existent"})
    assert "error" in res2
    assert "non_existent" in res2["error"]


def test_binarize_labels_null_and_non_string_labels():
    # Null labels
    session, binarize_tool = setup_session_with_data({
        "category_name": ["catA", None, "catB"]
    })
    res = binarize_tool.invoke({})
    assert "error" in res
    assert "null/missing" in res["error"]

    # Non-string labels (e.g. integer category codes)
    session_int, binarize_tool_int = setup_session_with_data({
        "category": [0, 1, 0, 1]
    })
    res_int = binarize_tool_int.invoke({})
    assert "error" in res_int
    assert "non-string labels" in res_int["error"]


def test_binarize_labels_empty_dataframe():
    session, binarize_tool = setup_session_with_data({
        "category_name": pd.Series([], dtype=str)
    })
    res = binarize_tool.invoke({})
    assert "error" in res
    assert "empty" in res["error"]


def test_binarize_labels_dataframe_remains_unchanged():
    initial_df = pd.DataFrame({
        "text": ["hello", "world"],
        "category_name": ["catA", "catB"]
    })
    session = SessionState()
    session.dataframe = initial_df.copy()
    tools = make_reduction_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")
    
    res = binarize_tool.invoke({})
    assert res["status"] == "success"
    
    # Assert DataFrame columns and shape remain completely identical
    pd.testing.assert_frame_equal(session.dataframe, initial_df)
