import json
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pydantic import ValidationError
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def setup_fixture_session():
    """Helper to initialize SessionState and load sample_fixture.csv via load_dataset_tool."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session


def test_tokenize_tool_known_answer_10():
    """
    Tests tokenize_tool against TEST_FIXTURE.md Known Answer #10.
    Verifies all 8 expected token lists, session.dataframe["unigrams"],
    session.artifacts["unigrams"], and summary stats.
    """
    session = setup_fixture_session()
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({"lowercase": True})

    # Expected token lists per Known Answer #10
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],  # Row 7 is empty text
    ]

    # 1. Verify summary fields
    assert res["status"] == "success"
    assert res["total_documents"] == 8
    expected_total_tokens = sum(len(doc) for doc in expected_tokens)  # 27
    assert res["total_tokens"] == expected_total_tokens
    assert res["avg_tokens_per_doc"] == round(expected_total_tokens / 8, 2)
    assert res["lowercase"] is True

    # 2. Verify session.dataframe["unigrams"]
    df_unigrams = session.dataframe["unigrams"].tolist()
    assert df_unigrams == expected_tokens

    # 3. Verify session.artifacts["unigrams"]
    assert session.artifacts["unigrams"] == expected_tokens


def test_tokenize_tool_lowercase_flag():
    """
    Verifies tokenize_tool casing behavior with mixed-case text for both lowercase=False and lowercase=True.
    """
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["Hello WORLD", "Foo BAR"],
        "label": ["A", "B"]
    })

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # Case 1: lowercase=False preserves original casing
    res_no_lower = tokenize_tool.invoke({"lowercase": False})
    assert res_no_lower["status"] == "success"
    assert session.dataframe["unigrams"].tolist() == [["Hello", "WORLD"], ["Foo", "BAR"]]

    # Case 2: lowercase=True converts to lower
    res_lower = tokenize_tool.invoke({"lowercase": True})
    assert res_lower["status"] == "success"
    assert session.dataframe["unigrams"].tolist() == [["hello", "world"], ["foo", "bar"]]


def test_tokenize_tool_empty_or_missing_text():
    """
    Verifies empty text or NaN/None values produce empty token lists [].
    """
    session = SessionState()
    # Create custom dataframe with NaN, None, empty string
    session.dataframe = pd.DataFrame({
        "text": ["hello world", None, "", float("nan")],
        "label": ["A", "A", "B", "B"]
    })

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({"lowercase": True})

    assert res["status"] == "success"
    assert res["total_documents"] == 4
    assert res["total_tokens"] == 2
    
    tokens = session.dataframe["unigrams"].tolist()
    assert tokens[0] == ["hello", "world"]
    assert tokens[1] == []
    assert tokens[2] == []
    assert tokens[3] == []


def test_tokenize_tool_error_handling():
    """
    Verifies error handling when no dataframe is loaded or 'text' column is missing.
    """
    session = SessionState()
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # Case 1: No dataframe loaded
    res_no_df = tokenize_tool.invoke({})
    assert "error" in res_no_df
    assert res_no_df["error"] == "No dataframe loaded in session."

    # Case 2: Missing 'text' column
    session.dataframe = pd.DataFrame({"other_col": [1, 2, 3]})
    res_no_text = tokenize_tool.invoke({})
    assert "error" in res_no_text
    assert res_no_text["error"] == "Dataframe missing required 'text' column."


# --- list_files_tool Tests ---

def test_list_files_tool_known_answer_11():
    """
    Verifies list_files_tool against TEST_FIXTURE.md Known Answer #11:
    list_files_tool(subdirectory="newdataset") returns directories: [],
    files: ["Reddit-stock-sentiment.csv"].
    Also verifies store_result recorded entry via session.results_store.
    """
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["status"] == "success"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]
    assert res["total_directories"] == 0
    assert res["total_files"] == 1
    assert res["result_id"].startswith("list_files_")
    
    # Verify session recorded entry
    assert res["result_id"] in session.results_store


def test_list_files_tool_sorting_and_root():
    """
    Verifies that files and directories are returned in sorted order when listing repo root or subfolders.
    """
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": ""})

    assert res["status"] == "success"
    # Directories and files must be sorted alphabetically
    assert res["directories"] == sorted(res["directories"])
    assert res["files"] == sorted(res["files"])


def test_list_files_tool_error_handling():
    """
    Verifies error handling for nonexistent directory and when path is a file, not a directory.
    """
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Case 1: Nonexistent directory
    res_nonexistent = list_files_tool.invoke({"subdirectory": "nonexistent_folder_xyz"})
    assert "error" in res_nonexistent
    assert "Directory not found" in res_nonexistent["error"]

    # Case 2: Path is a file, not a directory
    res_file_path = list_files_tool.invoke({"subdirectory": "newdataset/Reddit-stock-sentiment.csv"})
    assert "error" in res_file_path
    assert "Path is not a directory" in res_file_path["error"]


def test_list_files_tool_traversal_protection(tmp_path, monkeypatch):
    """
    Verifies directory traversal protection blocking ../ and absolute paths outside the repo root
    using platform-independent tmp_path and monkeypatch.
    """
    # Create controlled directory structure
    repo_root = tmp_path / "repo_root"
    repo_root.mkdir()
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()

    # Change working directory to repo_root
    monkeypatch.chdir(repo_root)

    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Case 1: Relative traversal out of repo root
    res_traversal = list_files_tool.invoke({"subdirectory": "../outside_dir"})
    assert "error" in res_traversal
    assert "Access denied" in res_traversal["error"]

    # Case 2: Platform-independent absolute path outside repo root
    res_abs_outside = list_files_tool.invoke({"subdirectory": str(outside_dir.resolve())})
    assert "error" in res_abs_outside
    assert "Access denied" in res_abs_outside["error"]


# --- inspect_data_tool Tests ---

def test_inspect_data_tool_default_and_rows_limit():
    """
    Tests inspect_data_tool default top 5 rows behavior and n_rows larger than dataset size.
    """
    session = setup_fixture_session()
    data_tools = make_tools(session)
    inspect_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # 1. Default n_rows=5 on 8-row dataset
    res = inspect_tool.invoke({})
    assert res["status"] == "success"
    assert res["total_rows"] == 8
    assert res["inspected_rows_count"] == 5
    assert len(res["sample_records"]) == 5

    # 2. n_rows=20 (larger than 8) returns all 8 rows
    res_large = inspect_tool.invoke({"n_rows": 20})
    assert res_large["status"] == "success"
    assert res_large["inspected_rows_count"] == 8
    assert len(res_large["sample_records"]) == 8


def test_inspect_data_tool_column_filtering_and_immutability():
    """
    Tests column filtering and verifies that session.dataframe is not mutated.
    """
    session = setup_fixture_session()
    original_cols = list(session.dataframe.columns)
    data_tools = make_tools(session)
    inspect_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_tool.invoke({"n_rows": 3, "columns": ["category_name"]})
    assert res["status"] == "success"
    assert res["columns"] == ["category_name"]
    assert res["inspected_rows_count"] == 3
    for rec in res["sample_records"]:
        assert list(rec.keys()) == ["category_name"]

    # Verify original dataframe columns unchanged
    assert list(session.dataframe.columns) == original_cols


def test_inspect_data_tool_validation_and_errors():
    """
    Tests error handling for missing DataFrame, invalid n_rows, and invalid or missing columns.
    """
    session = SessionState()
    data_tools = make_tools(session)
    inspect_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # 1. Missing DataFrame
    res_no_df = inspect_tool.invoke({})
    assert res_no_df["error"] == "No dataframe loaded in session."

    # Set up valid DataFrame
    session.dataframe = pd.DataFrame({"colA": [1, 2], "colB": ["x", "y"]})

    # 2. Invalid n_rows (negative, zero, or bool)
    assert "error" in inspect_tool.invoke({"n_rows": 0})
    assert "error" in inspect_tool.invoke({"n_rows": -5})
    assert "error" in inspect_tool.invoke({"n_rows": True})

    # 3. Invalid columns format (non-list or non-string elements trigger ValidationError from Pydantic)
    with pytest.raises(ValidationError):
        inspect_tool.invoke({"columns": "colA"})

    with pytest.raises(ValidationError):
        inspect_tool.invoke({"columns": [123]})

    # 4. Missing column names
    res_missing_col = inspect_tool.invoke({"columns": ["colA", "nonexistent_col"]})
    assert "error" in res_missing_col
    assert "Columns not found" in res_missing_col["error"]


def test_inspect_data_tool_serialization_infinity_nested_and_duplicates():
    """
    Tests serialization handling for nested lists/dicts, missing values, np.inf,
    strict json.dumps validation, and duplicate stringified column labels.
    """
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "num_col": [1.5, np.inf, -np.inf, np.nan],
        "list_col": [["a", "b"], [], None, [1, 2]],
        "dict_col": [{"k": "v"}, {}, None, {"a": 1}],
        "timestamp_col": [pd.Timestamp("2026-01-01"), pd.NaT, None, pd.Timestamp("2026-02-01")]
    })

    data_tools = make_tools(session)
    inspect_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_tool.invoke({"n_rows": 4})
    assert res["status"] == "success"

    # Strict JSON serialization check (allow_nan=False must pass)
    json_str = json.dumps(res, allow_nan=False)
    assert json_str is not None

    # Check sanitized values
    records = res["sample_records"]
    assert records[0]["num_col"] == 1.5
    assert records[1]["num_col"] is None  # inf -> None
    assert records[2]["num_col"] is None  # -inf -> None
    assert records[3]["num_col"] is None  # nan -> None
    assert records[0]["list_col"] == ["a", "b"]
    assert records[0]["dict_col"] == {"k": "v"}

    # Test duplicate stringified column names
    session_dup = SessionState()
    session_dup.dataframe = pd.DataFrame([[1, 2]], columns=["col", "col"])
    data_tools_dup = make_tools(session_dup)
    inspect_tool_dup = next(t for t in data_tools_dup if t.name == "inspect_data_tool")

    res_dup = inspect_tool_dup.invoke({})
    assert "error" in res_dup
    assert "Duplicate column names" in res_dup["error"]


# --- check_missing_tool Tests ---

def test_check_missing_tool_known_answer_2():
    """
    Tests check_missing_tool against TEST_FIXTURE.md Known Answer #2:
    8 rows, 1 missing text at 0-based positional row index [7].
    """
    session = setup_fixture_session()
    data_tools = make_tools(session)
    missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    res = missing_tool.invoke({})

    assert res["status"] == "success"
    assert res["total_rows"] == 8
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]
    assert res["result_id"].startswith("missing_")
    assert res["result_id"] in session.results_store


def test_check_missing_tool_edge_cases_and_non_default_index():
    """
    Tests check_missing_tool with None, NaN, empty string, whitespace-only text,
    non-default DataFrame index (verifying positional indices), and immutability.
    """
    session = SessionState()
    # Non-default DataFrame index labels: [100, 200, 300, 400, 500, 600]
    custom_df = pd.DataFrame({
        "text": ["valid text", None, float("nan"), "", "   ", "another valid"],
        "label": ["catA", "catA", "catB", "catB", "catA", "catB"]
    }, index=[100, 200, 300, 400, 500, 600])

    session.dataframe = custom_df.copy()

    data_tools = make_tools(session)
    missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    res = missing_tool.invoke({})

    assert res["status"] == "success"
    assert res["total_rows"] == 6
    assert res["missing_count"] == 4
    # Must return 0-based positional row indices [1, 2, 3, 4], NOT index labels [200, 300, 400, 500]
    assert res["missing_indices"] == [1, 2, 3, 4]

    # Verify DataFrame immutability
    pd.testing.assert_frame_equal(session.dataframe, custom_df)


def test_check_missing_tool_error_handling():
    """
    Tests error handling when no dataframe is loaded or 'text' column is missing.
    """
    session = SessionState()
    data_tools = make_tools(session)
    missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # 1. No dataframe loaded
    res_no_df = missing_tool.invoke({})
    assert res_no_df["error"] == "No dataframe loaded in session."

    # 2. Missing 'text' column
    session.dataframe = pd.DataFrame({"other_col": ["a", "b", "c"]})
    res_no_text = missing_tool.invoke({})
    assert res_no_text["error"] == "Dataframe missing required 'text' column."


# --- check_duplicates_tool Tests ---

def test_check_duplicates_tool_known_answer_3_check_only():
    """
    Tests check_duplicates_tool in check-only mode (drop=False) against Known Answer #3.
    Verifies duplicate count=2 (using keep=False), removed_count=0, remaining_rows=8,
    and session.dataframe immutability.
    """
    session = setup_fixture_session()
    initial_len = len(session.dataframe)
    data_tools = make_tools(session)
    dup_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    res = dup_tool.invoke({"subset": ["text"], "drop": False})

    assert res["status"] == "success"
    assert res["initial_rows"] == 8
    assert res["duplicate_count"] == 2
    assert res["removed_count"] == 0
    assert res["remaining_rows"] == 8
    assert res["dropped"] is False
    assert res["subset"] == ["text"]
    assert res["result_id"].startswith("duplicates_")
    assert res["result_id"] in session.results_store

    # Verify session.dataframe was not mutated
    assert len(session.dataframe) == initial_len


def test_check_duplicates_tool_known_answer_3_drop():
    """
    Tests check_duplicates_tool with drop=True against Known Answer #3.
    Verifies that keep=False behavior drops BOTH duplicated rows (rows 5 and 6),
    leaving exactly 6 remaining rows.
    """
    session = setup_fixture_session()
    data_tools = make_tools(session)
    dup_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    res = dup_tool.invoke({"subset": ["text"], "drop": True})

    assert res["status"] == "success"
    assert res["initial_rows"] == 8
    assert res["duplicate_count"] == 2
    assert res["removed_count"] == 2
    assert res["remaining_rows"] == 6
    assert res["dropped"] is True
    assert res["subset"] == ["text"]

    # Verify session.dataframe was actually updated to 6 rows and index reset
    assert len(session.dataframe) == 6
    assert list(session.dataframe.index) == list(range(6))
    # Verify both copies of "always gamma delta" were removed
    assert "always gamma delta" not in session.dataframe["text"].values


def test_check_duplicates_tool_non_default_index_and_immutability():
    """
    Tests check_duplicates_tool on custom DataFrame with non-default index
    and custom subset evaluation.
    """
    session = SessionState()
    custom_df = pd.DataFrame({
        "col1": ["a", "a", "b", "c"],
        "col2": ["x", "x", "y", "z"]
    }, index=[10, 20, 30, 40])
    session.dataframe = custom_df.copy()

    data_tools = make_tools(session)
    dup_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # Check duplicates on subset=["col1", "col2"] with drop=False
    res = dup_tool.invoke({"subset": ["col1", "col2"], "drop": False})

    assert res["status"] == "success"
    assert res["initial_rows"] == 4
    assert res["duplicate_count"] == 2
    assert res["removed_count"] == 0
    assert res["remaining_rows"] == 4
    assert res["subset"] == ["col1", "col2"]

    # Verify original non-default index preserved when drop=False
    pd.testing.assert_frame_equal(session.dataframe, custom_df)


def test_check_duplicates_tool_no_duplicates():
    """
    Tests check_duplicates_tool when no duplicates exist in DataFrame.
    """
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["doc1", "doc2", "doc3"]})

    data_tools = make_tools(session)
    dup_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    res = dup_tool.invoke({"subset": ["text"], "drop": True})

    assert res["status"] == "success"
    assert res["initial_rows"] == 3
    assert res["duplicate_count"] == 0
    assert res["removed_count"] == 0
    assert res["remaining_rows"] == 3
    assert len(session.dataframe) == 3


def test_check_duplicates_tool_error_handling():
    """
    Tests error handling for missing DataFrame, missing 'text' default column,
    invalid subset type, empty subset list, and missing subset columns.
    """
    session = SessionState()
    data_tools = make_tools(session)
    dup_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # 1. No DataFrame loaded
    res_no_df = dup_tool.invoke({})
    assert res_no_df["error"] == "No dataframe loaded in session."

    # 2. Default subset=["text"] missing from DataFrame
    session.dataframe = pd.DataFrame({"other": [1, 2]})
    res_no_text = dup_tool.invoke({})
    assert res_no_text["error"] == "Dataframe missing required default 'text' column for duplicate checking."

    # 3. Invalid subset format (non-list/tuple triggers ValidationError or error dict)
    with pytest.raises(ValidationError):
        dup_tool.invoke({"subset": "col1"})

    # 4. Empty subset list
    res_empty_subset = dup_tool.invoke({"subset": []})
    assert res_empty_subset["error"] == "subset cannot be empty."

    # 5. Subset contains columns not present in DataFrame
    res_missing_col = dup_tool.invoke({"subset": ["other", "nonexistent"]})
    assert "Columns not found in dataframe" in res_missing_col["error"]


# --- sample_data_tool Tests ---

def test_sample_data_tool_known_answer_13():
    """
    Tests sample_data_tool against Known Answer #13:
    n=4, random_state=42 returns exactly rows [1, 5, 0, 7].
    Verifies DataFrame mutation, index reset, and session result recording.
    """
    session = setup_fixture_session()
    original_texts = session.dataframe["text"].tolist()
    expected_texts = [original_texts[i] for i in [1, 5, 0, 7]]

    data_tools = make_tools(session)
    sample_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    res = sample_tool.invoke({"n": 4, "random_state": 42})

    assert res["status"] == "success"
    assert res["initial_rows"] == 8
    assert res["sampled_rows"] == 4
    assert res["sampling_mode"] == "n"
    assert res["parameter_value"] == 4
    assert res["replace"] is False
    assert res["random_state"] == 42
    assert res["result_id"].startswith("sample_")
    assert res["result_id"] in session.results_store

    # Verify session.dataframe was mutated in-place and index reset to [0, 1, 2, 3]
    assert len(session.dataframe) == 4
    assert list(session.dataframe.index) == [0, 1, 2, 3]
    assert session.dataframe["text"].tolist() == expected_texts


def test_sample_data_tool_frac_and_reproducibility():
    """
    Tests sample_data_tool with frac parameter and verifies reproducibility with random_state.
    """
    session1 = setup_fixture_session()
    data_tools1 = make_tools(session1)
    sample_tool1 = next(t for t in data_tools1 if t.name == "sample_data_tool")

    res1 = sample_tool1.invoke({"frac": 0.5, "random_state": 42})
    assert res1["status"] == "success"
    assert res1["initial_rows"] == 8
    assert res1["sampled_rows"] == 4
    assert res1["sampling_mode"] == "frac"
    assert res1["parameter_value"] == 0.5

    # Repeat with second session and same random_state
    session2 = setup_fixture_session()
    data_tools2 = make_tools(session2)
    sample_tool2 = next(t for t in data_tools2 if t.name == "sample_data_tool")

    res2 = sample_tool2.invoke({"frac": 0.5, "random_state": 42})
    
    # Verify exact same sampled text rows
    pd.testing.assert_frame_equal(session1.dataframe, session2.dataframe)


def test_sample_data_tool_replacement_and_oversampling():
    """
    Tests sampling with replacement=True, including oversampling (n > initial_rows and frac > 1.0).
    """
    session = setup_fixture_session()
    data_tools = make_tools(session)
    sample_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # 1. Oversampling with n=12 (replace=True on 8-row dataset)
    res_n_over = sample_tool.invoke({"n": 12, "replace": True, "random_state": 42})
    assert res_n_over["status"] == "success"
    assert res_n_over["initial_rows"] == 8
    assert res_n_over["sampled_rows"] == 12
    assert len(session.dataframe) == 12

    # Reset session state
    session_frac = setup_fixture_session()
    data_tools_frac = make_tools(session_frac)
    sample_tool_frac = next(t for t in data_tools_frac if t.name == "sample_data_tool")

    # 2. Oversampling with frac=1.5 (replace=True on 8-row dataset -> 12 rows)
    res_frac_over = sample_tool_frac.invoke({"frac": 1.5, "replace": True, "random_state": 42})
    assert res_frac_over["status"] == "success"
    assert res_frac_over["initial_rows"] == 8
    assert res_frac_over["sampled_rows"] == 12
    assert len(session_frac.dataframe) == 12


def test_sample_data_tool_error_handling_and_invalid_inputs():
    """
    Tests error handling for missing DataFrame, parameter conflicts, invalid types/booleans,
    out of range values, NaN, and infinity.
    """
    session = SessionState()
    data_tools = make_tools(session)
    sample_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # 1. No DataFrame loaded
    res_no_df = sample_tool.invoke({"n": 4})
    assert res_no_df["error"] == "No dataframe loaded in session."

    session = setup_fixture_session()
    data_tools = make_tools(session)
    sample_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # 2. Conflict: both n and frac provided or neither provided
    res_both = sample_tool.invoke({"n": 4, "frac": 0.5})
    assert "Must provide exactly one" in res_both["error"]

    res_neither = sample_tool.invoke({})
    assert "Must provide exactly one" in res_neither["error"]

    # 3. Invalid n (bool, zero, negative, float, n > initial_rows when replace=False)
    assert "error" in sample_tool.invoke({"n": True})
    assert "error" in sample_tool.invoke({"n": 0})
    assert "error" in sample_tool.invoke({"n": -2})
    assert "error" in sample_tool.invoke({"n": 20, "replace": False})

    # 4. Invalid frac (bool, zero, negative, NaN, Inf, frac > 1.0 when replace=False)
    assert "error" in sample_tool.invoke({"frac": True})
    assert "error" in sample_tool.invoke({"frac": 0.0})
    assert "error" in sample_tool.invoke({"frac": -0.5})
    assert "error" in sample_tool.invoke({"frac": float("nan")})
    assert "error" in sample_tool.invoke({"frac": float("inf")})
    assert "error" in sample_tool.invoke({"frac": float("-inf")})
    assert "error" in sample_tool.invoke({"frac": 1.5, "replace": False})

    # 5. Invalid replace (non-bool) or invalid random_state (bool, negative)
    assert "error" in sample_tool.invoke({"n": 2, "replace": "yes"})
    assert "error" in sample_tool.invoke({"n": 2, "random_state": True})
    assert "error" in sample_tool.invoke({"n": 2, "random_state": -1})


# --- describe_data_tool Tests ---

def test_describe_data_tool_known_answer_18():
    """
    Tests describe_data_tool against Known Answer #18 in TEST_FIXTURE.md.
    Verifies overall and per-category stats using approximate numerical tolerance,
    session.pending_figure boxplot generation, and DataFrame immutability.
    """
    session = setup_fixture_session()
    original_df = session.dataframe.copy()

    data_tools = make_tools(session)
    describe_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    res = describe_tool.invoke({})

    assert res["status"] == "success"
    assert res["result_id"].startswith("describe_")
    assert res["result_id"] in session.results_store

    # Compute expected std dynamically using pandas for catA lengths [29, 23, 17, 0]
    expected_catA_std = float(pd.Series([29, 23, 17, 0]).std())

    # 1. Overall Stats Verification (Known Answer #18)
    overall = res["overall_stats"]
    assert overall["count"] == pytest.approx(8.0)
    assert overall["mean"] == pytest.approx(19.875)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == pytest.approx(0.0)
    assert overall["25%"] == pytest.approx(17.75)
    assert overall["50%"] == pytest.approx(20.5)
    assert overall["75%"] == pytest.approx(25.25)
    assert overall["max"] == pytest.approx(30.0)

    # 2. Per-Category Stats Verification (Known Answer #18)
    per_cat = res["per_category_stats"]
    assert "catA" in per_cat
    assert "catB" in per_cat

    # catA
    catA = per_cat["catA"]
    assert catA["count"] == pytest.approx(4.0)
    assert catA["mean"] == pytest.approx(17.25)
    assert catA["std"] == pytest.approx(expected_catA_std)
    assert catA["min"] == pytest.approx(0.0)
    assert catA["25%"] == pytest.approx(12.75)
    assert catA["50%"] == pytest.approx(20.0)
    assert catA["75%"] == pytest.approx(24.5)
    assert catA["max"] == pytest.approx(29.0)

    # catB
    catB = per_cat["catB"]
    assert catB["count"] == pytest.approx(4.0)
    assert catB["mean"] == pytest.approx(22.5)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == pytest.approx(18.0)
    assert catB["25%"] == pytest.approx(18.0)
    assert catB["50%"] == pytest.approx(21.0)
    assert catB["75%"] == pytest.approx(25.5)
    assert catB["max"] == pytest.approx(30.0)

    # 3. Figure Verification
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)

    # 4. DataFrame Immutability Verification
    pd.testing.assert_frame_equal(session.dataframe, original_df)


def test_describe_data_tool_error_handling():
    """
    Tests error handling for missing DataFrame, missing required columns ('text', 'category_name'),
    non-string text items, and empty DataFrame.
    """
    session = SessionState()
    data_tools = make_tools(session)
    describe_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    # 1. No DataFrame loaded
    res_no_df = describe_tool.invoke({})
    assert res_no_df["error"] == "No dataframe loaded in session."

    # 2. Missing 'text' column
    session.dataframe = pd.DataFrame({"category_name": ["catA"]})
    res_no_text = describe_tool.invoke({})
    assert "Dataframe missing required 'text' column." in res_no_text["error"]

    # 3. Missing 'category_name' column
    session.dataframe = pd.DataFrame({"text": ["hello"]})
    res_no_cat = describe_tool.invoke({})
    assert "Dataframe missing required 'category_name' column." in res_no_cat["error"]

    # 4. Non-string items in 'text' column (e.g. integer or None)
    session.dataframe = pd.DataFrame({
        "text": ["hello", 123, None],
        "category_name": ["catA", "catA", "catB"]
    })
    res_non_str = describe_tool.invoke({})
    assert "All values in 'text' column must be strings." in res_non_str["error"]

    # 5. Empty DataFrame
    session.dataframe = pd.DataFrame({"text": [], "category_name": []})
    res_empty = describe_tool.invoke({})
    assert "Dataframe is empty." in res_empty["error"]
