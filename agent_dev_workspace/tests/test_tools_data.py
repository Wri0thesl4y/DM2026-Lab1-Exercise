import pytest
import json
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_check_missing_tool_known_answer_2():
    session = SessionState()
    
    # Load premade load_dataset_tool to load the test fixture
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Initialize check_missing_tool from tools_data using the exact same session state
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")
    
    result = check_missing_tool.invoke({})
    
    # Assert against Known Answer #2 in TEST_FIXTURE.md:
    # Exactly 1 row has missing/empty text (the last row, index label 7)
    assert result["missing_count"] == 1
    assert result["missing_indices"] == [7]
    assert result["truncated"] is False

def test_check_duplicates_tool_known_answer_3():
    # Fresh session state for test 1 (drop=False)
    session1 = SessionState()
    premade_tools1, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool1 = next(t for t in premade_tools1 if t.name == "load_dataset_tool")
    load_dataset_tool1.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools1 = make_tools(session1)
    check_duplicates_tool1 = next(t for t in tools1 if t.name == "check_duplicates_tool")
    
    # Assert duplicate detection with drop=False: 1 duplicate found, 0 rows removed
    result_check = check_duplicates_tool1.invoke({"drop": False})
    assert result_check["duplicate_count"] == 1
    assert result_check["initial_rows"] == 8
    assert result_check["remaining_rows"] == 8
    assert result_check["rows_removed"] == 0
    
    # Fresh session state for test 2 (drop=True) to ensure clean state
    session2 = SessionState()
    premade_tools2, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool2 = next(t for t in premade_tools2 if t.name == "load_dataset_tool")
    load_dataset_tool2.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools2 = make_tools(session2)
    check_duplicates_tool2 = next(t for t in tools2 if t.name == "check_duplicates_tool")
    
    # Assert duplicate dropping with drop=True per Known Answer #3:
    # Row 5 and 6 are both dropped (keep=False), leaving 6 remaining rows and 2 rows removed.
    result_drop = check_duplicates_tool2.invoke({"drop": True})
    assert result_drop["duplicate_count"] == 1
    assert result_drop["initial_rows"] == 8
    assert result_drop["remaining_rows"] == 6
    assert result_drop["rows_removed"] == 2

def test_tokenize_tool_known_answer_10():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")
    
    result = tokenize_tool.invoke({})
    
    assert result["n_documents"] == 8
    assert result["total_tokens"] == 27
    
    # Check exact token lists per document against Known Answer #10 in TEST_FIXTURE.md
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]
    
    unigrams = session.dataframe["unigrams"].tolist()
    assert unigrams == expected_tokens

def test_sample_data_tool_known_answer_13():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")
    
    # Check Known Answer #13 in TEST_FIXTURE.md (n=4, random_state=42)
    result = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert result["sampled_rows"] == 4
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert result["truncated"] is False
    
    # Verify original dataframe is unmutated (still has 8 rows)
    assert len(session.dataframe) == 8
    
    # Validation error test: passing both n and frac must return an error dict
    error_result = sample_data_tool.invoke({"n": 4, "frac": 0.5})
    assert "error" in error_result

def test_list_files_tool_known_answer_11():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")
    
    # Check Known Answer #11 in TEST_FIXTURE.md
    result = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]
    assert result["n_directories"] == 0
    assert result["n_files"] == 1
    assert result["truncated"] is False
    
    # Security test: path escaping outside repo root must return an error dict
    escape_result = list_files_tool.invoke({"subdirectory": "../.."})
    assert "error" in escape_result

def test_inspect_data_tool_custom():
    # Test error when no dataset loaded
    session_empty = SessionState()
    tools_empty = make_tools(session_empty)
    inspect_empty = next(t for t in tools_empty if t.name == "inspect_data_tool")
    assert "error" in inspect_empty.invoke({})

    # Test with loaded fixture
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")
    
    # Tokenize to add list-valued unigrams column
    tokenize_tool.invoke({})
    
    # Add a synthetic row with text > 200 chars to verify truncation
    long_text = "a" * 250
    session.dataframe.loc[0, "text"] = long_text
    
    # Call inspect_data_tool(n=3)
    res_3 = inspect_data_tool.invoke({"n": 3})
    assert res_3["total_rows"] == 8
    assert res_3["rows_returned"] == 3
    assert "text" in res_3["columns"]
    assert "category" in res_3["columns"]
    assert "category_name" in res_3["columns"]
    assert "unigrams" in res_3["columns"]
    
    # Check text truncation on long text
    assert len(res_3["sample_rows"][0]["text"]) == 203  # 200 chars + "..."
    assert res_3["sample_rows"][0]["text"].endswith("...")
    
    # Check unigrams formatting as item count
    assert res_3["sample_rows"][0]["unigrams"] == "<5 items>"
    
    # Check JSON serializability
    json_str = json.dumps(res_3)
    assert isinstance(json_str, str)
    
    # Call inspect_data_tool(n=100): should return 8 rows (capped by dataset size)
    res_100 = inspect_data_tool.invoke({"n": 100})
    assert res_100["rows_returned"] == 8

def test_describe_data_tool_known_answer_18():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")
    
    result = describe_data_tool.invoke({})
    
    # Check text_length column values per Known Answer #18
    expected_text_lengths = [29, 23, 17, 30, 24, 18, 18, 0]
    assert session.dataframe["text_length"].tolist() == expected_text_lengths
    
    # Overall stats assertions using pytest.approx
    overall = result["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == pytest.approx(0.0, abs=1e-3)
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == pytest.approx(30.0, abs=1e-3)
    
    # Per-category stats assertions for catA
    catA = result["per_category_stats"]["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == pytest.approx(0.0, abs=1e-3)
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == pytest.approx(29.0, abs=1e-3)
    
    # Per-category stats assertions for catB
    catB = result["per_category_stats"]["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == pytest.approx(18.0, abs=1e-3)
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == pytest.approx(30.0, abs=1e-3)
    
    # Check that session.pending_figure is set
    assert session.pending_figure is not None
