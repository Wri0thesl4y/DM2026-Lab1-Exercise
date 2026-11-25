import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_variance_filter_tool_known_answer_5():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    
    # Run with threshold 0.15 per Known Answer #5
    result = variance_filter_tool.invoke({"threshold": 0.15})
    
    assert result["initial_terms"] == 5
    assert result["kept_count"] == 4
    assert result["removed_count"] == 1
    assert result["removed_terms"] == ["always"]
    
    # Check kept_terms keys and rounded variance values in order
    expected_kept_keys = ["alpha", "gamma", "delta", "beta"]
    assert list(result["kept_terms"].keys()) == expected_kept_keys
    assert result["kept_terms"]["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert result["kept_terms"]["gamma"] == pytest.approx(1.1094, abs=1e-3)
    assert result["kept_terms"]["delta"] == pytest.approx(0.2500, abs=1e-3)
    assert result["kept_terms"]["beta"] == pytest.approx(0.2344, abs=1e-3)
    
    # Boundary check with threshold 0.25: delta has variance 0.25 exactly.
    # With strictly greater semantics (variance > threshold), delta (0.25) must be removed.
    result_boundary = variance_filter_tool.invoke({"threshold": 0.25})
    assert result_boundary["kept_count"] == 2
    assert list(result_boundary["kept_terms"].keys()) == ["alpha", "gamma"]
    assert "delta" in result_boundary["removed_terms"]

def test_pearson_filter_tool_known_answer_6():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Test unknown target class error handling
    err_res = pearson_filter_tool.invoke({"target_class": "nonexistent_class", "threshold": 0.5})
    assert "error" in err_res

    # Run with target_class="catB" and threshold 0.5
    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    assert result["initial_terms"] == 5
    assert result["kept_count"] == 4
    assert result["removed_count"] == 1
    assert "always" in result["removed_terms"]

    # Verify kept terms are ordered by absolute correlation |r| descending
    expected_order = ["delta", "gamma", "beta", "alpha"]
    assert list(result["kept_terms"].keys()) == expected_order

    # Check expected Pearson r values matching Known Answer #6
    assert result["kept_terms"]["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert result["kept_terms"]["gamma"] == pytest.approx(0.8307, abs=1e-3)
    assert result["kept_terms"]["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert result["kept_terms"]["alpha"] == pytest.approx(-0.6882, abs=1e-3)

def test_spearman_filter_tool_known_answer_6():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Test unknown target class error handling
    err_res = spearman_filter_tool.invoke({"target_class": "nonexistent_class", "threshold": 0.5})
    assert "error" in err_res

    # Run with target_class="catB" and threshold 0.5
    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    assert result["initial_terms"] == 5
    assert result["evaluated_terms"] == 5
    assert result["kept_count"] == 4
    assert result["removed_count"] == 1
    assert "always" in result["removed_terms"]

    # Verify kept terms are ordered by absolute Spearman correlation |r_s| descending
    expected_order = ["delta", "gamma", "beta", "alpha"]
    assert list(result["kept_terms"].keys()) == expected_order

    # Check expected Spearman r values matching Known Answer #6
    assert result["kept_terms"]["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert result["kept_terms"]["gamma"] == pytest.approx(0.9363, abs=1e-3)
    assert result["kept_terms"]["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert result["kept_terms"]["alpha"] == pytest.approx(-0.7500, abs=1e-3)

    # Test max_terms=2 pre-selection (top 2 variance terms on fixture are alpha and gamma)
    result_max2 = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0, "max_terms": 2})

    assert result_max2["initial_terms"] == 5
    assert result_max2["evaluated_terms"] == 2
    assert result_max2["kept_count"] == 2
    assert list(result_max2["kept_terms"].keys()) == ["gamma", "alpha"]
    assert result_max2["kept_terms"]["gamma"] == pytest.approx(0.9363, abs=1e-3)
    assert result_max2["kept_terms"]["alpha"] == pytest.approx(-0.7500, abs=1e-3)
