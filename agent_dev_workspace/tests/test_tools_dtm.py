import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool_known_answer_4():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    # Run build_dtm_tool with default arguments
    result = build_dtm_tool.invoke({})
    
    # Assert against Known Answer #4 in TEST_FIXTURE.md
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocab
    assert result["vocabulary_size"] == 5
    assert result["n_documents"] == 8
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)
    
    # Check that count_vectorizer artifact is stored
    assert "count_vectorizer" in session.artifacts
    assert session.artifacts["count_vectorizer"] is not None
    assert session.categories == ["catA", "catB"]
    
    # Check explicit min_df=1 argument to ensure int 1 is not converted to float 1.0
    result_min_df_1 = build_dtm_tool.invoke({"min_df": 1})
    assert result_min_df_1["vocabulary_size"] == 5
    assert session.feature_names == expected_vocab

def test_term_frequency_tool_known_answer_14():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")
    
    # Build DTM first
    build_dtm_tool.invoke({})
    
    # Run term_frequency_tool
    result = term_frequency_tool.invoke({})
    
    assert result["vocabulary_size"] == 5
    
    # Known Answer #14 total frequencies
    expected_frequencies = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3
    }
    
    assert result["term_frequencies"] == expected_frequencies
    assert list(result["term_frequencies"].keys()) == ["always", "gamma", "alpha", "delta", "beta"]

def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")
    
    # Build DTM first
    build_dtm_tool.invoke({})
    
    # Run dtm_heatmap_tool with defaults (n_terms=20, n_documents=20)
    result = dtm_heatmap_tool.invoke({})
    
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    
    # Check exact 8x5 matrix values per Known Answer #21 in TEST_FIXTURE.md
    expected_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    
    assert result["matrix_slice"] == expected_matrix
    assert session.pending_figure is not None
    
    # Check n_documents=100 still returns 8 rows (capped by actual dataset size)
    result_100 = dtm_heatmap_tool.invoke({"n_documents": 100})
    assert result_100["n_documents"] == 8
    assert result_100["matrix_slice"] == expected_matrix
