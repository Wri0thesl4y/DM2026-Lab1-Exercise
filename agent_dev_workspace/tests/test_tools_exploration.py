import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def test_cosine_similarity_tool_known_answer_16():
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

    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Test out-of-range index validation
    err_high = cosine_similarity_tool.invoke({"doc_a": 0, "doc_b": 8})
    assert "error" in err_high

    err_low = cosine_similarity_tool.invoke({"doc_a": -1, "doc_b": 3})
    assert "error" in err_low

    # Known Answer #16 assertions
    # Pair 0 vs 1 (both catA): 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc_a": 0, "doc_b": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)
    assert res_0_1["category_a"] == "catA"
    assert res_0_1["category_b"] == "catA"

    # Pair 0 vs 3 (catA vs catB, share only "always"): 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc_a": 0, "doc_b": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)
    assert res_0_3["category_a"] == "catA"
    assert res_0_3["category_b"] == "catB"

    # Pair 3 vs 4 (both catB): 0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc_a": 3, "doc_b": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Pair 0 vs 7 (catA vs empty text row 7): 0.0
    res_0_7 = cosine_similarity_tool.invoke({"doc_a": 0, "doc_b": 7})
    assert res_0_7["similarity"] == 0.0


def test_feature_correlation_matrix_tool_known_answer_19():
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

    exploration_tools = make_exploration_tools(session)
    feature_correlation_matrix_tool = next(
        t for t in exploration_tools if t.name == "feature_correlation_matrix_tool"
    )

    res = feature_correlation_matrix_tool.invoke({})

    # Terms ordering check
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    # Correlation matrix check
    expected_matrix = [
        [1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718, 1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882, 0.8307,  1.0000, -0.7746,  0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [ 0.2601,  0.3140,  0.3780,  0.2928,  1.0000]
    ]

    corr = res["correlation_matrix"]
    assert len(corr) == 5
    for i in range(5):
        assert len(corr[i]) == 5
        # Exact 1.0 on diagonal
        assert corr[i][i] == 1.0
        for j in range(5):
            # Symmetry check
            assert corr[i][j] == corr[j][i]
            # Approx value check against known answer
            assert corr[i][j] == pytest.approx(expected_matrix[i][j], abs=1e-3)

    # Plot figure check
    assert session.pending_figure is not None
