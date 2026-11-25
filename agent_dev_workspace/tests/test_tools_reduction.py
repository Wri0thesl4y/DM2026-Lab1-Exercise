import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools as make_reduction_tools

def test_binarize_labels_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_labels_tool.invoke({})

    # Check error when no dataset loaded in a separate session
    empty_session = SessionState()
    empty_binarize = next(t for t in make_reduction_tools(empty_session) if t.name == "binarize_labels_tool")
    assert "error" in empty_binarize.invoke({})

    # Assertions on sample_fixture.csv
    assert res["classes"] == ["catA", "catB"]
    assert res["matrix_shape"] == [8, 2]
    assert res["counts_per_class"] == {"catA": 4, "catB": 4}
    assert res["first_3_rows"] == [[1, 0], [1, 0], [1, 0]]

    # Artifact check
    assert "label_matrix" in session.artifacts
    matrix = session.artifacts["label_matrix"]
    assert matrix.shape == (8, 2)
    assert np.all(matrix.sum(axis=1) == 1)
    assert list(matrix[3]) == [0, 1]

def setup_fixture_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample_fixture.csv
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM using build_dtm_tool from tools_dtm
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session

def test_reduce_dimensions_pca_known_answer_7():
    session = setup_fixture_session()
    tools = make_reduction_tools(session)
    reduce_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({"method": "pca", "n_components": 2})
    
    assert res["method"] == "pca"
    assert res["shape"] == [8, 2]
    assert session.pending_figure is not None
    
    # Check explained_variance_ratio [0.7532, 0.1965]
    assert res["explained_variance_ratio"] is not None
    assert np.allclose(res["explained_variance_ratio"], [0.7532, 0.1965], atol=1e-3)
    
    # Known coordinates for PCA (TEST_FIXTURE #7)
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    
    actual_coords = np.array(session.artifacts["reduced_coordinates"])
    assert np.allclose(actual_coords, expected_coords, atol=1e-3)

def test_reduce_dimensions_tsne_known_answer_8():
    session1 = setup_fixture_session()
    tools1 = make_reduction_tools(session1)
    reduce_tool1 = next(t for t in tools1 if t.name == "reduce_dimensions_tool")
    
    # Reference coordinates in TEST_FIXTURE.md #8 are platform/sklearn-version dependent and not directly reproducible here.
    res1 = reduce_tool1.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    
    assert res1["method"] == "tsne"
    assert res1["shape"] == [8, 2]
    assert session1.pending_figure is not None
    
    coords1 = np.array(session1.artifacts["reduced_coordinates"])
    
    # Check duplicate rows 5 and 6 produce identical coordinates
    assert np.allclose(coords1[5], coords1[6], atol=1e-3)
    
    # Check repeatability across runs with the same random_state
    session2 = setup_fixture_session()
    tools2 = make_reduction_tools(session2)
    reduce_tool2 = next(t for t in tools2 if t.name == "reduce_dimensions_tool")
    res2 = reduce_tool2.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    coords2 = np.array(session2.artifacts["reduced_coordinates"])
    
    assert np.allclose(coords1, coords2, atol=1e-3)

def test_reduce_dimensions_umap_known_answer_9():
    session = setup_fixture_session()
    tools = make_reduction_tools(session)
    reduce_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    
    assert res["method"] == "umap"
    assert res["shape"] == [8, 2]
    assert session.pending_figure is not None
    
    # Known coordinates for UMAP (TEST_FIXTURE #9)
    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])
    
    actual_coords = np.array(session.artifacts["reduced_coordinates"])
    assert np.allclose(actual_coords, expected_coords, atol=1e-2)

def test_reduce_dimensions_errors():
    session = setup_fixture_session()
    tools = make_reduction_tools(session)
    reduce_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")
    
    # Test error when perplexity >= n_samples (8)
    res_tsne_err = reduce_tool.invoke({"method": "tsne", "perplexity": 8})
    assert "error" in res_tsne_err
    assert "Perplexity" in res_tsne_err["error"]
    
    # Test error when n_components < 2
    res_comp_err = reduce_tool.invoke({"method": "pca", "n_components": 1})
    assert "error" in res_comp_err
    assert "n_components" in res_comp_err["error"]
