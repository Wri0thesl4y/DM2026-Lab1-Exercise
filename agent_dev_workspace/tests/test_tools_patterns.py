import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools

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
    return session

def test_mine_patterns_fpgrowth_known_answer_17():
    session = setup_fixture_session()
    tools = make_patterns_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # catA with tfidf & fpgrowth: min_sup=1 -> pattern ['alpha'] with support 3
    res_a_1 = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_a_1["n_patterns"] == 1
    assert res_a_1["top_10_patterns"][0]["pattern"] == ["alpha"]
    assert res_a_1["top_10_patterns"][0]["support"] == 3

    # catA with min_sup=3 -> pattern ['alpha'] with support 3
    res_a_3 = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3
    })
    assert res_a_3["n_patterns"] == 1
    assert res_a_3["top_10_patterns"][0]["pattern"] == ["alpha"]
    assert res_a_3["top_10_patterns"][0]["support"] == 3

    # catA with min_sup=4 -> 0 patterns
    res_a_4 = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_a_4["n_patterns"] == 0

    # catB with tfidf & fpgrowth: min_sup=4 -> pattern ['gamma'] with support 4
    res_b_4 = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_b_4["n_patterns"] == 1
    assert res_b_4["top_10_patterns"][0]["pattern"] == ["gamma"]
    assert res_b_4["top_10_patterns"][0]["support"] == 4

def test_mine_patterns_topk_and_maxfpgrowth_known_answer_20():
    session = setup_fixture_session()
    tools = make_patterns_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Top-K tests for k in [1, 2, 3]
    for k in [1, 2, 3]:
        res_a_topk = mine_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k
        })
        assert res_a_topk["n_patterns"] == 1
        assert res_a_topk["top_10_patterns"][0]["pattern"] == ["alpha"]
        assert res_a_topk["top_10_patterns"][0]["support"] == 3

        res_b_topk = mine_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k
        })
        assert res_b_topk["n_patterns"] == 1
        assert res_b_topk["top_10_patterns"][0]["pattern"] == ["gamma"]
        assert res_b_topk["top_10_patterns"][0]["support"] == 4

    # MaxFPGrowth tests for min_sup in [1, 4] -> returns 0 patterns
    for min_sup in [1, 4]:
        res_a_max = mine_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": min_sup
        })
        assert res_a_max["n_patterns"] == 0

        res_b_max = mine_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": min_sup
        })
        assert res_b_max["n_patterns"] == 0

def test_mine_patterns_zero_terms_survived():
    session = setup_fixture_session()
    tools = make_patterns_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    for method in ["variance", "term_frequency"]:
        for cat in ["catA", "catB"]:
            res = mine_tool.invoke({
                "category_name": cat,
                "filtering_method": method,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            assert res["n_patterns"] == 0
            assert res["note"] == "no terms survived filtering"

def test_mine_patterns_error_cases():
    session = setup_fixture_session()
    tools = make_patterns_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Unknown category
    res_unknown_cat = mine_tool.invoke({
        "category_name": "unknown_cat",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" in res_unknown_cat

    # Missing min_sup for fpgrowth
    res_missing_minsup = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth"
    })
    assert "error" in res_missing_minsup

    # Missing k for topk
    res_missing_k = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk"
    })
    assert "error" in res_missing_k


def test_mine_patterns_handles_non_ascii_terms():
    """Real posts contain accented words; PAMI's file I/O used to crash on them on Windows."""
    import pandas as pd
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["caf\u00e9 \u00edndice plan", "caf\u00e9 \u00edndice plan", "caf\u00e9 \u00edndice plan", "zebra yacht xylo"],
        "category_name": ["catA", "catA", "catA", "catB"],
        "category": [0, 0, 0, 1],
    })
    tools = make_patterns_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")
    res = mine_tool.invoke({
        "category_name": "catA", "filtering_method": "term_frequency",
        "algorithm": "fpgrowth", "min_sup": 3,
    })
    assert "error" not in res
    assert res["n_patterns"] > 0
    flat = {term for p in res["top_10_patterns"] for term in p["pattern"]}
    assert "caf\u00e9" in flat or "\u00edndice" in flat
