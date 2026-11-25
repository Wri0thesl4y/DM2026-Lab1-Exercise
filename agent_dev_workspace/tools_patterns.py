import os
import tempfile
from typing import Dict, Any, List, Optional, Literal
from langchain_core.tools import tool
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mine frequent patterns for a specific category using PAMI algorithms.

        Args:
            category_name: Target category label to mine.
            filtering_method: Vocabulary filter method ("variance", "tfidf", or "term_frequency").
            algorithm: PAMI mining algorithm ("fpgrowth", "topk", or "maxfpgrowth").
            min_sup: Minimum support threshold (required for "fpgrowth" and "maxfpgrowth").
            k: Top-K pattern limit (required for "topk").

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm,
            n_patterns, top_10_patterns, and note.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Call load_dataset_tool first."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "DataFrame missing 'category_name' column."}

        # Validate category_name existence
        valid_categories = session.dataframe["category_name"].unique().tolist()
        if category_name not in valid_categories:
            return {"error": f"Category '{category_name}' not found. Valid categories: {valid_categories}"}

        # Validate method and algorithm inputs
        if filtering_method not in ["variance", "tfidf", "term_frequency"]:
            return {"error": f"Invalid filtering_method '{filtering_method}'. Must be 'variance', 'tfidf', or 'term_frequency'."}

        if algorithm not in ["fpgrowth", "topk", "maxfpgrowth"]:
            return {"error": f"Invalid algorithm '{algorithm}'. Must be 'fpgrowth', 'topk', or 'maxfpgrowth'."}

        # Validate algorithm parameter requirements
        if algorithm in ["fpgrowth", "maxfpgrowth"] and min_sup is None:
            return {"error": f"Parameter 'min_sup' is required when using algorithm '{algorithm}'."}

        if algorithm == "topk" and k is None:
            return {"error": "Parameter 'k' is required when using algorithm 'topk'."}

        # Step 1: Filter DataFrame to category and build per-category DTM
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        texts = cat_df["text"].dropna().tolist()

        if not texts:
            return {"error": f"No text content found for category '{category_name}'."}

        vectorizer = CountVectorizer(stop_words="english")
        dtm = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

        if dtm.shape[1] == 0:
            res_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "top_10_patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm
            }, summary)
            return summary

        dtm_dense = dtm.toarray()

        # Step 2: Calculate filtering metric across vocabulary columns
        if filtering_method == "variance":
            metrics = np.var(dtm_dense, axis=0)
            low_p, high_p = np.percentile(metrics, 5), np.percentile(metrics, 95)
            kept_mask = (metrics >= low_p) & (metrics <= high_p)
        elif filtering_method == "term_frequency":
            metrics = np.sum(dtm_dense, axis=0)
            low_p, high_p = np.percentile(metrics, 5), np.percentile(metrics, 95)
            kept_mask = (metrics >= low_p) & (metrics <= high_p)
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_mat = transformer.fit_transform(dtm_dense).toarray()
            metrics = np.mean(tfidf_mat, axis=0)
            low_p = np.percentile(metrics, 20)
            kept_mask = metrics >= low_p

        kept_terms = feature_names[kept_mask]
        kept_matrix = dtm_dense[:, kept_mask]

        # Step 3: Handle zero terms surviving filtering
        if len(kept_terms) == 0:
            res_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "top_10_patterns": [],
                "note": "no terms survived filtering"
            }
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm
            }, summary)
            return summary

        # Step 4: Convert count matrix DataFrame to PAMI Transactional DB via temp file
        # PAMI reads/writes its files with the platform default encoding, so a
        # term with non-ASCII characters (e.g. an accented letter) crashes it
        # on Windows. Give it plain ASCII item ids and map them back afterwards.
        id_to_term = {f"t{i}": str(term) for i, term in enumerate(kept_terms)}
        count_df = pd.DataFrame(kept_matrix, columns=list(id_to_term.keys()))

        patterns_list = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = os.path.join(tmp_dir, "transactional_db.txt")

            from PAMI.extras.convert import DF2DB
            converter = DF2DB.DF2DB(count_df, DFtype="dense")
            converter.convert2TransactionalDatabase(db_file, ">=", 1)

            out_file = os.path.join(tmp_dir, "output_patterns.txt")

            # Run requested PAMI mining algorithm
            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                miner = FPGrowth.FPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
                miner.mine()
                miner.save(out_file)
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                miner = FAE.FAE(iFile=db_file, k=k, sep="\t")
                miner.mine()
                miner.save(out_file)
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                miner = MaxFPGrowth.MaxFPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
                miner.mine()
                miner.save(out_file)

            # Step 5: Read and parse PAMI output file
            if os.path.exists(out_file):
                with open(out_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # PAMI pattern output lines are formatted like: item1 item2 ... : support
                        if ":" in line:
                            pattern_str, sup_str = line.rsplit(":", 1)
                            items = [item.strip() for item in pattern_str.strip().split("\t") if item.strip()]
                            if not items:
                                items = [item.strip() for item in pattern_str.strip().split() if item.strip()]
                            try:
                                sup = int(float(sup_str.strip()))
                            except ValueError:
                                sup = 0
                            items = [id_to_term.get(item, item) for item in items]
                            patterns_list.append({
                                "pattern": items,
                                "support": sup
                            })

        # Sort patterns descending by support
        patterns_list.sort(key=lambda x: x["support"], reverse=True)

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        top_10 = patterns_list[:10]
        res_id = session.next_result_id("mine_patterns")

        summary = {
            "result_id": res_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns_list),
            "top_10_patterns": top_10,
            "note": "success" if patterns_list else "no patterns found"
        }

        session.store_result("mine_patterns_tool", {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k
        }, summary, full_report={"patterns": patterns_list})

        return summary

    return [mine_patterns_tool]
