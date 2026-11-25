from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Flag near-constant terms in the global DTM by variance.

        Uses strictly greater than threshold semantics (variance > threshold).

        Args:
            threshold: Minimum population variance threshold (default 0.0). Terms with variance <= threshold are flagged as removed.

        Returns:
            Dict containing result_id, threshold, initial_terms, kept_count, removed_count, kept_terms (capped at 10), and removed_terms (capped at 10).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available. Call build_dtm_tool first."}

        dtm = session.feature_matrix
        feature_names = session.feature_names

        # Compute population variance (ddof=0) for each column
        # Convert sparse matrix column operations to 1D array
        means = np.asarray(dtm.mean(axis=0)).ravel()
        sq_means = np.asarray(dtm.power(2).mean(axis=0)).ravel()
        variances = sq_means - (means ** 2)
        # Ensure non-negative due to float precision
        variances = np.maximum(0.0, variances)

        # Build DataFrame with term as column 1, variance as column 2
        # Sort descending by variance, then ascending by term for ties
        df_var = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by=["variance", "term"], ascending=[False, True]).reset_index(drop=True)

        initial_terms = len(df_var)

        # Strictly greater than threshold per VarianceThreshold semantics (variance > threshold)
        kept_df = df_var[df_var["variance"] > threshold]
        removed_df = df_var[df_var["variance"] <= threshold]

        kept_count = len(kept_df)
        removed_count = len(removed_df)

        # Cap returned summaries at 10 items
        top_10_kept = kept_df.head(10)
        kept_terms = {row["term"]: round(float(row["variance"]), 4) for _, row in top_10_kept.iterrows()}
        removed_terms = removed_df["term"].head(10).tolist()

        res_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": res_id,
            "threshold": threshold,
            "initial_terms": initial_terms,
            "kept_count": kept_count,
            "removed_count": removed_count,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms
        }

        # Store full_report with term as 1st col, variance as 2nd col sorted descending for visualize_result_tool
        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=df_var)
        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Correlate term counts with target_class using Pearson r.

        Args:
            target_class: Category name for one-vs-rest binary indicator.
            threshold: Minimum absolute correlation threshold |r| >= threshold (default 0.0).

        Returns:
            Dict containing result_id, target_class, threshold, initial_terms, kept_count, removed_count, kept_terms (capped at 10), and removed_terms (capped at 10).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available. Call build_dtm_tool first."}

        if session.labels is None or session.categories is None:
            return {"error": "No document labels available in session."}

        if target_class not in session.categories:
            return {"error": f"Target class '{target_class}' not found in dataset categories: {session.categories}"}

        dtm = session.feature_matrix
        feature_names = session.feature_names
        n_samples = dtm.shape[0]

        if n_samples == 0:
            return {"error": "Document-term matrix is empty."}

        # One-vs-rest binary target indicator y (length n_samples)
        y = (session.labels == target_class).astype(float)
        
        # Vectorized population stats across all terms
        # sum_x: 1D array of column sums across all documents
        sum_x = np.asarray(dtm.sum(axis=0)).ravel().astype(float)
        # sum_x2: 1D array of column sum of squares across all documents
        sum_x2 = np.asarray(dtm.power(2).sum(axis=0)).ravel().astype(float)
        # sum_xy: matrix-vector product dtm.T @ y (sum of x * y for each term)
        sum_xy = np.asarray(dtm.T.dot(y)).ravel().astype(float)

        sum_y = float(np.sum(y))
        sum_y2 = float(np.sum(y ** 2))

        # Population covariances and standard deviations
        mean_x = sum_x / n_samples
        mean_y = sum_y / n_samples

        var_x = (sum_x2 / n_samples) - (mean_x ** 2)
        var_x = np.maximum(0.0, var_x)
        std_x = np.sqrt(var_x)

        var_y = (sum_y2 / n_samples) - (mean_y ** 2)
        var_y = max(0.0, var_y)
        std_y = np.sqrt(var_y)

        cov_xy = (sum_xy / n_samples) - (mean_x * mean_y)

        # Handle zero variance vectorized: where std_x == 0 or std_y == 0, r = 0.0
        denom = std_x * std_y
        with np.errstate(divide='ignore', invalid='ignore'):
            r_array = np.where((std_x == 0) | (std_y == 0) | (denom == 0), 0.0, cov_xy / denom)
            r_array = np.nan_to_num(r_array, nan=0.0)

        # Build DataFrame
        df_pearson = pd.DataFrame({
            "term": feature_names,
            "pearson_r": r_array
        })

        # Rank by absolute |r| descending, keeping sign. Ties broken by term ascending
        df_pearson["abs_r"] = df_pearson["pearson_r"].abs()
        df_pearson = df_pearson.sort_values(by=["abs_r", "term"], ascending=[False, True]).reset_index(drop=True)

        initial_terms = len(df_pearson)

        # Filter: keep terms with |r| >= threshold
        kept_df = df_pearson[df_pearson["abs_r"] >= threshold]
        removed_df = df_pearson[df_pearson["abs_r"] < threshold]

        kept_count = len(kept_df)
        removed_count = len(removed_df)

        top_10_kept = kept_df.head(10)
        kept_terms = {row["term"]: round(float(row["pearson_r"]), 4) for _, row in top_10_kept.iterrows()}
        removed_terms = removed_df["term"].head(10).tolist()

        # Clean up temporary abs_r column for full_report DataFrame
        full_report_df = df_pearson[["term", "pearson_r"]].copy()

        res_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": res_id,
            "target_class": target_class,
            "threshold": threshold,
            "initial_terms": initial_terms,
            "kept_count": kept_count,
            "removed_count": removed_count,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms
        }

        session.store_result("pearson_filter_tool", {"target_class": target_class, "threshold": threshold}, summary, full_report=full_report_df)
        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0, max_terms: int = 1500) -> Dict[str, Any]:
        """Correlate term counts with target_class using Spearman rank correlation.

        Args:
            target_class: Category name for one-vs-rest binary indicator.
            threshold: Minimum absolute correlation threshold |r_s| >= threshold (default 0.0).
            max_terms: Maximum number of highest-variance terms to pre-select before correlation (default 1500).

        Returns:
            Dict containing result_id, target_class, threshold, max_terms, initial_terms, evaluated_terms, kept_count, removed_count, kept_terms (capped at 10), and removed_terms (capped at 10).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available. Call build_dtm_tool first."}

        if session.labels is None or session.categories is None:
            return {"error": "No document labels available in session."}

        if target_class not in session.categories:
            return {"error": f"Target class '{target_class}' not found in dataset categories: {session.categories}"}

        dtm = session.feature_matrix
        feature_names = session.feature_names
        n_samples = dtm.shape[0]

        if n_samples == 0:
            return {"error": "Document-term matrix is empty."}

        initial_terms = len(feature_names)

        # Pre-filter to max_terms highest variance terms if vocabulary exceeds max_terms
        if initial_terms > max_terms:
            means = np.asarray(dtm.mean(axis=0)).ravel()
            sq_means = np.asarray(dtm.power(2).mean(axis=0)).ravel()
            variances = np.maximum(0.0, sq_means - (means ** 2))
            top_indices = np.argsort(-variances)[:max_terms]
            # Keep indices in original column order or sorted order
            top_indices = np.sort(top_indices)
            eval_dtm = dtm[:, top_indices]
            eval_feature_names = [feature_names[i] for i in top_indices]
        else:
            eval_dtm = dtm
            eval_feature_names = feature_names

        evaluated_terms = len(eval_feature_names)

        # Target indicator y (length n_samples) and rank y
        y = (session.labels == target_class).astype(float)
        rank_y = rankdata(y, method="average")

        # Densify eval_dtm for column-wise ranking
        dense_dtm = eval_dtm.toarray()

        # Compute rank correlation across evaluated terms
        # Compute rank_x for each column in dense_dtm using average ranks for ties
        rank_x_mat = np.apply_along_axis(lambda col: rankdata(col, method="average"), axis=0, arr=dense_dtm)

        # Calculate Pearson correlation on rank_x_mat and rank_y (which yields Spearman r)
        sum_x = np.sum(rank_x_mat, axis=0)
        sum_x2 = np.sum(rank_x_mat ** 2, axis=0)
        sum_xy = np.dot(rank_x_mat.T, rank_y)

        sum_y = float(np.sum(rank_y))
        sum_y2 = float(np.sum(rank_y ** 2))

        mean_x = sum_x / n_samples
        mean_y = sum_y / n_samples

        var_x = (sum_x2 / n_samples) - (mean_x ** 2)
        var_x = np.maximum(0.0, var_x)
        std_x = np.sqrt(var_x)

        var_y = (sum_y2 / n_samples) - (mean_y ** 2)
        var_y = max(0.0, var_y)
        std_y = np.sqrt(var_y)

        cov_xy = (sum_xy / n_samples) - (mean_x * mean_y)

        denom = std_x * std_y
        with np.errstate(divide='ignore', invalid='ignore'):
            r_array = np.where((std_x == 0) | (std_y == 0) | (denom == 0), 0.0, cov_xy / denom)
            r_array = np.nan_to_num(r_array, nan=0.0)

        df_spearman = pd.DataFrame({
            "term": eval_feature_names,
            "spearman_r": r_array
        })

        df_spearman["abs_r"] = df_spearman["spearman_r"].abs()
        df_spearman = df_spearman.sort_values(by=["abs_r", "term"], ascending=[False, True]).reset_index(drop=True)

        kept_df = df_spearman[df_spearman["abs_r"] >= threshold]
        removed_df = df_spearman[df_spearman["abs_r"] < threshold]

        kept_count = len(kept_df)
        removed_count = len(removed_df)

        top_10_kept = kept_df.head(10)
        kept_terms = {row["term"]: round(float(row["spearman_r"]), 4) for _, row in top_10_kept.iterrows()}
        removed_terms = removed_df["term"].head(10).tolist()

        full_report_df = df_spearman[["term", "spearman_r"]].copy()

        res_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": res_id,
            "target_class": target_class,
            "threshold": threshold,
            "max_terms": max_terms,
            "initial_terms": initial_terms,
            "evaluated_terms": evaluated_terms,
            "kept_count": kept_count,
            "removed_count": removed_count,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms
        }

        session.store_result("spearman_filter_tool", {"target_class": target_class, "threshold": threshold, "max_terms": max_terms}, summary, full_report=full_report_df)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
