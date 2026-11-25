from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_a: int, doc_b: int) -> Dict[str, Any]:
        """Compute cosine similarity between two documents by their row indices in session.feature_matrix.

        Args:
            doc_a: Row index of the first document (0-indexed).
            doc_b: Row index of the second document (0-indexed).

        Returns:
            Dict containing result_id, doc_a, doc_b, category_a, category_b, and similarity (float rounded to 4 decimals).
        """
        if session.feature_matrix is None:
            return {"error": "No document-term matrix available. Call build_dtm_tool first."}

        n_documents = session.feature_matrix.shape[0]

        if not (0 <= doc_a < n_documents):
            return {"error": f"doc_a index {doc_a} is out of bounds for matrix with {n_documents} documents."}

        if not (0 <= doc_b < n_documents):
            return {"error": f"doc_b index {doc_b} is out of bounds for matrix with {n_documents} documents."}

        # Extract 1D row vectors from sparse matrix
        vec_a = session.feature_matrix[doc_a].toarray().ravel()
        vec_b = session.feature_matrix[doc_b].toarray().ravel()

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        # Handle zero vectors (e.g. empty text rows) safely without producing NaN or errors
        if norm_a == 0.0 or norm_b == 0.0:
            similarity = 0.0
        else:
            dot_product = np.dot(vec_a, vec_b)
            similarity = round(float(dot_product / (norm_a * norm_b)), 4)

        # Get category names for doc_a and doc_b if labels are present
        category_a = None
        category_b = None
        if session.labels is not None:
            category_a = str(session.labels[doc_a])
            category_b = str(session.labels[doc_b])

        res_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": res_id,
            "doc_a": doc_a,
            "doc_b": doc_b,
            "category_a": category_a,
            "category_b": category_b,
            "similarity": similarity
        }

        session.store_result("cosine_similarity_tool", {"doc_a": doc_a, "doc_b": doc_b}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """Compute term-to-term Pearson correlation matrix for top 20 terms by variance across the DTM and plot a heatmap.

        Returns:
            Dict containing result_id, terms (list of str), and correlation_matrix (nested list of floats rounded to 4 decimals).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available. Call build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)
        n_terms = len(feature_names)

        if n_terms == 0:
            return {"error": "Feature matrix has no terms."}

        # Population variance on sparse matrix: E[X^2] - (E[X])^2
        means = np.ravel(X.mean(axis=0))
        means_sq = np.ravel(X.power(2).mean(axis=0))
        variances = np.maximum(0.0, means_sq - (means ** 2))

        # Sort by (-variance, term_name) for deterministic tie-breaking (asc term name)
        sort_keys = [(-var, name, idx) for idx, (var, name) in enumerate(zip(variances, feature_names))]
        sort_keys.sort(key=lambda x: (x[0], x[1]))

        # Take top 20 terms
        top_k = min(20, n_terms)
        top_indices = [x[2] for x in sort_keys[:top_k]]
        selected_terms = [str(feature_names[i]) for i in top_indices]

        # Convert only top columns to dense
        X_top = X[:, top_indices].toarray()

        # Compute Pearson correlation matrix
        corr = np.corrcoef(X_top, rowvar=False)
        if corr.ndim == 0:  # Single element case
            corr = np.array([[1.0]])

        # Handle NaNs (due to zero variance) and force diagonal to 1.0
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        # Build figure and heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax
        )
        ax.set_title("Feature Correlation Matrix (Top 20 Terms by Variance)")
        fig.tight_layout()
        session.pending_figure = fig
        plt.close(fig)

        # Convert matrix to nested Python float list rounded to 4 decimals
        corr_matrix_list = [
            [round(float(val), 4) for val in row]
            for row in corr
        ]

        res_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": res_id,
            "terms": selected_terms,
            "correlation_matrix": corr_matrix_list
        }

        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
