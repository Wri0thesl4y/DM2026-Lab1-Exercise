from typing import Dict, Any, List, Optional, Union
from langchain_core.tools import tool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: Optional[List[int]] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0
    ) -> Dict[str, Any]:
        """Build the document-term matrix (DTM) from document text.

        Stores the count vectorizer in session.artifacts['count_vectorizer'],
        the matrix in session.feature_matrix, vocabulary in session.feature_names,
        and per-document labels in session.labels (via session.set_labels).

        Args:
            ngram_range: List of two ints [min_n, max_n] for n-gram bounds (default [1, 1]).
            min_df: Minimum document frequency threshold (count as int or fraction as float, default 1).
            max_df: Maximum document frequency threshold (count as int or fraction as float, default 1.0).

        Returns:
            Dict containing result_id, n_documents, vocabulary_size, non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Call load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"error": "Dataset does not contain a 'text' column."}

        if ngram_range is None:
            ngram_range = [1, 1]

        # Validate ngram_range: must be list/sequence of 2 ints
        if not isinstance(ngram_range, (list, tuple)) or len(ngram_range) != 2:
            return {"error": "'ngram_range' must be a list of 2 integers, e.g. [1, 1]."}

        tuple_ngram_range = (int(ngram_range[0]), int(ngram_range[1]))

        # Initialize CountVectorizer with vectorizer parameters matching sklearn defaults
        vectorizer = CountVectorizer(
            ngram_range=tuple_ngram_range,
            min_df=min_df,
            max_df=max_df
        )

        texts = session.dataframe["text"].fillna("").astype(str).tolist()
        
        try:
            dtm = vectorizer.fit_transform(texts)
        except ValueError as e:
            return {"error": f"Failed to build DTM: {str(e)}"}

        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state per contract
        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Set labels and update session.categories if category_name is present
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        n_documents, vocabulary_size = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = n_documents * vocabulary_size

        if total_elements > 0:
            sparsity_pct = round(100.0 * (1.0 - non_zero / total_elements), 4)
        else:
            sparsity_pct = 0.0

        res_id = session.next_result_id("dtm")
        summary = {
            "result_id": res_id,
            "n_documents": n_documents,
            "vocabulary_size": vocabulary_size,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct
        }

        session.store_result("build_dtm_tool", {"ngram_range": ngram_range, "min_df": min_df, "max_df": max_df}, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Aggregate term frequencies across all documents in the global DTM.

        Args:
            top_n: Number of top frequent terms to return in the summary (default 20).

        Returns:
            Dict containing result_id, vocabulary_size, and term_frequencies.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available. Call build_dtm_tool first."}

        # Sum column counts across all documents (1D array)
        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        
        # Build DataFrame with term as first column, total_frequency as second column
        # Deterministic sort: frequency descending, then term ascending for ties
        df_tf = pd.DataFrame({
            "term": session.feature_names,
            "frequency": counts
        }).sort_values(by=["frequency", "term"], ascending=[False, True]).reset_index(drop=True)

        vocabulary_size = len(df_tf)
        top_n_effective = min(max(1, top_n), vocabulary_size) if vocabulary_size > 0 else 0
        
        top_terms_df = df_tf.head(top_n_effective)
        term_frequencies = dict(zip(top_terms_df["term"], top_terms_df["frequency"].astype(int)))

        res_id = session.next_result_id("tf")
        summary = {
            "result_id": res_id,
            "vocabulary_size": vocabulary_size,
            "term_frequencies": term_frequencies
        }

        # Store full_report DataFrame sorted descending for visualize_result_tool
        session.store_result("term_frequency_tool", {"top_n": top_n}, summary, full_report=df_tf)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generate a heatmap visualization of a slice of the raw Document-Term Matrix (DTM).

        Args:
            n_terms: Number of terms (columns) to slice from DTM in positional order (default 20, capped at 50).
            n_documents: Number of documents (rows) to slice from DTM in positional order (default 20, capped at 50).

        Returns:
            Dict containing result_id, matrix_slice (nested list of ints), terms, document_indices, n_terms, and n_documents.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix available. Call build_dtm_tool first."}

        total_docs, total_terms = session.feature_matrix.shape

        # Positional slicing: cap n_terms and n_documents at 50 to prevent massive summaries
        n_docs_slice = min(max(1, n_documents), total_docs, 50) if total_docs > 0 else 0
        n_terms_slice = min(max(1, n_terms), total_terms, 50) if total_terms > 0 else 0

        # Slice matrix positionally without sorting
        sliced_matrix_sparse = session.feature_matrix[0:n_docs_slice, 0:n_terms_slice]
        matrix_dense = sliced_matrix_sparse.toarray().astype(int)

        terms_sliced = session.feature_names[0:n_terms_slice]
        doc_indices = list(range(n_docs_slice))

        # Build annotated heatmap using seaborn
        fig, ax = plt.subplots(figsize=(max(6, n_terms_slice * 0.8), max(4, n_docs_slice * 0.5)))
        sns.heatmap(
            matrix_dense,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=terms_sliced,
            yticklabels=doc_indices,
            ax=ax
        )
        ax.set_title("Document-Term Matrix (DTM) Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        # Convert dense numpy matrix to plain python nested list of ints for JSON safety
        matrix_slice_list = matrix_dense.tolist()

        res_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": res_id,
            "matrix_slice": matrix_slice_list,
            "terms": terms_sliced,
            "document_indices": doc_indices,
            "n_terms": n_terms_slice,
            "n_documents": n_docs_slice
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
