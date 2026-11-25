from typing import Dict, Any, List, Optional, Literal
from langchain_core.tools import tool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def make_tools(session):
    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encode the category labels in session.dataframe into a label matrix.

        Returns:
            Dict containing result_id, classes (list of str), matrix_shape (list of int),
            counts_per_class (dict), and first_3_rows (list of list of int).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Call load_dataset_tool first."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "DataFrame missing 'category_name' column."}

        # One-hot encode category_name ensuring one column per class
        encoded_df = pd.get_dummies(session.dataframe["category_name"], dtype=int)
        
        # Sort class column names deterministically
        sorted_classes = sorted(encoded_df.columns.tolist())
        encoded_df = encoded_df[sorted_classes]

        label_matrix = encoded_df.to_numpy()
        session.artifacts["label_matrix"] = label_matrix

        counts_per_class = {str(col): int(encoded_df[col].sum()) for col in sorted_classes}
        first_3_rows = [[int(val) for val in row] for row in label_matrix[:3]]

        res_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": res_id,
            "classes": [str(c) for c in sorted_classes],
            "matrix_shape": list(label_matrix.shape),
            "counts_per_class": counts_per_class,
            "first_3_rows": first_3_rows
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> Dict[str, Any]:
        """Reduce the dimensionality of session.feature_matrix using PCA, t-SNE, or UMAP.

        Args:
            method: Dimensionality reduction technique ("pca", "tsne", or "umap").
            n_components: Number of target dimensions (default 2).
            random_state: Random seed for reproducible output (default 42).
            perplexity: Perplexity parameter for t-SNE (must be < n_documents).
            n_neighbors: Number of neighbors parameter for UMAP (must be < n_documents).

        Returns:
            Dict containing result_id, method, shape, explained_variance_ratio (PCA only),
            first_10_rows (rounded to 4 decimal places), and truncated flag.
        """
        if session.feature_matrix is None:
            return {"error": "Document-term matrix (feature_matrix) has not been built yet. Call build_dtm_tool first."}

        if n_components < 2:
            return {"error": f"n_components ({n_components}) must be at least 2 for 2D visualization."}

        n_samples = session.feature_matrix.shape[0]

        # Parameter validation for method-specific sample count requirements
        if method == "tsne" and perplexity >= n_samples:
            return {"error": f"Perplexity ({perplexity}) must be strictly less than the number of documents ({n_samples})."}
        if method == "umap" and n_neighbors >= n_samples:
            return {"error": f"n_neighbors ({n_neighbors}) must be strictly less than the number of documents ({n_samples})."}

        # Convert sparse matrix to dense array inside function body
        X_dense = session.feature_matrix.toarray()

        explained_variance_ratio = None

        if method == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X_dense)
            explained_variance_ratio = [float(v) for v in model.explained_variance_ratio_]
        elif method == "tsne":
            from sklearn.manifold import TSNE
            model = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
            coords = model.fit_transform(X_dense)
        elif method == "umap":
            import umap
            model = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=n_neighbors)
            coords = model.fit_transform(X_dense)
        else:
            return {"error": f"Invalid method '{method}'. Must be one of 'pca', 'tsne', or 'umap'."}

        # Save reduced coordinates to session artifacts
        session.artifacts["reduced_coordinates"] = coords

        # Generate scatter plot colored by session.labels (numpy array aligned with matrix rows)
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=session.labels,
            ax=ax,
            palette="deep"
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.tight_layout()

        # Attach figure to session state and close figure locally
        session.pending_figure = fig
        plt.close(fig)

        # Prepare summary results
        first_10 = [[round(float(val), 4) for val in row[:n_components]] for row in coords[:10]]
        truncated = len(coords) > 10

        res_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": res_id,
            "method": method,
            "shape": list(coords.shape),
            "explained_variance_ratio": explained_variance_ratio,
            "first_10_rows": first_10,
            "truncated": truncated
        }

        full_report = {
            "coordinates": coords.tolist(),
            "method": method
        }

        session.store_result("reduce_dimensions_tool", {
            "method": method,
            "n_components": n_components,
            "random_state": random_state,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors
        }, summary, full_report=full_report)

        return summary

    return [binarize_labels_tool, reduce_dimensions_tool]
