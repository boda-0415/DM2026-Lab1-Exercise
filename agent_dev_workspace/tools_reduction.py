from typing import Any, Dict, Literal, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from langchain_core.tools import tool

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def make_tools(session):
    """
    Factory function returning reduction tools bound to the given SessionState.
    """

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Reduces dimensionality of the Document-Term Matrix (PCA, t-SNE, UMAP) to 2D and generates a scatter plot.
        
        Args:
            method: Reduction method ('pca', 'tsne', or 'umap'). Default 'pca'.
            n_components: Number of target dimensions (must be 2 for visualization). Default 2.
            perplexity: Perplexity parameter for t-SNE (must satisfy 0 < perplexity < n_samples). Default 30.0.
            n_neighbors: Number of neighbors for UMAP (must satisfy 2 <= n_neighbors <= n_samples). Default 15.
            random_state: Random state seed for reproducibility. Default 42.
            
        Returns:
            Dict[str, Any]: Summary containing result_id, status, method, coordinates,
                            and explained_variance_ratio (if PCA).
        """
        if session.feature_matrix is None:
            return {"error": "No Document-Term Matrix found in session. Please run build_dtm_tool first."}

        X = session.feature_matrix
        n_samples, n_features = X.shape

        if n_samples == 0:
            return {"error": "Document-Term Matrix contains 0 documents."}

        if n_features == 0:
            return {"error": "Document-Term Matrix contains 0 features."}

        if n_components != 2:
            return {"error": f"n_components must be 2 for 2D visualization (got {n_components})."}

        # Convert sparse matrix to dense array if necessary
        matrix = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

        method_lower = method.lower()
        explained_variance_ratio = None

        if method_lower == "pca":
            if min(n_samples, n_features) < 2:
                return {
                    "error": f"PCA requires at least 2 samples and 2 features (got {n_samples} samples and {n_features} features)."
                }
            pca = PCA(n_components=n_components, svd_solver="full", random_state=random_state)
            coords = pca.fit_transform(matrix)
            explained_variance_ratio = [float(v) for v in pca.explained_variance_ratio_]

        elif method_lower == "tsne":
            if not (0 < perplexity < n_samples):
                return {
                    "error": f"perplexity ({perplexity}) must satisfy 0 < perplexity < number of samples ({n_samples}). "
                             f"Try setting perplexity between 1 and {n_samples - 1} (e.g. perplexity=2)."
                }
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = tsne.fit_transform(matrix)

        elif method_lower == "umap":
            if not HAS_UMAP:
                return {"error": "umap-learn package is not installed."}
            if not (2 <= n_neighbors <= n_samples):
                return {
                    "error": f"n_neighbors ({n_neighbors}) must satisfy 2 <= n_neighbors <= number of samples ({n_samples}). "
                             f"Try setting n_neighbors between 2 and {n_samples} (e.g. n_neighbors=3)."
                }
            umap_model = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = umap_model.fit_transform(matrix)

        else:
            return {"error": f"Unknown reduction method: '{method}'. Choose 'pca', 'tsne', or 'umap'."}

        # Retrieve categories for color-coding the scatter plot
        labels = ["Uncategorized"] * n_samples
        if session.dataframe is not None:
            df = session.dataframe
            if "category_name" in df.columns and len(df["category_name"]) == n_samples:
                labels = df["category_name"].tolist()
            elif "category" in df.columns and len(df["category"]) == n_samples:
                labels = df["category"].tolist()

        # Build 2D scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=labels,
            style=labels,
            ax=ax,
            palette="tab10",
            s=100,
        )
        dim1_label = "PC1" if method_lower == "pca" else "Dim 1"
        dim2_label = "PC2" if method_lower == "pca" else "Dim 2"
        ax.set_xlabel(dim1_label)
        ax.set_ylabel(dim2_label)
        ax.set_title(f"2D {method_lower.upper()} Dimensionality Reduction")
        ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        # Prepare summary and store result
        result_id = session.next_result_id("dim_red")
        coords_list = [[float(x), float(y)] for x, y in coords]
        session.artifacts["reduced_coordinates"] = coords_list

        summary = {
            "result_id": result_id,
            "status": "success",
            "method": method_lower,
            "n_components": n_components,
            "coordinates": coords_list,
            "explained_variance_ratio": explained_variance_ratio,
        }

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state,
            },
            summary,
        )

        return summary

    @tool
    def binarize_labels_tool(
        column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        One-hot encodes category labels in session.dataframe into binary indicators.

        Always creates one binary indicator column per distinct class (including for binary classification).
        Determines deterministic class ordering and stores classes, encoded matrix, and column name in session.artifacts["binarized_labels"].

        Args:
            column: Optional target column name in session.dataframe to binarize.
                    If None, defaults to 'category_name' if available, otherwise 'category'.

        Returns:
            Dict[str, Any]: Summary containing result_id, status, column, classes, shape, and class_counts.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Please run load_dataset_tool first."}

        df = session.dataframe

        if len(df) == 0:
            return {"error": "session.dataframe is empty."}

        # Resolve target column
        target_col = column
        if target_col is None:
            if "category_name" in df.columns:
                target_col = "category_name"
            elif "category" in df.columns:
                target_col = "category"
            else:
                return {"error": "Neither 'category_name' nor 'category' column found in session.dataframe."}

        if target_col not in df.columns:
            return {"error": f"Column '{target_col}' not found in session.dataframe."}

        series = df[target_col]

        # Reject null labels
        if series.isnull().any():
            return {"error": f"Column '{target_col}' contains null/missing values."}

        labels = series.tolist()

        # Reject non-string labels
        if not all(isinstance(lbl, str) for lbl in labels):
            return {"error": f"Column '{target_col}' contains non-string labels. Please select a string categorical column."}

        # Deterministic class ordering
        unique_classes = sorted(list(set(labels)))

        if len(unique_classes) == 0:
            return {"error": f"Column '{target_col}' contains no valid labels."}

        # One-hot encode labels (always 1 column per class)
        encoded_matrix = []
        class_counts = {c: 0 for c in unique_classes}

        for label in labels:
            row = [1 if label == c else 0 for c in unique_classes]
            encoded_matrix.append(row)
            class_counts[label] += 1

        session.artifacts["binarized_labels"] = {
            "classes": unique_classes,
            "encoded_labels": encoded_matrix,
            "column": target_col,
        }

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "status": "success",
            "column": target_col,
            "classes": unique_classes,
            "shape": [len(encoded_matrix), len(unique_classes)],
            "class_counts": class_counts,
        }

        session.store_result(
            "binarize_labels_tool",
            {"column": column},
            summary,
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
