from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    """
    Factory function returning exploration tools bound to the given SessionState.
    """

    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int,
    ) -> Dict[str, Any]:
        """
        Computes the cosine similarity between two documents in SessionState using raw term counts.
        
        Args:
            doc_index_1: Positional index (0-based) of the first document.
            doc_index_2: Positional index (0-based) of the second document.
            
        Returns:
            Dict[str, Any]: Summary containing result_id, status, doc_index_1,
                            doc_index_2, similarity, and document_texts.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        if session.feature_matrix is None:
            return {"error": "No Document-Term Matrix found in session. Please run build_dtm_tool first."}

        # 1. Validate doc_index_1 type and bounds
        if isinstance(doc_index_1, bool) or not isinstance(doc_index_1, (int, np.integer)):
            return {"error": "doc_index_1 must be an integer."}
        idx1 = int(doc_index_1)

        # 2. Validate doc_index_2 type and bounds
        if isinstance(doc_index_2, bool) or not isinstance(doc_index_2, (int, np.integer)):
            return {"error": "doc_index_2 must be an integer."}
        idx2 = int(doc_index_2)

        df = session.dataframe
        X = session.feature_matrix
        n_docs_df = len(df)
        n_docs_dtm = X.shape[0]

        # Ensure dataframe and feature_matrix row counts match
        if n_docs_df != n_docs_dtm:
            return {"error": f"Session dataframe row count ({n_docs_df}) does not match Document-Term Matrix row count ({n_docs_dtm})."}

        if idx1 < 0 or idx1 >= n_docs_df:
            return {"error": f"doc_index_1 ({idx1}) is out of bounds for dataset with {n_docs_df} documents."}

        if idx2 < 0 or idx2 >= n_docs_df:
            return {"error": f"doc_index_2 ({idx2}) is out of bounds for dataset with {n_docs_df} documents."}

        # Slice 1xV CSR sparse vectors
        v1 = X[idx1]
        v2 = X[idx2]

        # Compute cosine similarity matrix using sklearn.metrics.pairwise.cosine_similarity
        sim_matrix = cosine_similarity(v1, v2)
        raw_sim = float(sim_matrix[0, 0])

        # Handle zero vector or NaN cases safely (e.g., empty text rows)
        similarity = 0.0 if np.isnan(raw_sim) else raw_sim

        # Extract document texts if 'text' column is available
        doc1_text = str(df["text"].iloc[idx1]) if "text" in df.columns else ""
        doc2_text = str(df["text"].iloc[idx2]) if "text" in df.columns else ""

        result_id = session.next_result_id("cosine")

        summary = {
            "result_id": result_id,
            "status": "success",
            "doc_index_1": idx1,
            "doc_index_2": idx2,
            "similarity": similarity,
            "document_texts": {
                "doc_1": doc1_text,
                "doc_2": doc2_text,
            },
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2},
            summary,
        )

        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """
        Computes the Pearson correlation matrix for top-20 terms by variance in the DTM and plots a heatmap.
        Undefined correlations (e.g. for constant zero-variance features) are set to 0.0, including on the diagonal.
        
        Returns:
            Dict[str, Any]: Summary containing result_id, terms, correlation_matrix,
                            n_terms, and status.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No Document-Term Matrix found in session. Please run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = list(session.feature_names)

        n_docs = X.shape[0]
        if n_docs < 2:
            return {"error": f"Cannot compute feature correlation matrix with fewer than 2 documents (found {n_docs})."}

        if len(feature_names) == 0 or X.shape[1] == 0:
            return {"error": "Document-Term Matrix vocabulary is empty."}

        # Convert sparse matrix to dense array if necessary
        matrix = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

        # 1. Compute term variances across all documents (axis 0)
        variances = np.var(matrix, axis=0)

        # 2. Rank terms by variance using stable sort (descending order)
        sorted_indices = np.argsort(-variances, kind="stable")

        # 3. Take top 20 terms (or fewer if fewer terms exist in vocabulary)
        top_k = min(20, len(feature_names))
        selected_indices = sorted_indices[:top_k]
        selected_terms = [feature_names[i] for i in selected_indices]

        # Extract submatrix for top-k terms
        submatrix = matrix[:, selected_indices]

        # 4. Compute Pearson correlation matrix across documents
        if top_k == 1:
            # Single feature case: check variance
            var_val = float(np.var(submatrix[:, 0]))
            corr_matrix = np.array([[1.0 if var_val > 0 else 0.0]])
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_matrix = np.corrcoef(submatrix, rowvar=False)

            # Convert scalar/0D result from np.corrcoef to 2D array if necessary
            if np.ndim(corr_matrix) == 0:
                corr_matrix = np.array([[float(corr_matrix)]])

            # Replace NaNs (undefined correlations, e.g. for constant zero-variance features) with 0.0.
            # Do NOT force diagonal to 1.0 for zero-variance features where correlation is undefined.
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # 5. Build annotated heatmap figure
        fig, ax = plt.subplots(figsize=(10, 8))
        if corr_matrix.size > 0:
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                xticklabels=selected_terms,
                yticklabels=selected_terms,
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                ax=ax,
            )
            ax.set_title("Top-20 Feature Correlation Heatmap")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
        else:
            ax.text(0.5, 0.5, "No features available", ha="center", va="center")

        fig.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        # 6. Prepare summary and store result (preserve full float precision in JSON)
        result_id = session.next_result_id("feature_corr")
        corr_list = [[float(val) for val in row] for row in corr_matrix]

        summary = {
            "result_id": result_id,
            "status": "success",
            "terms": selected_terms,
            "correlation_matrix": corr_list,
            "n_terms": len(selected_terms),
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {},
            summary,
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
