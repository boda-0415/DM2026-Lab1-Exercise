from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        ngram_range: List[int] = [1, 1],
    ) -> dict:
        """Build a Document-Term Matrix (DTM) from session.dataframe text column.

        Fits a CountVectorizer on the text dataset, populates session.feature_matrix
        and session.feature_names, calculates matrix sparsity, and stores the fitted
        vectorizer in session.artifacts['count_vectorizer'].

        Args:
            max_features: Optional maximum number of terms to keep in the vocabulary based on frequency.
            ngram_range: Lower and upper boundary of range of n-values for different n-grams to be extracted, e.g. [1, 1] or [1, 2].
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Please run load_dataset_tool first."}

        # Input validation for ngram_range
        if (
            not isinstance(ngram_range, (list, tuple))
            or len(ngram_range) != 2
            or not all(isinstance(x, int) and x > 0 for x in ngram_range)
            or ngram_range[0] > ngram_range[1]
        ):
            return {
                "error": "ngram_range must be a list/tuple of two positive integers in ascending order, e.g. [1, 1] or [1, 2]."
            }

        text_series = session.dataframe["text"].fillna("")

        ngram_tuple = (ngram_range[0], ngram_range[1])
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_tuple)

        try:
            X = vectorizer.fit_transform(text_series)
        except ValueError as e:
            # Handles empty vocabulary errors when text contains no valid terms or only stop words
            return {"error": f"Failed to build Document-Term Matrix: {str(e)}"}

        session.feature_matrix = X
        session.feature_names = vectorizer.get_feature_names_out().tolist()

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        session.artifacts["count_vectorizer"] = vectorizer

        n_rows, n_cols = X.shape
        non_zero = int(X.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - (non_zero / total_elements))) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "vocabulary_sample": session.feature_names[:10]
        }

        session.store_result("build_dtm_tool", {"max_features": max_features, "ngram_range": ngram_range}, summary)
        return summary

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = None,
    ) -> dict:
        """Aggregates term frequencies across all documents in session.feature_matrix.

        Sums term counts across columns of the global Document-Term Matrix, ranks terms by frequency
        descending, and passes a full_report DataFrame with 'term' and 'frequency' columns to session.store_result.

        Args:
            top_n: Optional number of top frequent terms to return in top_terms list in the summary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No Document-Term Matrix found in session. Please run build_dtm_tool first."
            }

        # Validate top_n parameter if provided
        if top_n is not None and (not isinstance(top_n, int) or top_n <= 0):
            return {
                "error": "top_n must be a positive integer (greater than 0) or None."
            }

        X = session.feature_matrix
        feature_names = session.feature_names

        # Compute column sums across all documents
        counts = np.ravel(X.sum(axis=0))

        # Create full DataFrame report sorted by frequency descending, then alphabetically by term
        df_report = pd.DataFrame({
            "term": feature_names,
            "frequency": [int(c) for c in counts]
        }).sort_values(by=["frequency", "term"], ascending=[False, True]).reset_index(drop=True)

        total_terms = len(df_report)
        total_word_count = int(df_report["frequency"].sum())

        slice_limit = top_n if top_n is not None else total_terms
        top_report = df_report.head(slice_limit)

        term_freq_dict = {
            str(row["term"]): int(row["frequency"])
            for _, row in df_report.iterrows()
        }

        top_terms_list = [
            {"term": str(row["term"]), "frequency": int(row["frequency"])}
            for _, row in top_report.iterrows()
        ]

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "total_terms": total_terms,
            "total_word_count": total_word_count,
            "term_frequencies": term_freq_dict,
            "top_terms": top_terms_list
        }

        # Store complete full_report regardless of top_n
        session.store_result(
            "term_frequency_tool",
            {"top_n": top_n},
            summary,
            full_report=df_report
        )
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> dict:
        """Renders a heatmap of a positional slice of the global Document-Term Matrix.

        Slices session.feature_matrix[0:n_documents, 0:n_terms] in existing row/column order,
        generates a seaborn heatmap, sets session.pending_figure, and returns the sliced matrix.

        Args:
            n_terms: Number of terms (columns) to include from the start of session.feature_names (default: 20).
            n_documents: Number of documents (rows) to include from the start of session.dataframe (default: 20).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No Document-Term Matrix found in session. Please run build_dtm_tool first."
            }

        if not isinstance(n_terms, int) or n_terms <= 0:
            return {"error": "n_terms must be a positive integer (greater than 0)."}

        if not isinstance(n_documents, int) or n_documents <= 0:
            return {"error": "n_documents must be a positive integer (greater than 0)."}

        X = session.feature_matrix
        feature_names = session.feature_names
        total_docs, total_terms = X.shape

        max_docs = min(n_documents, total_docs)
        max_terms = min(n_terms, total_terms)

        sub_matrix = X[:max_docs, :max_terms].toarray()
        included_terms = feature_names[:max_terms]
        included_documents = [f"Doc_{i}" for i in range(max_docs)]

        # Generate Seaborn heatmap figure
        fig, ax = plt.subplots(figsize=(max(6, max_terms * 0.8), max(4, max_docs * 0.5)))
        sns.heatmap(
            sub_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=included_terms,
            yticklabels=included_documents,
            ax=ax
        )
        ax.set_title("Document-Term Matrix Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        fig.tight_layout()

        # Set pending_figure on session without closing or plt.show()
        session.pending_figure = fig

        sliced_matrix = [[int(val) for val in row] for row in sub_matrix]

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_terms": max_terms,
            "n_documents": max_docs,
            "included_terms": included_terms,
            "included_documents": included_documents,
            "sliced_matrix": sliced_matrix
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
