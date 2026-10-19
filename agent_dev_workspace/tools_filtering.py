from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.1,
    ) -> dict:
        """Flags terms in the Document-Term Matrix (DTM) with variance below a given threshold.

        Calculates population variance (ddof=0) for each feature in session.feature_matrix,
        ranks terms by variance, and reports retained vs. removed features. Passes a sorted
        full_report DataFrame to session.store_result for visualization.

        Args:
            threshold: Minimum variance required for a term to be retained (default: 0.1).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No Document-Term Matrix found in session. Please run build_dtm_tool first."
            }

        X = session.feature_matrix
        feature_names = session.feature_names

        # Compute mean and mean of squares using element-wise sparse multiplication (ddof=0)
        mean = np.ravel(X.mean(axis=0))
        mean_sq = np.ravel(X.multiply(X).mean(axis=0))
        variances = mean_sq - np.square(mean)

        # Create DataFrame report sorted by variance descending
        df_report = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        retained_mask = df_report["variance"] >= threshold
        retained_terms = df_report.loc[retained_mask, "term"].tolist()
        removed_terms = df_report.loc[~retained_mask, "term"].tolist()

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_features": len(feature_names),
            "retained_features_count": len(retained_terms),
            "removed_features_count": len(removed_terms),
            "retained_features": retained_terms,
            "removed_features": removed_terms,
            "variances": {str(k): float(v) for k, v in zip(df_report["term"], df_report["variance"].round(6))}
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=df_report
        )
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0,
    ) -> dict:
        """Calculates Pearson correlation between each term's frequency and a target class.

        Performs a One-vs-Rest calculation against target_class on session.feature_matrix.
        Ranks features by signed Pearson r descending, filtering out features below threshold.
        Passes a sorted full_report DataFrame with 'term' and 'pearson_r' columns to session.store_result.

        Args:
            target_class: The category name to compute linear correlation against (e.g. 'catB').
            threshold: Minimum signed Pearson r value required for a term to be retained (default: 0.0).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No Document-Term Matrix found in session. Please run build_dtm_tool first."
            }

        X = session.feature_matrix
        feature_names = session.feature_names
        n_samples = X.shape[0]

        # Retrieve and validate labels against DTM row count
        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        if labels is None:
            return {
                "error": "No class labels found in session. Please run load_dataset_tool and build_dtm_tool first."
            }

        if len(labels) != n_samples:
            return {
                "error": f"Label count ({len(labels)}) does not match Document-Term Matrix row count ({n_samples})."
            }

        # Validate target_class strictly against unique labels present in the dataset
        actual_classes = list(np.unique(labels))
        if target_class not in actual_classes:
            return {
                "error": f"Target class '{target_class}' not found in dataset labels. Available classes: {actual_classes}"
            }

        # Binary indicator y: 1 if document belongs to target_class, else 0
        y = (labels == target_class).astype(float)

        std_y = np.std(y, ddof=0)
        if std_y == 0:
            return {
                "error": f"Target class binary indicator for '{target_class}' has zero variance (present in all or no documents)."
            }

        y_centered = y - np.mean(y)

        # Compute term means and standard deviations
        mean_X = np.ravel(X.mean(axis=0))
        mean_sq_X = np.ravel(X.multiply(X).mean(axis=0))
        var_X = np.maximum(0.0, mean_sq_X - np.square(mean_X))
        std_X = np.sqrt(var_X)

        # Compute covariance term-by-term: E[(X - mean_X)(y - mean_y)] = X.T.dot(y_centered) / N
        cov_X_y = np.ravel(X.T.dot(y_centered)) / n_samples

        # Compute Pearson r, safely handling zero-variance terms (std_X == 0)
        denom = std_X * std_y
        with np.errstate(divide='ignore', invalid='ignore'):
            pearson_r = np.where(denom > 0, cov_X_y / denom, 0.0)

        pearson_r = np.nan_to_num(pearson_r, nan=0.0)

        # Create DataFrame report sorted by pearson_r descending
        df_report = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_r
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        retained_mask = df_report["pearson_r"] >= threshold
        retained_terms = df_report.loc[retained_mask, "term"].tolist()
        removed_terms = df_report.loc[~retained_mask, "term"].tolist()

        # Convert numpy floats to native Python floats for reliable JSON serialization
        pearson_r_dict = {
            str(k): float(v) for k, v in zip(df_report["term"], df_report["pearson_r"].round(6))
        }

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_features": len(feature_names),
            "retained_features_count": len(retained_terms),
            "removed_features_count": len(removed_terms),
            "retained_features": retained_terms,
            "removed_features": removed_terms,
            "pearson_r": pearson_r_dict
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report
        )
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
    ) -> dict:
        """Calculates Spearman rank correlation between each term's frequency and a target class.

        Performs a One-vs-Rest calculation against target_class using rankdata(method='average').
        Ranks features by signed Spearman rho descending, filtering out features below threshold.
        Passes a sorted full_report DataFrame with 'term' and 'spearman_r' columns to session.store_result.

        Args:
            target_class: The category name to compute rank correlation against (e.g. 'catB').
            threshold: Minimum signed Spearman rho value required for a term to be retained (default: 0.0).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No Document-Term Matrix found in session. Please run build_dtm_tool first."
            }

        X = session.feature_matrix
        feature_names = session.feature_names
        n_samples, n_features = X.shape

        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        if labels is None:
            return {
                "error": "No class labels found in session. Please run load_dataset_tool and build_dtm_tool first."
            }

        if len(labels) != n_samples:
            return {
                "error": f"Label count ({len(labels)}) does not match Document-Term Matrix row count ({n_samples})."
            }

        actual_classes = list(np.unique(labels))
        if target_class not in actual_classes:
            return {
                "error": f"Target class '{target_class}' not found in dataset labels. Available classes: {actual_classes}"
            }

        # Binary indicator y: 1 if document belongs to target_class, else 0
        y = (labels == target_class).astype(float)
        std_y_raw = np.std(y, ddof=0)
        if std_y_raw == 0:
            return {
                "error": f"Target class binary indicator for '{target_class}' has zero variance (present in all or no documents)."
            }

        # Rank target indicator y using average ties method
        y_ranked = rankdata(y, method="average")
        std_y_ranked = np.std(y_ranked, ddof=0)
        y_ranked_centered = y_ranked - np.mean(y_ranked)

        # Compute Spearman rho term-by-term by ranking each column individually
        spearman_rho = np.zeros(n_features, dtype=float)

        # Convert to CSC format for fast column slicing without full dense matrix conversion
        X_csc = X.tocsc()

        for j in range(n_features):
            # Extract 1D array for single column j
            col_data = np.ravel(X_csc[:, j].toarray())
            std_col = np.std(col_data, ddof=0)
            if std_col == 0:
                spearman_rho[j] = 0.0
                continue

            col_ranked = rankdata(col_data, method="average")
            std_col_ranked = np.std(col_ranked, ddof=0)
            if std_col_ranked == 0:
                spearman_rho[j] = 0.0
                continue

            cov_ranked = np.mean((col_ranked - np.mean(col_ranked)) * y_ranked_centered)
            spearman_rho[j] = cov_ranked / (std_col_ranked * std_y_ranked)

        spearman_rho = np.nan_to_num(spearman_rho, nan=0.0)

        # Create DataFrame report sorted by spearman_r descending
        df_report = pd.DataFrame({
            "term": feature_names,
            "spearman_r": spearman_rho
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        retained_mask = df_report["spearman_r"] >= threshold
        retained_terms = df_report.loc[retained_mask, "term"].tolist()
        removed_terms = df_report.loc[~retained_mask, "term"].tolist()

        spearman_r_dict = {
            str(k): float(v) for k, v in zip(df_report["term"], df_report["spearman_r"].round(6))
        }

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_features": len(feature_names),
            "retained_features_count": len(retained_terms),
            "removed_features_count": len(removed_terms),
            "retained_features": retained_terms,
            "removed_features": removed_terms,
            "spearman_r": spearman_r_dict
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
