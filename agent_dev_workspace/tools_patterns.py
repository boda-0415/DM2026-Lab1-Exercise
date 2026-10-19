import os
import tempfile
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import StrictInt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from PAMI.extras.convert.DF2DB import DF2DB
from PAMI.frequentPattern.basic.FPGrowth import FPGrowth
from PAMI.frequentPattern.maximal.MaxFPGrowth import MaxFPGrowth
from PAMI.frequentPattern.topk.FAE import FAE


def make_tools(session: Any) -> List[Any]:
    """Factory function that returns tools for pattern mining bound to session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: Optional[StrictInt] = None,
        k: Optional[StrictInt] = None,
    ) -> Dict[str, Any]:
        """Mine frequent itemsets/patterns for a specific category using PAMI.

        Args:
            category_name: The category to filter documents by (e.g. 'catA', 'catB').
            filtering_method: Vocabulary filtering method ('variance', 'tfidf', or 'term_frequency').
            algorithm: Mining algorithm to run ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support count (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-K count (required for 'topk').

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm,
            kept_terms_count, patterns list, and note if applicable.
        """
        # 1. Validate session dataframe
        df = session.dataframe
        if df is None or df.empty:
            return {"error": "No dataframe loaded in session."}

        # Check category column name
        cat_col = None
        if "category_name" in df.columns:
            cat_col = "category_name"
        elif "category" in df.columns:
            cat_col = "category"
        else:
            return {"error": "Column 'category_name' or 'category' not found in session dataframe."}

        if "text" not in df.columns:
            return {"error": "Column 'text' not found in session dataframe."}

        filtering_method = filtering_method.lower()
        if filtering_method not in ["variance", "tfidf", "term_frequency"]:
            return {
                "error": f"Invalid filtering_method '{filtering_method}'. Must be 'variance', 'tfidf', or 'term_frequency'."
            }

        algorithm = algorithm.lower()
        if algorithm not in ["fpgrowth", "topk", "maxfpgrowth"]:
            return {
                "error": f"Invalid algorithm '{algorithm}'. Must be 'fpgrowth', 'topk', or 'maxfpgrowth'."
            }

        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or type(min_sup) is not int or min_sup <= 0:
                return {"error": f"Parameter 'min_sup' must be a positive integer for algorithm '{algorithm}'."}
        elif algorithm == "topk":
            if k is None or type(k) is not int or k <= 0:
                return {"error": "Parameter 'k' must be a positive integer for algorithm 'topk'."}

        # Filter dataframe by category_name
        cat_df = df[df[cat_col] == category_name]
        if cat_df.empty:
            return {"error": f"No documents found for category_name '{category_name}'."}

        # Filter out empty text rows
        text_series = cat_df["text"].dropna().astype(str)
        text_series = text_series[text_series.str.strip() != ""]
        if text_series.empty:
            return {"error": f"No non-empty text documents found for category '{category_name}'."}

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        # Helper for zero terms surviving
        def _return_no_terms_survived():
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "kept_terms_count": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.artifacts[artifact_key] = []
            session.store_result(
                "mine_patterns_tool",
                {
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                    "min_sup": min_sup,
                    "k": k,
                },
                summary,
            )
            return summary

        # 2. Build fresh category-local CountVectorizer DTM (stop_words="english")
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X = vectorizer.fit_transform(text_series)
        except ValueError as exc:
            # Only treat empty-vocabulary ValueError as a no-terms result
            err_msg = str(exc).lower()
            if "empty vocabulary" in err_msg or "vocabulary" in err_msg:
                return _return_no_terms_survived()
            raise

        feature_names = np.array(vectorizer.get_feature_names_out())
        n_terms = len(feature_names)
        if n_terms == 0:
            return _return_no_terms_survived()

        X_array = X.toarray()

        # 3. Apply filtering_method
        if filtering_method == "variance":
            # Variance per term column
            variances = np.var(X_array, axis=0)
            p5 = np.percentile(variances, 5)
            p95 = np.percentile(variances, 95)
            mask = (variances > p5) & (variances < p95)
        elif filtering_method == "term_frequency":
            # Column sums
            freqs = np.sum(X_array, axis=0)
            p5 = np.percentile(freqs, 5)
            p95 = np.percentile(freqs, 95)
            mask = (freqs > p5) & (freqs < p95)
        elif filtering_method == "tfidf":
            # TF-IDF transformer on local DTM
            tfidf_transformer = TfidfTransformer()
            tfidf_mat = tfidf_transformer.fit_transform(X).toarray()
            mean_tfidf = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            mask = mean_tfidf >= p20

        surviving_indices = np.where(mask)[0]
        if len(surviving_indices) == 0:
            return _return_no_terms_survived()

        surviving_terms = feature_names[surviving_indices]
        surviving_X = X_array[:, surviving_indices]

        # 4. Build transactional DB via PAMI DF2DB
        local_df = pd.DataFrame(surviving_X, columns=surviving_terms)

        tmp_db_file = None
        patterns_list = []
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
                tmp_db_file = f.name

            # Convert to transactional DB (condition=">", thresholdValue=0)
            converter = DF2DB(inputDF=local_df, DFtype="dense")
            converter.convert2TransactionalDatabase(
                oFile=tmp_db_file, condition=">", thresholdValue=0
            )

            # 5. Mine using PAMI algorithm
            miner = None
            if algorithm == "fpgrowth":
                miner = FPGrowth(iFile=tmp_db_file, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                miner = FAE(iFile=tmp_db_file, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth(iFile=tmp_db_file, minSup=min_sup, sep="\t")

            miner.mine()
            patterns_df = miner.getPatternsAsDataFrame()

            if patterns_df is not None and not patterns_df.empty:
                # Verify required columns exist
                cols_lower = {c.lower(): c for c in patterns_df.columns}
                if "patterns" not in cols_lower or "support" not in cols_lower:
                    return {
                        "error": f"Unexpected columns in PAMI output dataframe: {list(patterns_df.columns)}"
                    }
                pat_col = cols_lower["patterns"]
                sup_col = cols_lower["support"]

                for idx in range(len(patterns_df)):
                    row = patterns_df.iloc[idx]
                    raw_pat = row[pat_col]
                    sup = int(row[sup_col])
                    # PAMI patterns are tab or space delimited strings or lists
                    if isinstance(raw_pat, str):
                        raw_pat_str = raw_pat.strip()
                        if "\t" in raw_pat_str:
                            terms = [t.strip() for t in raw_pat_str.split("\t") if t.strip()]
                        else:
                            terms = [t.strip() for t in raw_pat_str.split() if t.strip()]
                    elif isinstance(raw_pat, (list, tuple)):
                        terms = [str(t).strip() for t in raw_pat if str(t).strip()]
                    else:
                        terms = [str(raw_pat).strip()]

                    patterns_list.append({"pattern": terms, "support": sup})

                # Sort pattern terms and patterns list for deterministic output
                for p_item in patterns_list:
                    p_item["pattern"].sort()
                patterns_list.sort(key=lambda x: (-x["support"], " ".join(x["pattern"])))
        finally:
            if tmp_db_file and os.path.exists(tmp_db_file):
                try:
                    os.remove(tmp_db_file)
                except Exception:
                    pass

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "kept_terms_count": len(surviving_terms),
            "patterns": patterns_list,
        }
        if len(patterns_list) == 0:
            summary["note"] = "zero patterns found"

        session.artifacts[artifact_key] = patterns_list
        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
        )
        return summary

    return [mine_patterns_tool]
