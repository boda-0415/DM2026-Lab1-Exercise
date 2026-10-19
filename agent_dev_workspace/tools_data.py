from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    """
    Factory function returning data management tools bound to the given SessionState.
    """

    @tool
    def tokenize_tool(lowercase: bool = True) -> Dict[str, Any]:
        """
        Tokenizes the document text column into unigrams using NLTK word_tokenize.
        
        Args:
            lowercase: Whether to convert text to lowercase before tokenizing (default: True).
            
        Returns:
            Dict[str, Any]: Summary containing result_id, status, total_documents, total_tokens,
                            avg_tokens_per_doc, and lowercase setting.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        if "text" not in session.dataframe.columns:
            return {"error": "Dataframe missing required 'text' column."}

        df = session.dataframe

        def tokenize_text(val):
            if pd.isna(val) or val is None:
                return []
            text_str = str(val)
            if not text_str.strip():
                return []
            if lowercase:
                text_str = text_str.lower()
            return nltk.word_tokenize(text_str, preserve_line=True)

        tokens_list = [tokenize_text(val) for val in df["text"]]

        # Store in session
        df["unigrams"] = tokens_list
        session.artifacts["unigrams"] = tokens_list

        total_docs = len(tokens_list)
        total_tokens = sum(len(doc_tokens) for doc_tokens in tokens_list)
        avg_tokens = (total_tokens / total_docs) if total_docs > 0 else 0.0

        result_id = session.next_result_id("tokenize")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_documents": total_docs,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(avg_tokens, 2),
            "lowercase": lowercase,
        }

        session.store_result("tokenize_tool", {"lowercase": lowercase}, summary)

        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """
        Lists files and subdirectories within a given subdirectory of the repository root.
        
        Args:
            subdirectory: Subdirectory path relative to the repo root (default: "").
            
        Returns:
            Dict[str, Any]: Summary containing result_id, subdirectory, sorted lists of
                            directories and files, and totals.
        """
        base_path = Path.cwd().resolve()
        
        try:
            target_path = (base_path / subdirectory).resolve()
            # Enforce path containment within repo root using relative_to()
            target_path.relative_to(base_path)
        except (ValueError, OSError):
            return {"error": "Access denied: Path is outside the repository base directory."}

        if not target_path.exists():
            return {"error": f"Directory not found: {subdirectory}"}

        if not target_path.is_dir():
            return {"error": f"Path is not a directory: {subdirectory}"}

        directories = []
        files = []

        try:
            for item in target_path.iterdir():
                if item.is_dir():
                    directories.append(item.name)
                elif item.is_file():
                    files.append(item.name)
        except Exception as e:
            return {"error": f"Failed to list directory contents: {str(e)}"}

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")

        summary = {
            "result_id": result_id,
            "status": "success",
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
            "total_directories": len(directories),
            "total_files": len(files),
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)

        return summary

    @tool
    def inspect_data_tool(n_rows: Any = 5, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Inspects the top n_rows of the working DataFrame in SessionState.
        
        Args:
            n_rows: Number of rows to inspect (must be a positive integer, default: 5).
            columns: Optional list of column names to include (default: None for all columns).
            
        Returns:
            Dict[str, Any]: Summary containing result_id, total_rows, total_columns,
                            columns inspected, count of rows returned, and sample records.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        # Normalize n_rows before validation
        if not isinstance(n_rows, bool):
            if isinstance(n_rows, str) and n_rows.isdigit():
                n_rows = int(n_rows)
            elif isinstance(n_rows, (float, np.floating)) and float(n_rows).is_integer():
                n_rows = int(n_rows)

        # 1. Validate n_rows
        if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows <= 0:
            return {"error": "n_rows must be a positive integer."}

        # 2. Validate columns type
        if columns is not None:
            if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
                return {"error": "columns must be None or a list of column names (strings)."}

        df = session.dataframe

        # 3. Validate column existence
        if columns is not None:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return {"error": f"Columns not found in dataframe: {missing_cols}"}
            target_cols = columns
        else:
            target_cols = list(df.columns)

        # Ensure column string keys are unique when stringified
        str_cols = [str(c) for c in target_cols]
        if len(set(str_cols)) != len(str_cols):
            return {"error": "Duplicate column names encountered when converting to string keys."}

        # Slice DataFrame without modifying original
        df_slice = df[target_cols].head(n_rows)

        # JSON-safe sanitizer that handles infinity and lists/tuples BEFORE pd.isna check
        def sanitize_val(val):
            if val is None:
                return None
            if isinstance(val, (list, tuple)):
                return [sanitize_val(v) for v in val]
            if isinstance(val, dict):
                return {str(k): sanitize_val(v) for k, v in val.items()}
            if pd.isna(val):
                return None
            if isinstance(val, (float, np.floating)):
                if math.isinf(val):
                    return None
                return float(val)
            if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
                return int(val)
            if isinstance(val, (bool, np.bool_)):
                return bool(val)
            if isinstance(val, (pd.Timestamp, pd.Timedelta)):
                return str(val)
            return str(val) if not isinstance(val, (str, int, float, bool)) else val

        records = []
        for _, row in df_slice.iterrows():
            record = {str(col): sanitize_val(row[col]) for col in target_cols}
            records.append(record)

        result_id = session.next_result_id("inspect")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": str_cols,
            "inspected_rows_count": len(records),
            "sample_records": records,
        }

        session.store_result("inspect_data_tool", {"n_rows": n_rows, "columns": columns}, summary)

        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """
        Checks the working DataFrame in SessionState for missing or empty text values.
        
        Returns:
            Dict[str, Any]: Summary containing result_id, status, total_rows, missing_count,
                            and list of zero-based row positions (not index labels) with missing text.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        if "text" not in session.dataframe.columns:
            return {"error": "Dataframe missing required 'text' column."}

        df = session.dataframe

        def is_missing(val):
            if pd.isna(val) or val is None:
                return True
            if isinstance(val, str) and not val.strip():
                return True
            return False

        # Returns 0-based positional row indices (enumerate), regardless of df.index labels
        missing_indices = [int(pos) for pos, val in enumerate(df["text"]) if is_missing(val)]

        result_id = session.next_result_id("missing")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": len(df),
            "missing_count": len(missing_indices),
            "missing_indices": missing_indices,
        }

        session.store_result("check_missing_tool", {}, summary)

        return summary

    @tool
    def check_duplicates_tool(subset: Optional[List[str]] = None, drop: bool = False) -> Dict[str, Any]:
        """
        Checks for and optionally drops duplicate documents in SessionState.dataframe using keep=False.
        
        Args:
            subset: Optional list of column names to evaluate duplicates on (default: None for ["text"]).
            drop: Whether to drop duplicate rows from session.dataframe using keep=False (default: False).
            
        Returns:
            Dict[str, Any]: Summary containing result_id, status, initial_rows, duplicate_count,
                            removed_count, remaining_rows, dropped mode, and subset used.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe

        if subset is None:
            if "text" not in df.columns:
                return {"error": "Dataframe missing required default 'text' column for duplicate checking."}
            target_subset = ["text"]
        else:
            if not isinstance(subset, (list, tuple)) or not all(isinstance(c, str) for c in subset):
                return {"error": "subset must be None or a list of column names (strings)."}
            if len(subset) == 0:
                return {"error": "subset cannot be empty."}
            target_subset = list(subset)
            missing = [c for c in target_subset if c not in df.columns]
            if missing:
                return {"error": f"Columns not found in dataframe: {missing}"}

        # Identify duplicate rows using keep=False (flags every copy of duplicated rows)
        dup_mask = df.duplicated(subset=target_subset, keep=False)
        duplicate_count = int(dup_mask.sum())
        initial_rows = len(df)

        removed_count = 0
        if drop and duplicate_count > 0:
            # Drop every copy of duplicate rows using keep=False matching Master's drop_duplicates(keep=False)
            session.dataframe = df.drop_duplicates(subset=target_subset, keep=False).reset_index(drop=True)
            removed_count = duplicate_count

        remaining_rows = len(session.dataframe)

        result_id = session.next_result_id("duplicates")

        summary = {
            "result_id": result_id,
            "status": "success",
            "initial_rows": initial_rows,
            "duplicate_count": duplicate_count,
            "removed_count": removed_count,
            "remaining_rows": remaining_rows,
            "dropped": drop,
            "subset": target_subset,
        }

        session.store_result("check_duplicates_tool", {"subset": subset, "drop": drop}, summary)

        return summary

    @tool
    def sample_data_tool(
        n: Any = None,
        frac: Any = None,
        replace: Any = False,
        random_state: Any = None,
    ) -> Dict[str, Any]:
        """
        Subsamples the working DataFrame in SessionState.
        
        Args:
            n: Number of items to sample (cannot be used with frac).
            frac: Fraction of items to sample (cannot be used with n).
            replace: Whether to sample with replacement (default: False).
            random_state: Seed for random number generator for reproducibility.
            
        Returns:
            Dict[str, Any]: Summary containing result_id, status, initial_rows,
                            sampled_rows, sampling_mode ('n' or 'frac'),
                            parameter_value, replace, and random_state.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        # 1. Validate replace boolean type
        if not isinstance(replace, bool):
            return {"error": "replace must be a boolean."}

        # 2. Validate random_state type
        if random_state is not None:
            if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)) or random_state < 0:
                return {"error": "random_state must be a non-negative integer."}
            random_state = int(random_state)

        # 3. Exactly one of n or frac must be provided
        if (n is None and frac is None) or (n is not None and frac is not None):
            return {"error": "Must provide exactly one of 'n' or 'frac'."}

        df = session.dataframe
        initial_rows = len(df)

        sampling_mode = ""
        parameter_value = None

        if n is not None:
            if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
                return {"error": "n must be a positive integer."}
            n_val = int(n)
            if not replace and n_val > initial_rows:
                return {"error": f"Cannot sample n={n_val} samples when dataframe has {initial_rows} rows and replace=False."}
            sampling_mode = "n"
            parameter_value = n_val
            sample_args = {"n": n_val, "frac": None}
        else:
            if isinstance(frac, bool) or not isinstance(frac, (int, float, np.number)):
                return {"error": "frac must be a positive float."}
            frac_val = float(frac)
            if not math.isfinite(frac_val) or frac_val <= 0.0:
                return {"error": "frac must be a finite positive number."}
            if not replace and frac_val > 1.0:
                return {"error": "frac cannot be greater than 1.0 when replace=False."}
            sampling_mode = "frac"
            parameter_value = frac_val
            sample_args = {"n": None, "frac": frac_val}

        # Sample DataFrame and update session.dataframe in place
        sampled_df = df.sample(
            n=sample_args["n"],
            frac=sample_args["frac"],
            replace=replace,
            random_state=random_state,
        ).reset_index(drop=True)

        session.dataframe = sampled_df
        sampled_rows = len(sampled_df)

        result_id = session.next_result_id("sample")

        summary = {
            "result_id": result_id,
            "status": "success",
            "initial_rows": initial_rows,
            "sampled_rows": sampled_rows,
            "sampling_mode": sampling_mode,
            "parameter_value": parameter_value,
            "replace": replace,
            "random_state": random_state,
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "frac": frac, "replace": replace, "random_state": random_state},
            summary,
        )

        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """
        Computes descriptive statistics on document text character length and
        generates a boxplot grouped by category.
        
        Returns:
            Dict[str, Any]: Summary containing result_id, status, overall_stats,
                            and per_category_stats.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe

        if "text" not in df.columns:
            return {"error": "Dataframe missing required 'text' column."}

        if "category_name" not in df.columns:
            return {"error": "Dataframe missing required 'category_name' column."}

        # Strict string type check for all values in 'text'
        if not all(isinstance(v, str) for v in df["text"]):
            return {"error": "All values in 'text' column must be strings."}

        if len(df) == 0:
            return {"error": "Dataframe is empty."}

        # Compute text character lengths exactly per spec: df["text"].apply(len)
        text_lengths = df["text"].apply(len)

        # Helper to convert pandas/numpy describe stats to clean JSON-safe floats/None
        def sanitize_describe(stats_series: pd.Series) -> Dict[str, Any]:
            res = {}
            for k, v in stats_series.items():
                val = float(v)
                res[str(k)] = val if math.isfinite(val) else None
            return res

        overall_stats = sanitize_describe(text_lengths.describe())

        # Build per-category stats
        df_temp = pd.DataFrame({
            "text_length": text_lengths,
            "category_name": df["category_name"],
        })

        per_category_stats = {}
        for cat_name, group in df_temp.groupby("category_name"):
            per_category_stats[str(cat_name)] = sanitize_describe(group["text_length"].describe())

        # Build boxplot figure
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_temp, x="category_name", y="text_length", ax=ax, palette="Set2")
        ax.set_title("Document Text Length Distribution by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        result_id = session.next_result_id("describe")

        summary = {
            "result_id": result_id,
            "status": "success",
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats,
        }

        session.store_result("describe_data_tool", {}, summary)

        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]
