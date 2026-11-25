from typing import Dict, Any, Optional
from langchain_core.tools import tool
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the punkt tokenizer model is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def make_tools(session):
    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Check the working dataset for missing, null, or whitespace-only text values.

        Returns:
            Dict containing result_id, missing_count, missing_indices (DataFrame index
            labels capped at 10), and a truncated flag indicating if more missing rows exist.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Call load_dataset_tool first."}
        
        if "text" not in session.dataframe.columns:
            return {"error": "Dataset does not contain a 'text' column."}

        # Check for NaN/null or empty/whitespace-only text
        series = session.dataframe["text"]
        is_missing = series.isna() | (series.astype(str).str.strip() == "")
        
        missing_indices_all = session.dataframe[is_missing].index.tolist()
        missing_count = len(missing_indices_all)
        
        truncated = missing_count > 10
        missing_indices = missing_indices_all[:10]
        
        res_id = session.next_result_id("missing")
        summary = {
            "result_id": res_id,
            "missing_count": missing_count,
            "missing_indices": missing_indices,
            "truncated": truncated
        }
        
        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Check for (and optionally drop) duplicate rows in the dataset.

        Args:
            drop: If True, drops all copies of duplicated rows (keep=False).

        Returns:
            Dict containing result_id, duplicate_count, initial_rows, remaining_rows, and rows_removed.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Call load_dataset_tool first."}

        initial_rows = len(session.dataframe)
        # Duplicate detection across all columns matching Master's dataframe.duplicated(keep="first")
        duplicate_mask = session.dataframe.duplicated(keep="first")
        duplicate_count = int(duplicate_mask.sum())
        
        if drop and duplicate_count > 0:
            # Per contract: use keep=False on the whole DataFrame to drop every copy of a duplicated row
            session.dataframe = session.dataframe.drop_duplicates(keep=False)
        
        remaining_rows = len(session.dataframe)
        rows_removed = initial_rows - remaining_rows
        
        res_id = session.next_result_id("duplicates")
        summary = {
            "result_id": res_id,
            "duplicate_count": duplicate_count,
            "initial_rows": initial_rows,
            "remaining_rows": remaining_rows,
            "rows_removed": rows_removed
        }
        
        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenize document text into lowercased unigrams.

        Stores the resulting token lists in session.dataframe['unigrams'].

        Returns:
            Dict containing result_id, n_documents, and total_tokens.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Call load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"error": "Dataset does not contain a 'text' column."}

        def tokenize_row(text):
            if not isinstance(text, str) or not text.strip():
                return []
            return word_tokenize(text.lower())

        session.dataframe["unigrams"] = session.dataframe["text"].apply(tokenize_row)
        
        n_documents = len(session.dataframe)
        total_tokens = int(session.dataframe["unigrams"].apply(len).sum())
        
        res_id = session.next_result_id("tokenize")
        summary = {
            "result_id": res_id,
            "n_documents": n_documents,
            "total_tokens": total_tokens
        }
        
        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """Sample a subset of the working dataset without modifying original data.

        Args:
            n: Number of items to sample. Exactly one of n or frac must be specified.
            frac: Fraction of items to sample (0.0 to 1.0). Exactly one of n or frac must be specified.
            random_state: Seed for random number generator.

        Returns:
            Dict containing result_id, initial_rows, sampled_rows, sampled_indices (capped at 10), truncated flag, and parameters.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Call load_dataset_tool first."}

        # Argument validation: exactly one of n or frac must be specified
        if (n is None and frac is None) or (n is not None and frac is not None):
            return {"error": "Specify exactly one of 'n' or 'frac'."}

        initial_rows = len(session.dataframe)

        if n is not None:
            if n <= 0:
                return {"error": "'n' must be greater than 0."}
            if n > initial_rows:
                return {"error": f"'n' ({n}) cannot be greater than available rows ({initial_rows})."}

        if frac is not None:
            if frac <= 0.0 or frac > 1.0:
                return {"error": "'frac' must be between 0.0 and 1.0."}

        # Perform sampling on a copy per spec (does not mutate session.dataframe)
        df_copy = session.dataframe.copy()
        sampled_df = df_copy.sample(n=n, frac=frac, random_state=random_state)
        
        sampled_rows = len(sampled_df)
        sampled_indices_all = sampled_df.index.tolist()
        
        truncated = sampled_rows > 10
        sampled_indices = sampled_indices_all[:10]
        
        res_id = session.next_result_id("sample")
        summary = {
            "result_id": res_id,
            "initial_rows": initial_rows,
            "sampled_rows": sampled_rows,
            "sampled_indices": sampled_indices,
            "truncated": truncated,
            "params": {"n": n, "frac": frac, "random_state": random_state}
        }
        
        # Store full_report with the sampled DataFrame copy
        session.store_result("sample_data_tool", {"n": n, "frac": frac, "random_state": random_state}, summary, full_report=sampled_df)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """List files and subdirectories in a relative repository directory.

        Args:
            subdirectory: Relative path within the repository root (default is top-level root).

        Returns:
            Dict containing result_id, subdirectory, n_directories, n_files, directories (capped at 50), files (capped at 50), and truncated.
        """
        repo_root = Path.cwd().resolve()
        
        if Path(subdirectory).is_absolute():
            return {"error": "Absolute paths are not allowed."}

        target_path = (repo_root / subdirectory).resolve()

        # Check path traversal: target must be inside repo_root
        try:
            target_path.relative_to(repo_root)
        except ValueError:
            return {"error": "Path escapes the repository root."}

        if not target_path.exists():
            return {"error": f"Directory '{subdirectory}' does not exist."}

        if not target_path.is_dir():
            return {"error": f"Path '{subdirectory}' is not a directory."}

        all_directories = []
        all_files = []
        
        for item in sorted(target_path.iterdir()):
            if item.is_dir():
                all_directories.append(item.name)
            elif item.is_file():
                all_files.append(item.name)

        n_directories = len(all_directories)
        n_files = len(all_files)
        
        truncated = n_directories > 50 or n_files > 50
        
        directories = all_directories[:50]
        files = all_files[:50]

        res_id = session.next_result_id("list_files")
        summary = {
            "result_id": res_id,
            "subdirectory": subdirectory,
            "n_directories": n_directories,
            "n_files": n_files,
            "directories": directories,
            "files": files,
            "truncated": truncated
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> Dict[str, Any]:
        """Inspect the first few rows of the working DataFrame.

        Args:
            n: Number of rows to inspect (default 5, capped at 20).

        Returns:
            Dict containing result_id, total_rows, columns, rows_returned, and sample_rows (text truncated to 200 chars, lists shown as count).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Call load_dataset_tool first."}

        # Cap n between 1 and 20
        n_effective = min(max(1, n), 20)

        total_rows = len(session.dataframe)
        columns = session.dataframe.columns.tolist()

        head_df = session.dataframe.head(n_effective).copy()

        sample_rows = []
        for idx, row in head_df.iterrows():
            row_dict = {}
            for col in columns:
                val = row[col]
                # Convert list/collection values (e.g. unigrams) to item count string
                if isinstance(val, (list, tuple)):
                    val = f"<{len(val)} items>"
                # Truncate text values longer than 200 chars
                elif col == "text" and isinstance(val, str) and len(val) > 200:
                    val = val[:200] + "..."
                # Convert numpy types to native Python types for JSON serialization safety
                elif hasattr(val, "item"):
                    val = val.item()
                row_dict[col] = val
            sample_rows.append(row_dict)

        res_id = session.next_result_id("inspect")
        summary = {
            "result_id": res_id,
            "total_rows": total_rows,
            "columns": columns,
            "rows_returned": len(sample_rows),
            "sample_rows": sample_rows
        }

        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Describe text length statistics overall and per category, building a boxplot.

        Returns:
            Dict containing result_id, overall_stats, and per_category_stats.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Call load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"error": "Dataset does not contain a 'text' column."}

        # Compute text_length as character count per document matching Master
        session.dataframe["text_length"] = session.dataframe["text"].apply(lambda t: len(t) if isinstance(t, str) else 0)

        # Compute overall .describe() statistics
        overall_desc = session.dataframe["text_length"].describe().to_dict()
        overall_stats = {k: round(float(v), 4) for k, v in overall_desc.items()}

        # Compute per-category stats if category_name column is present
        per_category_stats = {}
        if "category_name" in session.dataframe.columns:
            for cat, group in session.dataframe.groupby("category_name"):
                cat_desc = group["text_length"].describe().to_dict()
                per_category_stats[str(cat)] = {k: round(float(v), 4) for k, v in cat_desc.items()}

        # Build boxplot of text_length grouped by category_name
        fig, ax = plt.subplots(figsize=(8, 5))
        if "category_name" in session.dataframe.columns:
            sns.boxplot(data=session.dataframe, x="category_name", y="text_length", ax=ax)
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=session.dataframe, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution")
        ax.set_ylabel("Text Length (characters)")
        
        session.pending_figure = fig
        plt.close(fig)

        res_id = session.next_result_id("describe")
        summary = {
            "result_id": res_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        tokenize_tool,
        sample_data_tool,
        list_files_tool,
        inspect_data_tool,
        describe_data_tool
    ]
