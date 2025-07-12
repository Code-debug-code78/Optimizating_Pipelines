# Save this as benchmark_runner.py in your project's root or a 'scripts' directory

import os
import pandas as pd
from typing import List, Dict, Tuple
import argparse # For command-line execution
import sys # Added for sys.path manipulation if needed

# Make sure 'alethia' and 'scoring' are accessible in your environment.
# Assuming 'alethia' is a module you have with 'create_recommendation_matrix'.
try:
    from alethia import create_recommendation_matrix
except ImportError:
    print("Warning: 'alethia' module not found. Using a dummy 'create_recommendation_matrix'.")
    def create_recommendation_matrix():
        # This is a dummy for demonstration. Replace with your actual model list.
        return {
            "fast_embedding": ["sentence-transformers/all-MiniLM-L6-v2", "rapidfuzz_embedding_dummy"],
            "general_purpose": ["sentence-transformers/paraphrase-MiniLM-L3-v2", "BAAI/bge-small-en-v1.5_dummy"],
            "large_models": ["sentence-transformers/distilbert-base-nli-stsb-mean-tokens", "microsoft/Phi-3-mini-4k-instruct_dummy"]
        }


# If scoring.py is not in the same directory as benchmark_runner.py, you'll need to add its path
# For example, if scoring.py is in 'alethia_root/scripts/scoring.py' and benchmark_runner.py is also there:
# from scoring import evaluate_and_save_results
# If scoring is a package/module:
# from alethia.scoring import evaluate_and_save_results
# For now, I'll assume it's directly importable or handled by your environment config.
from scoring import evaluate_and_save_results

def load_and_prepare_data(dataset_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load dataset and prepare queries, references, and candidate pool dynamically.
    Assumes the first column in tableA.csv and tableB.csv is the ID column.
    """
    print(f"Loading data from: {dataset_dir}")
    try:
        A = pd.read_csv(os.path.join(dataset_dir, "tableA.csv"))
        B = pd.read_csv(os.path.join(dataset_dir, "tableB.csv"))
        train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing dataset file in {dataset_dir}: {e}. Ensure tableA.csv, tableB.csv, and train.csv exist.")
    except Exception as e:
        raise ValueError(f"Error loading or parsing CSV files: {e}. Check file format and content.")

    train_pos = train[train.label == 1].copy()

    if train_pos.empty:
        print(f"Warning: No positive matches (label=1) found in train.csv for {dataset_dir}. Queries and references will be empty.")
        return [], [], []

    # Optimized data loading for queries and references using dictionaries for faster lookups
    # Set the ID columns as index for fast lookups
    A_indexed = A.set_index(A.columns[0])
    B_indexed = B.set_index(B.columns[0])

    queries_aligned = []
    references_aligned = []

    for _, row in train_pos.iterrows():
        ltable_id = row['ltable_id']
        rtable_id = row['rtable_id']

        # Query string from table A
        if ltable_id in A_indexed.index:
            query_str = " ".join(A_indexed.loc[ltable_id].astype(str).fillna('').tolist())
            queries_aligned.append(query_str)
        else:
            # Handle cases where ltable_id might not be found in A (though train.csv should ensure it is)
            queries_aligned.append("") # Or handle as an error
            print(f"Warning: ltable_id {ltable_id} not found in tableA.csv for dataset {dataset_dir}")


        # Reference string from table B
        if rtable_id in B_indexed.index:
            ref_str = " ".join(B_indexed.loc[rtable_id].astype(str).fillna('').tolist())
            references_aligned.append(ref_str)
        else:
            # Handle cases where rtable_id might not be found in B (though train.csv should ensure it is)
            references_aligned.append("") # Or handle as an error
            print(f"Warning: rtable_id {rtable_id} not found in tableB.csv for dataset {dataset_dir}")


    # Create candidate pool from all rows in table B, concatenating all columns except the first (ID)
    candidate_pool = [" ".join(row.astype(str).fillna('').tolist()) for _, row in B.iloc[:, 1:].iterrows()]

    print(f"Data loaded: {len(queries_aligned)} queries, {len(references_aligned)} references, {len(candidate_pool)} candidates.")
    return queries_aligned, references_aligned, candidate_pool

def run_benchmark_on_dataset(dataset_dir: str, model_category: str):
    """
    Runs the benchmark for a specified dataset and model category.

    Args:
        dataset_dir (str): Path to the directory containing dataset files (tableA.csv, tableB.csv, train.csv).
        model_category (str): The category of models to benchmark (e.g., "fast_embedding", "general_purpose").
    """
    print(f"\n--- Starting benchmark for Dataset: {os.path.basename(dataset_dir)}, Model Category: {model_category} ---")
    queries, references, candidate_pool = load_and_prepare_data(dataset_dir)

    if not queries or not references or not candidate_pool:
        print("Skipping benchmark due to empty queries, references, or candidate pool after data loading.")
        print(f"--- Benchmark for {os.path.basename(dataset_dir)} finished (SKIPPED) ---")
        return

    # Paths
    dataset_name = os.path.basename(dataset_dir)
    predictions_path = f"./results/{dataset_name}_{model_category}_alethia_results.csv"
    metrics_path = f"./results/{dataset_name}_{model_category}_alethia_accuracy.csv"
    os.makedirs("./results", exist_ok=True)

    all_metrics: List[Dict[str, float]] = []
    all_predictions_dfs: List[pd.DataFrame] = []

    # Run Baseline (RapidFuzz)
    print("\n🔍 Evaluating baseline fuzzy matcher (RapidFuzz)...")
    try:
        baseline_metrics, baseline_preds_df = evaluate_and_save_results(
            queries=queries,
            references=references,
            candidate_pool=candidate_pool,
            model_name="rapidfuzz", # Model name for metrics
            use_fuzzy=True
        )
        all_metrics.append(baseline_metrics)
        all_predictions_dfs.append(baseline_preds_df)
        print("RapidFuzz evaluation completed.")
    except Exception as e:
        print(f"Error during RapidFuzz evaluation: {e}")
        # Optionally, decide if you want to stop or continue without baseline


    # Run Embedding Models
    print("\n🚀 Evaluating embedding models...")
    recommendation_dict = create_recommendation_matrix()
    model_list = recommendation_dict.get(model_category, [])

    if not model_list:
        print(f"No models found for category: {model_category}. Skipping embedding model evaluation.")

    for model_name in model_list:
        print(f"\n⚙️ Evaluating model: {model_name}")
        try:
            metrics, preds_df = evaluate_and_save_results(
                queries=queries,
                references=references,
                candidate_pool=candidate_pool,
                model_name=model_name,
                use_fuzzy=False
            )
            all_metrics.append(metrics)
            all_predictions_dfs.append(preds_df)
            print(f"Evaluation for {model_name} completed.")
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            print(f"Skipping {model_name} due to error.")
            # Continue to next model even if one fails

    # Save all predictions to a single CSV
    if all_predictions_dfs:
        final_predictions_df = pd.concat(all_predictions_dfs, ignore_index=True)
        final_predictions_df.to_csv(predictions_path, index=False)
        print(f"\n📄 All predictions saved to: {predictions_path}")
    else:
        print("\nNo predictions generated to save.")


    # Save Accuracy + Inference Time + Memory Usage
    if all_metrics:
        final_metrics_df = pd.DataFrame(all_metrics)
        final_metrics_df['dataset_name'] = dataset_name # Add dataset_name column for plotting
        final_metrics_df.to_csv(metrics_path, index=False)
        print(f"📊 All metrics saved to: {metrics_path}")
    else:
        print("\nNo metrics generated to save.")


    print(f"\n✅ Benchmark completed for {os.path.basename(dataset_dir)} and {model_category}.")
    print(f"--- Benchmark for {os.path.basename(dataset_dir)} finished ---")


# This block allows the script to be run directly from the command line
# e.g., python benchmark_runner.py --dataset_dir "./data/amazon_itunes" --model_category "fast_embedding"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Alethia benchmarks on datasets.")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the dataset directory (e.g., './data/amazon_itunes').")
    parser.add_argument("--model_category", type=str, required=True,
                        choices=list(create_recommendation_matrix().keys()), # Ensures valid category
                        help="Category of models to benchmark (e.g., 'fast_embedding', 'general_purpose').")

    args = parser.parse_args()

    # Add sys.path for modules like 'alethia' and 'scoring' if they are not in standard locations
    # For example, if your structure is:
    # project_root/
    # ├── scripts/
    # │   ├── benchmark_runner.py
    # │   ├── plot_results.py
    # │   └── scoring.py
    # └── alethia_package/
    #     └── __init__.py (containing create_recommendation_matrix)
    #
    # You might need:
    # import sys
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../alethia_package')))
    # sys.path.insert(0, os.path.abspath(os.path.dirname(__file__))) # For scoring.py if it's in the same dir

    run_benchmark_on_dataset(args.dataset_dir, args.model_category)