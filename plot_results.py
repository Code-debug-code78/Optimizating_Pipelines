import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List
from matplotlib.backends.backend_pdf import PdfPages # Import PdfPages

def plot_benchmark_results(metrics_file: str, predictions_file: str, output_dir: str = "./plots", pdf_output_filename: str = "benchmark_report.pdf"):
    """
    Generates various plots from benchmark metrics and predictions data,
    and saves them into a single PDF file.

    Args:
        metrics_file (str): Path to the CSV file containing benchmark metrics.
        predictions_file (str): Path to the CSV file containing predictions data.
        output_dir (str): Directory to save the PDF file.
        pdf_output_filename (str): Name of the output PDF file.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, pdf_output_filename)

    print(f"Loading metrics from: {metrics_file}")
    try:
        df_metrics = pd.read_csv(metrics_file)
    except FileNotFoundError:
        print(f"Error: Metrics file not found at {metrics_file}. Skipping metric plots.")
        df_metrics = pd.DataFrame()
    except Exception as e:
        print(f"Error loading metrics CSV: {e}. Skipping metric plots.")
        df_metrics = pd.DataFrame()

    print(f"Loading predictions from: {predictions_file}")
    try:
        # We still load predictions for the "Correct vs Incorrect" plot
        df_predictions = pd.read_csv(predictions_file)
    except FileNotFoundError:
        print(f"Error: Predictions file not found at {predictions_file}. Skipping prediction plots.")
        df_predictions = pd.DataFrame()
    except Exception as e:
        print(f"Error loading predictions CSV: {e}. Skipping prediction plots.")
        df_predictions = pd.DataFrame()

    # Create a PDF object to save the plots
    with PdfPages(pdf_path) as pdf:
        # --- 1. Top-1 Accuracy per Model (Bar Plot) ---
        if not df_metrics.empty and 'top1_accuracy' in df_metrics.columns and 'model_name' in df_metrics.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='model_name', y='top1_accuracy', data=df_metrics, hue='method', palette='viridis', ax=ax)
            ax.set_title('Top-1 Accuracy per Model')
            ax.set_xlabel('Model Name')
            ax.set_ylabel('Top-1 Accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print("Added Top-1 Accuracy plot to PDF.")
        else:
            print("Skipping Top-1 Accuracy plot: Metrics data is empty or missing required columns.")

        # --- NEW PLOT: Mean Reciprocal Rank (MRR) per Model (Bar Plot) ---
        if not df_metrics.empty and 'mrr_score' in df_metrics.columns and 'model_name' in df_metrics.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='model_name', y='mrr_score', data=df_metrics, hue='method', palette='cividis', ax=ax)
            ax.set_title('Mean Reciprocal Rank (MRR) per Model')
            ax.set_xlabel('Model Name')
            ax.set_ylabel('MRR Score')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print("Added Mean Reciprocal Rank (MRR) plot to PDF.")
        else:
            print("Skipping MRR plot: Metrics data is empty or missing 'mrr_score' or 'model_name' columns.")

        # --- 2. Average Inference Time per Model (Bar Plot) --- (was 2, now logically 3rd if MRR is 2nd)
        if not df_metrics.empty and 'avg_inference_time' in df_metrics.columns and 'model_name' in df_metrics.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='model_name', y='avg_inference_time', data=df_metrics, hue='method', palette='plasma', ax=ax)
            ax.set_title('Average Inference Time per Model')
            ax.set_xlabel('Model Name')
            ax.set_ylabel('Average Inference Time (seconds per query)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print("Added Average Inference Time plot to PDF.")
        else:
            print("Skipping Average Inference Time plot: Metrics data is empty or missing required columns.")

        # --- 3. Memory Usage per Model (Bar Plot) --- (was 3, now logically 4th)
        if not df_metrics.empty and 'memory_usage_mb' in df_metrics.columns and 'model_name' in df_metrics.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='model_name', y='memory_usage_mb', data=df_metrics, hue='method', palette='cubehelix', ax=ax) # Changed palette for variety
            ax.set_title('Memory Usage per Model')
            ax.set_xlabel('Model Name')
            ax.set_ylabel('Memory Usage (MB)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print("Added Memory Usage plot to PDF.")
        else:
            print("Skipping Memory Usage plot: Metrics data is empty or missing required columns.")

        # --- 4. Accuracy vs. Inference Time (Scatter Plot) --- (was 4, now logically 5th)
        if not df_metrics.empty and all(col in df_metrics.columns for col in ['avg_inference_time', 'top1_accuracy', 'memory_usage_mb', 'model_name']):
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(
                x='avg_inference_time',
                y='top1_accuracy',
                hue='model_name',
                size='memory_usage_mb',
                sizes=(50, 500),
                data=df_metrics,
                palette='tab10',
                legend='full',
                ax=ax
            )
            ax.set_title('Accuracy vs. Inference Time (Size indicates Memory Usage)')
            ax.set_xlabel('Average Inference Time (seconds per query)')
            ax.set_ylabel('Top-1 Accuracy')
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print("Added Accuracy vs. Inference Time plot to PDF.")
        else:
            print("Skipping Accuracy vs. Inference Time plot: Metrics data is empty or missing required columns.")

        # --- REMOVED: 5. Score Distribution per Model (Violin Plot) ---
        # This plot is intentionally removed as it was causing parsing issues with scores.
        print("Skipped Match Score Distribution plot as requested.")


        # --- 6. Correct vs Incorrect Predictions (Per Model) (Stacked Bar) --- (was 6, now logically 6th)
        if not df_predictions.empty and 'correct' in df_predictions.columns and 'model' in df_predictions.columns:
            df_predictions['correct_label'] = df_predictions['correct'].map({1: 'Correct', 0: 'Incorrect'})
            correct_incorrect_counts = df_predictions.groupby(['model', 'correct_label']).size().unstack(fill_value=0)

            fig, ax = plt.subplots(figsize=(10, 6))
            correct_incorrect_counts.plot(kind='bar', stacked=True, colormap='Paired', ax=ax)
            ax.set_title('Correct vs. Incorrect Predictions per Model')
            ax.set_xlabel('Model Name')
            ax.set_ylabel('Number of Predictions')
            plt.xticks(rotation=45, ha='right')
            ax.legend(title='Prediction Status')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print("Added Correct vs Incorrect Predictions plot to PDF.")
        else:
            print("Skipping Correct vs Incorrect Predictions plot: Predictions data is empty or missing required columns.")

    print(f"\nAll requested plots generated and saved to '{pdf_path}'.")


# Example usage (as provided by you):
if __name__ == "__main__":
    dataset_base_name = "amazon_itunes" # This should match the output name from benchmark_runner

    metrics_file_path = f"./results/{dataset_base_name}_metrics.csv"
    predictions_file_path = f"./results/{dataset_base_name}_predictions.csv"

    plots_output_directory = f"./plots/{dataset_base_name}_benchmark_plots"
    pdf_report_filename = f"{dataset_base_name}_benchmark_report.pdf"

    plot_benchmark_results(metrics_file_path, predictions_file_path, plots_output_directory, pdf_report_filename)