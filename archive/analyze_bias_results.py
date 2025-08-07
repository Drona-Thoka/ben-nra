import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# --- Bias Definitions ---
COGNITIVE_BIASES_LIST = [
    "recency", "frequency", "false_consensus", "status_quo", "confirmation", 
    "availability", "premature_closure", "diagnosis_momentum", "gamblers_fallacy", 
    "overconfidence", "omission", "representativeness", "commission", "sunk_cost", 
    "affective", "aggregate", "anchoring", "bandwagon", "outcome", 
    "vertical_line_failure", "zebra_retreat", "suttons_slip"
]

DEMOGRAPHIC_BIASES_LIST = [
    "race", "sexual_orientation", "cultural", "education", "religion", 
    "socioeconomic", "gender", "age", "disability", "weight", "mental_health"
]

def load_all_results(logs_dir="logs"):
    """Load all bias testing results from log files"""
    results = []
    log_files = glob.glob(os.path.join(logs_dir, "*_log.json"))
    
    for log_file in log_files:
        # Extract dataset and bias from filename
        filename = os.path.basename(log_file)
        parts = filename.split('_log.json')[0].split('_', 1)
        
        if len(parts) < 2:
            print(f"Skipping malformatted filename: {filename}")
            continue
            
        dataset, bias = parts
        
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    entry['dataset'] = dataset
                    entry['bias'] = bias
                    results.append(entry)
        except Exception as e:
            print(f"Error loading {log_file}: {e}")
    
    return results

def calculate_bias_impact(results):
    """Calculate the impact of each bias on diagnostic accuracy and other metrics"""
    # Extract nested fields to top-level for aggregation
    for entry in results:
        ca = entry.get('consultation_analysis', {})
        entry['diagnoses_considered_count'] = ca.get('diagnoses_considered_count')
        entry['disagreements'] = ca.get('disagreements')
    # Create a DataFrame
    df = pd.DataFrame(results)

    # Group by dataset and bias
    grouped = df.groupby(['dataset', 'bias']).agg({
        'is_correct': ['count', 'sum', 'mean'],
        'tests_requested_count': ['mean', 'std'],
        'diagnoses_considered_count': ['mean', 'std'],
        'disagreements': ['mean', 'std']
    }).reset_index()
    
    # Flatten the MultiIndex columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Calculate accuracy
    grouped['accuracy'] = grouped['is_correct_sum'] / grouped['is_correct_count'] * 100
    
    # Calculate impact compared to no bias (baseline) for multiple metrics
    comparison_data = []
    metrics_to_compare = {
        'accuracy': 'accuracy', # This is already a percentage
        'tests_requested_count': 'tests_requested_count_mean',
        'diagnoses_considered_count': 'diagnoses_considered_count_mean',
        'disagreements': 'disagreements_mean'
    }
    
    for dataset in df['dataset'].unique():
        baseline_row = grouped[(grouped['dataset'] == dataset) & (grouped['bias'] == 'none')]
        
        if baseline_row.empty:
            print(f"Warning: No 'none' bias baseline found for dataset {dataset}. Skipping impact calculation for this dataset.")
            continue
            
        baseline_metrics = baseline_row.iloc[0]
        
        for _, row in grouped[(grouped['dataset'] == dataset) & (grouped['bias'] != 'none')].iterrows():
            entry_data = {
                'dataset': dataset,
                'bias': row['bias'],
                'samples': row['is_correct_count']
            }
            for metric_key, col_name in metrics_to_compare.items():
                baseline_value = baseline_metrics[col_name]
                biased_value = row[col_name]
                impact_value = biased_value - baseline_value
                
                entry_data[f'baseline_{metric_key}'] = baseline_value
                entry_data[f'biased_{metric_key}'] = biased_value
                entry_data[f'{metric_key}_impact'] = impact_value
            
            comparison_data.append(entry_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    return grouped, comparison_df

def plot_bias_impact(comparison_df, output_dir="logs"):
    """Create visualizations of bias impact"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")

    if comparison_df.empty:
        print("No comparison data to plot (e.g., no biased runs found or no 'none' baseline for comparison).")
        # Save an empty df if it's expected, or just return
        comparison_df.to_csv(os.path.join(output_dir, 'detailed_bias_comparison.csv'), index=False)
        return

    # Plot bias impact by dataset (for accuracy) - Individual dataset plots
    for dataset in comparison_df['dataset'].unique():
        dataset_df = comparison_df[comparison_df['dataset'] == dataset].sort_values('accuracy_impact')
        
        if dataset_df.empty:
            continue

        plt.figure(figsize=(12, 10))
        ax = sns.barplot(x='accuracy_impact', y='bias', data=dataset_df, 
                          palette=sns.color_palette("coolwarm", len(dataset_df)))
        
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
        plt.title(f'Impact of Biases on Diagnostic Accuracy - {dataset} Dataset', fontsize=16)
        plt.xlabel('Accuracy Impact (percentage points)', fontsize=14)
        plt.ylabel('Bias Type', fontsize=14)
        
        # Add value labels
        for i, v in enumerate(dataset_df['accuracy_impact']):
            ax.text(v + (0.5 if v < 0 else -0.5), i, f"{v:.1f}pp", 
                    color='black', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'bias_impact_{dataset}.png'))
        plt.close()

    # Visualizations for other metrics - Individual dataset plots (excluding disagreements)
    metrics_for_individual_plots = {
        'tests_requested_count': 'Tests Requested Count',
        'diagnoses_considered_count': 'Diagnoses Considered Count',
        # 'disagreements': 'Disagreements' # Removed as per request
    }
    for metric_key, metric_label in metrics_for_individual_plots.items():
        for dataset in comparison_df['dataset'].unique():
            dataset_df = comparison_df[comparison_df['dataset'] == dataset].sort_values(f'{metric_key}_impact')
            if dataset_df.empty:
                continue

            plt.figure(figsize=(12, 10))
            ax = sns.barplot(
                x=f'{metric_key}_impact',
                y='bias',
                data=dataset_df,
                palette=sns.color_palette("coolwarm", len(dataset_df))
            )

            plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
            plt.title(f'Impact of Biases on {metric_label} - {dataset} Dataset', fontsize=16)
            plt.xlabel(f'{metric_label} Impact', fontsize=14)
            plt.ylabel('Bias Type', fontsize=14)

            # Add value labels
            for i, v in enumerate(dataset_df[f'{metric_key}_impact']):
                ax.text(
                    v + (0.5 if v < 0 else -0.5), # Adjusted offset logic slightly
                    i,
                    f"{v:.1f}",
                    color='black',
                    va='center',
                    fontweight='bold'
                )

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'bias_{metric_key}_impact_{dataset}.png'))
            plt.close()

    # New: Combined dataset plots (Cognitive vs Implicit biases)
    metrics_for_combined_plots = {
        'accuracy': 'Diagnostic Accuracy',
        'tests_requested_count': 'Tests Requested Count',
        'diagnoses_considered_count': 'Diagnoses Considered Count'
    }
    
    bias_groups = {
        'cognitive': COGNITIVE_BIASES_LIST,
        'implicit': DEMOGRAPHIC_BIASES_LIST
    }

    for group_name, bias_list in bias_groups.items():
        for metric_key, metric_label in metrics_for_combined_plots.items():
            # Filter for the current bias group and ensure the impact column exists
            if f'{metric_key}_impact' not in comparison_df.columns:
                print(f"Warning: Metric impact column '{metric_key}_impact' not found in comparison_df. Skipping plot for {group_name} - {metric_label}.")
                continue

            plot_df = comparison_df[comparison_df['bias'].isin(bias_list)].copy() # Use .copy() to avoid SettingWithCopyWarning

            if plot_df.empty:
                print(f"No data for {group_name} biases and metric {metric_label}. Skipping plot.")
                continue
            
            # Sort biases for consistent plotting order
            # Ensure 'bias' column is of a type that can be sorted, e.g., string
            plot_df['bias'] = plot_df['bias'].astype(str)
            sorted_biases = sorted(plot_df['bias'].unique())


            plt.figure(figsize=(9, 12)) # Dynamic height, adjusted for taller plots
            
            ax = sns.barplot(
                x=f'{metric_key}_impact',
                y='bias',
                hue='dataset',
                data=plot_df,
                order=sorted_biases,
                palette="viridis" 
            )

            plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
            title = f'Impact of {group_name.capitalize()} Biases on {metric_label}\n(Combined Datasets)'
            plt.title(title, fontsize=16)
            
            x_label = f'{metric_label} Impact'
            if metric_key == 'accuracy':
                x_label += ' (percentage points)'
            plt.xlabel(x_label, fontsize=14)
            plt.ylabel('Bias Type', fontsize=14)
            
            plt.legend(title='Dataset', loc='best') # Ensure legend is shown

            # Add value labels to bars
            for container in ax.containers:
                try:
                    ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3, fontsize=9, fontweight='bold')
                except AttributeError: # Older matplotlib might not have ax.containers or bar_label options
                    print("Skipping bar labels due to matplotlib version or container issue.")


            plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to make space for legend if needed
            
            plot_filename = f'{group_name}_biases_{metric_key}_impact_combined_datasets.png'
            plt.savefig(os.path.join(output_dir, plot_filename))
            plt.close()
            print(f"Saved combined plot: {plot_filename}")

    # Save the detailed comparison DataFrame
    comparison_df.to_csv(os.path.join(output_dir, 'detailed_bias_comparison.csv'), index=False)
    return

def main():
    print("Analyzing bias testing results...")
    results = load_all_results()
    print(f"Loaded {len(results)} scenario results from log files")
    
    if not results:
        print("No results found. Make sure you've run the bias testing first.")
        return
    
    grouped_df, comparison_df = calculate_bias_impact(results)

    if grouped_df is not None and not grouped_df.empty:
        grouped_df.to_csv(os.path.join("logs", 'grouped_bias_metrics.csv'), index=False)
        print(f"\nRaw aggregated metrics saved to logs/grouped_bias_metrics.csv")

    if comparison_df is not None and not comparison_df.empty:
        plot_bias_impact(comparison_df)
        print(f"\nDetailed comparison against baseline saved to logs/detailed_bias_comparison.csv")
        print("\nBias Impact Analysis (Accuracy):")
        # Display a small part of the comparison_df for quick view, if needed
        # For example, accuracy impact per bias, averaged over datasets
        if 'accuracy_impact' in comparison_df.columns:
            avg_accuracy_impact = comparison_df.groupby('bias')['accuracy_impact'].mean().sort_values()
            print(avg_accuracy_impact)
    else:
        print("\nNo comparison data generated (e.g., only baseline 'none' runs, or no 'none' baseline found).")
    
    print("\nAnalysis complete! Visualizations and data saved to the logs directory.")

if __name__ == "__main__":
    main()
