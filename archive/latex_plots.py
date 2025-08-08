import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# --- Style Settings for LaTeX Serif Fonts + Clean Look ---
# Call sns.set() before updating rcParams to ensure custom settings are not overridden
sns.set(style="whitegrid")

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# --- Bias Groups ---
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

# --- Cognitive Bias Subgroups ---
COGNITIVE_RECALL_BIASES = [] # Recency moved to Estimation
COGNITIVE_ESTIMATION_BIASES = [
    "availability", "affective", "anchoring", "false_consensus", 
    "frequency", "gamblers_fallacy", "overconfidence", "recency" # Added recency
]
COGNITIVE_HYPOTHESIS_ASSESSMENT_BIASES = [
    "aggregate", "confirmation", "diagnosis_momentum", "premature_closure", 
    "representativeness", "suttons_slip", "vertical_line_failure", "zebra_retreat"
]
COGNITIVE_DECISION_BIASES = [
    "commission", "omission", "outcome", "status_quo", "sunk_cost", "bandwagon" # Added bandwagon
]
COGNITIVE_OPINION_REPORTING_BIASES = [] # Bandwagon moved to Decision

COGNITIVE_SUBGROUPS = {
    # "Recall" is removed as COGNITIVE_RECALL_BIASES is empty
    "Estimation": {"key": "estimation", "biases": COGNITIVE_ESTIMATION_BIASES},
    "Hypothesis Assessment": {"key": "hypothesis_assessment", "biases": COGNITIVE_HYPOTHESIS_ASSESSMENT_BIASES},
    "Decision": {"key": "decision", "biases": COGNITIVE_DECISION_BIASES},
    # "Opinion Reporting" is removed as COGNITIVE_OPINION_REPORTING_BIASES is empty
}

# --- Bias to Category Mapping ---
_BIAS_TO_CATEGORY_MAP = {}
for cat_display_name, group_info in COGNITIVE_SUBGROUPS.items():
    for bias_key in group_info["biases"]:
        _BIAS_TO_CATEGORY_MAP[bias_key] = cat_display_name
for bias_key in DEMOGRAPHIC_BIASES_LIST:
    _BIAS_TO_CATEGORY_MAP[bias_key] = "Implicit"

# Define an explicit order for categories for consistent plotting
ORDERED_BIAS_CATEGORIES = [
    # "Recall" removed
    "Estimation", "Hypothesis Assessment", "Decision", 
    # "Opinion Reporting" removed
    "Implicit"
]


# --- Bias Display Names ---
BIAS_DISPLAY_NAMES = {
    # Cognitive Biases
    "recency": "Recency", "frequency": "Frequency", "false_consensus": "False Consensus",
    "status_quo": "Status Quo", "confirmation": "Confirmation", "availability": "Availability",
    "premature_closure": "Premature Closure", "diagnosis_momentum": "Diagnosis Momentum",
    "gamblers_fallacy": "Gambler's Fallacy", "overconfidence": "Overconfidence",
    "omission": "Omission", "representativeness": "Representativeness",
    "commission": "Commission", "sunk_cost": "Sunk Cost", "affective": "Affective",
    "aggregate": "Aggregate", "anchoring": "Anchoring", "bandwagon": "Bandwagon",
    "outcome": "Outcome", "vertical_line_failure": "Vertical Line Failure",
    "zebra_retreat": "Zebra Retreat", "suttons_slip": "Sutton's Slip",
    # Demographic Biases
    "race": "Race", "sexual_orientation": "Sexual Orientation", "cultural": "Cultural",
    "education": "Education", "religion": "Religion", "socioeconomic": "Socioeconomic",
    "gender": "Gender", "age": "Age", "disability": "Disability", "weight": "Weight",
    "mental_health": "Mental Health",
    "none": "None" # For baseline, though not plotted directly as a bias
}


def load_all_results(logs_dir="logs"):
    results = []
    log_files = glob.glob(os.path.join(logs_dir, "*_log.json"))
    for log_file in log_files:
        filename = os.path.basename(log_file)
        parts = filename.split('_log.json')[0].split('_', 1)
        if len(parts) < 2:
            continue
        dataset, bias = parts
        with open(log_file, 'r') as f:
            data = json.load(f)
            for entry in data:
                entry['dataset'] = dataset
                entry['bias'] = bias
                results.append(entry)
    return results

def calculate_bias_impact(results):
    for entry in results:
        ca = entry.get('consultation_analysis', {})
        entry['diagnoses_considered_count'] = ca.get('diagnoses_considered_count')
        entry['disagreements'] = ca.get('disagreements')

    df = pd.DataFrame(results)
    grouped = df.groupby(['dataset', 'bias']).agg({
        'is_correct': ['count', 'sum', 'mean'],
        'tests_requested_count': ['mean', 'std'],
        'diagnoses_considered_count': ['mean', 'std'],
        'disagreements': ['mean', 'std']
    }).reset_index()

    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped['accuracy'] = grouped['is_correct_sum'] / grouped['is_correct_count'] * 100

    comparison_data = []
    metrics_to_compare = {
        'accuracy': 'accuracy',
        'tests_requested_count': 'tests_requested_count_mean',
        'diagnoses_considered_count': 'diagnoses_considered_count_mean',
    }

    for dataset in df['dataset'].unique():
        baseline_row = grouped[(grouped['dataset'] == dataset) & (grouped['bias'] == 'none')]
        if baseline_row.empty:
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
                entry_data[f'{metric_key}_impact'] = impact_value
            comparison_data.append(entry_data)

    return pd.DataFrame(comparison_data)

def get_bias_category(bias_name):
    """Maps a bias name to its category."""
    return _BIAS_TO_CATEGORY_MAP.get(bias_name, "Unknown")

def aggregate_impact_by_category(comparison_df):
    """Aggregates bias impact data by category."""
    if comparison_df.empty:
        return pd.DataFrame()

    df_with_categories = comparison_df.copy()
    df_with_categories['bias_category'] = df_with_categories['bias'].apply(get_bias_category)

    # Filter out any 'Unknown' categories if they arise
    df_with_categories = df_with_categories[df_with_categories['bias_category'] != "Unknown"]

    if df_with_categories.empty:
        return pd.DataFrame()

    impact_cols = [col for col in df_with_categories.columns if '_impact' in col]
    grouping_cols = ['dataset', 'bias_category']
    
    agg_dict = {impact_col: 'mean' for impact_col in impact_cols}
    
    category_summary_df = df_with_categories.groupby(grouping_cols).agg(agg_dict).reset_index()
    
    return category_summary_df

def plot_category_summary_impact(category_summary_df, output_dir="figures"):
    """Plots the aggregated impact of bias categories."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        'accuracy': 'Accuracy Impact (\%)', # Changed from (pp)
        'tests_requested_count': 'Tests Requested Impact',
        'diagnoses_considered_count': 'Diagnoses Considered Impact'
    }

    # Ensure 'bias_category' is treated as a categorical type with the defined order
    # This helps in sorting the y-axis of the plots correctly.
    category_summary_df['bias_category'] = pd.Categorical(
        category_summary_df['bias_category'],
        categories=ORDERED_BIAS_CATEGORIES,
        ordered=True
    )
    # Sort by the categorical order to ensure plots are consistent
    category_summary_df = category_summary_df.sort_values(by=['bias_category', 'dataset'])


    for metric_key, label in metrics.items():
        col = f'{metric_key}_impact'
        if col not in category_summary_df.columns:
            print(f"Metric column {col} not found in category summary data. Skipping plot.")
            continue
        
        df_metric = category_summary_df[category_summary_df[col].notnull()].copy()

        if df_metric.empty:
            print(f"No data for metric {metric_key} in category summary. Skipping plot.")
            continue
        
        plt.figure(figsize=(6, 5.2)) # Adjusted for potentially more categories
        ax = sns.barplot(
            x=col,
            y='bias_category',
            hue='dataset',
            data=df_metric,
            # order parameter is not needed if 'bias_category' is categorical and sorted
            palette='viridis' 
        )

        plt.axvline(x=0, color='black', linestyle='--', lw=1, alpha=0.7)
        # plt.title(f'Bias Category Summary: {label}', fontsize=11) # Removed title
        plt.xlabel(label)
        plt.ylabel('Bias Category') # Explicitly set Y-axis label

        ax.legend(loc='best', title='Dataset', frameon=False)
        plt.tight_layout()
        filename = os.path.join(output_dir, f'category_summary_{metric_key}_impact.pdf')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

def plot_bias_impact(comparison_df, bias_list, group_name, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        'accuracy': 'Accuracy Impact (\%)', # Changed from (pp)
        'tests_requested_count': 'Tests Requested',
        'diagnoses_considered_count': 'Diagnoses Considered'
    }

    for metric_key, label in metrics.items():
        col = f'{metric_key}_impact'
        df = comparison_df[
            (comparison_df['bias'].isin(bias_list)) &
            (comparison_df[col].notnull())
        ].copy()

        if df.empty:
            continue

        # Map internal bias names to display names
        df['bias_display_name'] = df['bias'].map(BIAS_DISPLAY_NAMES).fillna(df['bias'])
        
        # Sort biases by their display names
        sorted_bias_display_names = sorted(df['bias_display_name'].unique())

        plt.figure(figsize=(5.2, 5.2))  # Small, publishable size
        ax = sns.barplot(
            x=col,
            y='bias_display_name', # Use display name for y-axis
            hue='dataset',
            data=df,
            order=sorted_bias_display_names, # Order by display name
            palette='viridis'
        )

        plt.axvline(x=0, color='black', linestyle='--', lw=1, alpha=0.7)
        # Use group_name directly as it will be more descriptive now
        # plt.title(f'{group_name} Biases: {label}', fontsize=11) # Removed title
        plt.xlabel(label)
        plt.ylabel('')

        ax.legend(loc='best', title='Dataset', frameon=False)
        plt.tight_layout()
        filename = os.path.join(output_dir, f'{group_name}_{metric_key}_impact.pdf')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

def main():
    results = load_all_results()
    if not results:
        print("No log files found.")
        return

    comparison_df = calculate_bias_impact(results)
    if comparison_df.empty:
        print("No comparison data.")
        return

    base_output_dir = "figures"

    # --- Individual Bias Plots ---
    individual_plots_dir = os.path.join(base_output_dir, "individual_bias_impact")
    
    print("Generating all cognitive bias plots...")
    plot_bias_impact(comparison_df, COGNITIVE_BIASES_LIST, "Cognitive (All)", os.path.join(individual_plots_dir, "cognitive_all"))

    print("Generating implicit bias plots...")
    plot_bias_impact(comparison_df, DEMOGRAPHIC_BIASES_LIST, "Implicit", os.path.join(individual_plots_dir, "implicit_all"))

    print("Generating cognitive bias subgroup plots...")
    cognitive_subgroups_base_dir = os.path.join(individual_plots_dir, "cognitive_subgroups")
    for display_name, group_info in COGNITIVE_SUBGROUPS.items():
        subgroup_key = group_info["key"]
        subgroup_biases = group_info["biases"]
        
        relevant_biases_in_df = [
            b for b in subgroup_biases if b in comparison_df['bias'].unique()
        ]
        if not relevant_biases_in_df:
            print(f"Skipping subgroup {display_name} as no relevant data found.")
            continue

        plot_group_name = f"Cognitive {display_name}" 
        subgroup_output_dir = os.path.join(cognitive_subgroups_base_dir, f"cognitive_{subgroup_key}")
        
        print(f"Generating plots for {plot_group_name} in {subgroup_output_dir}...")
        plot_bias_impact(comparison_df, relevant_biases_in_df, plot_group_name, subgroup_output_dir)

    # --- Category Level Plots ---
    print("Aggregating impact by category...")
    category_summary_df = aggregate_impact_by_category(comparison_df)

    if not category_summary_df.empty:
        category_comparison_plot_dir = os.path.join(base_output_dir, "category_summary_impact")
        print(f"Generating category summary plots in {category_comparison_plot_dir}...")
        plot_category_summary_impact(category_summary_df, category_comparison_plot_dir)
    else:
        print("No data for category summary, skipping category-level plots.")

    print(f"All plots saved to '{base_output_dir}/' and its subfolders.")

if __name__ == "__main__":
    main()
