import wandb
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional


def fetch_run_history(
    entity: str,
    project: str,
    run_id: str,
    keys: List[str] = ["val_MSE", "_step"]
) -> pd.DataFrame:
    
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # Fetch the history
    history = run.history(keys=keys)
    
    return history


def fetch_multiple_runs(
    entity: str,
    project: str,
    run_ids: List[str],
    metric: str = "val_MSE"
) -> Dict[str, pd.DataFrame]:
    
    runs_data = {}
    api = wandb.Api()
    
    for run_id in run_ids:
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
            history = run.history(keys=[metric, "_step"])
            runs_data[run_id] = {
                "history": history,
                "name": run.name,
                "config": run.config
            }
            print(f"✓ Fetched data for run: {run.name} ({run_id})")
        except Exception as e:
            print(f"✗ Error fetching run {run_id}: {e}")
    
    return runs_data


def plot_training_curves(
    runs_data: Dict[str, Dict],
    metric: str = "val_MSE",
    run_labels: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    xlabel: str = "Step",
    ylabel: str = "Validation MSE",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    plt.figure(figsize=figsize)
    
    for run_id, data in runs_data.items():
        history = data["history"]
        
        # Get label for this run
        if run_labels and run_id in run_labels:
            label = run_labels[run_id]
        else:
            label = data["name"]
        
        # Plot the metric
        if metric in history.columns and "_step" in history.columns:
            # Filter out NaN values
            valid_data = history[[metric, "_step"]].dropna()
            plt.plot(valid_data["_step"], valid_data[metric], label=label, marker='o', markersize=3, alpha=0.7)
        else:
            print(f"Warning: {metric} not found in run {run_id}")
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title or "Training Curves Comparison", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def compare_runs(
    entity: str = "carlahugod-danmarks-tekniske-universitet-dtu",
    project: str = "GNN_semi_supervised",
    run_ids: List[str] = None,
    metric: str = "val_MSE",
    run_labels: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    if run_ids is None:
        raise ValueError("Please provide a list of run_ids")
    
    print(f"Fetching data for {len(run_ids)} runs...")
    runs_data = fetch_multiple_runs(entity, project, run_ids, metric)
    
    if runs_data:
        print(f"\nPlotting {len(runs_data)} runs...")
        plot_training_curves(
            runs_data,
            metric=metric,
            run_labels=run_labels,
            title=title,
            save_path=save_path,
            show_plot=show_plot
        )
    else:
        print("No data fetched. Please check your run IDs and credentials.")


def compare_runs_by_category(
    entity: str = "carlahugod-danmarks-tekniske-universitet-dtu",
    project: str = "GNN_semi_supervised",
    run_groups: Dict[str, Dict[str, str]] = None,
    metric: str = "val_MSE",
    output_dir: str = "outputs",
    show_plots: bool = True
):
    """
    Create separate plots for each hyperparameter category.
    
    Args:
        entity: wandb entity name
        project: wandb project name
        run_groups: Dictionary mapping category names to {run_id: label} dictionaries
                   Example: {
                       "batch_size": {"run1": "BS=32", "run2": "BS=64"},
                       "optimizer": {"run3": "Adam", "run4": "SGD"}
                   }
        metric: Metric to plot
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    if run_groups is None:
        raise ValueError("Please provide run_groups dictionary")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create overall comparison with all runs
    print("\n" + "="*50)
    print("Creating overall comparison plot...")
    print("="*50)
    
    all_run_ids = []
    all_labels = {}
    for category, runs in run_groups.items():
        all_run_ids.extend(runs.keys())
        all_labels.update(runs)
    
    compare_runs(
        entity=entity,
        project=project,
        run_ids=all_run_ids,
        metric=metric,
        run_labels=all_labels,
        title=f"All Experiments - {metric}",
        save_path=f"{output_dir}/all_runs_comparison.png",
        show_plot=show_plots
    )
    
    # Create individual plots for each category
    for category, runs in run_groups.items():
        print("\n" + "="*50)
        print(f"Creating plot for: {category}")
        print("="*50)
        
        compare_runs(
            entity=entity,
            project=project,
            run_ids=list(runs.keys()),
            metric=metric,
            run_labels=runs,
            title=f"{category.replace('_', ' ').title()} Comparison - {metric}",
            save_path=f"{output_dir}/{category}_comparison.png",
            show_plot=show_plots
        )
    
    print("\n" + "="*50)
    print(f"All plots saved to: {output_dir}/")
    print("="*50)


# Example usage
if __name__ == "__main__":
    # Define run groups by hyperparameter category
    run_groups = {
        "batch_size": {
            "yq0bj14f": "BatchSize=256",
            "oa94im9q": "BatchSize=128 ✓", # chosen
            "ksn7m63w": "BatchSize=32",  # original
        },
        "optimizer": {
            "771iuvv2": "SGD",
            "yjuu1ngw": "Adam",
            "04ftmt8x": "AdamW ✓", # chosen, original
        },
    }
    
    # Add baseline to all groups for comparison
    # TODO make new baseline run
    baseline_id = "4paozhos"
    baseline_label = "Baseline"
    for category in run_groups:
        if baseline_id not in run_groups[category]:
            run_groups[category][baseline_id] = baseline_label
    
    # Create all plots (overall + per category)
    compare_runs_by_category(
        run_groups=run_groups,
        metric="val_MSE",
        output_dir="outputs/hyperparameter_comparison",
        show_plots=True
    )