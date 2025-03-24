import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_csv_metrics(csv_files, output_file='metrics_comparison.png'):
    """
    Plot bar graphs for specified metrics across multiple CSV files and save to a file.
    
    Parameters:
    -----------
    csv_files : list
        List of CSV file paths to process
    output_file : str, optional
        Filename to save the plot (default: 'metrics_comparison.png')
    """
    # Metrics to plot
    metrics = [
        'retrieval_score_mean_umbrela_score', 
        'generation_score_vital_nuggetizer_score', 
        'generation_score_hallucination_scores', 
        'generation_score_citation_f1_score'
    ]
    
    # Set up the plot
    plt.figure(figsize=(16, 10))
    
    # Color palette for visual appeal
    colors = plt.cm.Spectral(np.linspace(0, 1, len(csv_files)))
    
    # Width of each bar
    bar_width = 0.8 / len(csv_files)
    
    # Iterate through metrics
    for metric_idx, metric in enumerate(metrics):
        # Subplot for each metric
        plt.subplot(2, 2, metric_idx + 1)
        
        # Collect values for this metric
        metric_values = []
        
        # Iterate through CSV files
        for file_idx, csv_file in enumerate(csv_files):
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Calculate mean of the metric
            mean_value = df[metric].mean()
            metric_values.append(mean_value)
            
            # Plot bar with label and color
            plt.bar(
                file_idx, 
                mean_value, 
                width=bar_width, 
                color=colors[file_idx], 
                label=f'{os.path.basename(csv_file)} ({mean_value:.4f})'
            )
        
        # Customize subplot
        plt.title(f'Mean {metric.replace("_", " ").title()}', fontsize=12)
        plt.ylabel('Mean Value', fontsize=10)
        plt.xticks([])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.grid(axis='y', which='minor', linestyle=':', alpha=0.4)
        
        # Add value labels on bars
        for i, v in enumerate(metric_values):
            plt.text(
                i, 
                v, 
                f'{v:.4f}', 
                ha='center', 
                va='bottom',
                fontweight='bold'
            )
        
        # Add legend
        plt.legend(title='CSV Files', loc='best', bbox_to_anchor=(1, 1), fontsize=8)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()  # Close the figure to free up memory
    
    print(f"Graph saved to {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot metrics from CSV files')
    parser.add_argument(
        'csv_files', 
        nargs='+', 
        help='List of CSV files to process'
    )
    parser.add_argument(
        '-o', '--output', 
        default='metrics_comparison.png', 
        help='Output filename for the graph (default: metrics_comparison.png)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Plot the metrics
    plot_csv_metrics(args.csv_files, args.output)

if __name__ == '__main__':
    main()