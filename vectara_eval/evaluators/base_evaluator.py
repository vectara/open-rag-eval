from abc import ABC, abstractmethod

class Evaluator(ABC):

    @abstractmethod
    def plot_metrics(self, csv_files, output_file='metrics_comparison.png'):
        """Plot metrics from multiple CSV files."""
        pass
