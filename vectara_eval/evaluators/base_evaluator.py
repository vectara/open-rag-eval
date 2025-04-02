from abc import ABC, abstractmethod

class Evaluator(ABC):

    @classmethod
    @abstractmethod
    def plot_metrics(cls, csv_files, output_file='metrics_comparison.png'):
        """Plot metrics from multiple CSV files."""
        pass
