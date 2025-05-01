from abc import abstractmethod
class Connector:

    @abstractmethod
    def fetch_data(
        self,
        input_csv: str = "queries.csv",
        output_csv: str = "results.csv",
    ) -> None:
        pass
