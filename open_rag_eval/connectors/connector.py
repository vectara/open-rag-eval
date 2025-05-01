from abc import abstractmethod
class Connector:

    @abstractmethod
    def fetch_data(
        self,
    ) -> None:
        pass
