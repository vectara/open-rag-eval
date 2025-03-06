from abc import ABC, abstractmethod


class Connector(ABC):
    @abstractmethod
    def fetch_data(self):
        pass

    def get_data(self):
        return self.fetch_data()