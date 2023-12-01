import abc

class Model(abc.ABC):
    @abc.abstractmethod
    def fit(self, trainset: dict) -> None:
        pass

    @abc.abstractmethod
    def predict(self, inputs: list) -> list:
        pass

    @abc.abstractmethod
    def evaluate(self, testset: dict) -> tuple:
        pass

    @abc.abstractmethod
    def plot(self) -> None:
        pass

    @abc.abstractmethod
    def dump(self, path: str) -> None:
        pass
