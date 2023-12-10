import enum
import model
import sklearn.cluster
import sklearn.metrics
import numpy
import pandas

class Strategy(enum.Enum):
    Lloyd = "lloyd"
    Elkan = "elkan"
    Auto = "auto"
    Full = "full"

class Clustering(model.Model):
    @staticmethod
    def number_of_clusters(df: pandas.DataFrame, field: str) -> int:
        return len(set(df[field]))
    
    def __init__(self,
                 n_clusters: int,
                 strategy: Strategy = Strategy.Lloyd):
        self.model = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                            algorithm=strategy.value)

    def fit(self, trainset: dict) -> None:
        self.model.fit(trainset['x'])

    def predict(self, inputs: list) -> list:
        return self.model.predict(inputs)

    def evaluate(self, testset: dict) -> tuple:
        y_pred = self.predict(testset['x'])
        acc = sklearn.metrics.accuracy_score(numpy.ravel(testset['y']), y_pred)
        return 1-acc, acc

    def plot(self) -> None:
        raise Exception("not implemented")

    def dump(self, path: str) -> None:
        raise Exception("not implemented")
