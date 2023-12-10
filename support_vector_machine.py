import enum
import model
import sklearn.svm
import sklearn.metrics
import numpy

class Kernel(enum.Enum):
    Linear = 'linear'
    Poly = 'poly'
    Rbf = 'rbf'
    Sigmoid = 'sigmoid'
    Precomputed = 'precomputed'

class SVM(model.Model):
    def __init__(self, kernel: Kernel, regularization: int):
        self.model = sklearn.svm.SVC(kernel=kernel.value, random_state=42, C=regularization)

    def fit(self, trainset: dict) -> None:
        self.model.fit(trainset['x'], numpy.ravel(trainset['y']))

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
