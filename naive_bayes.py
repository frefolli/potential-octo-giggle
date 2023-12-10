import sklearn.naive_bayes
import sklearn.metrics
import matplotlib.pyplot as plt
import model
import abc
import numpy
import enum

class Probability(enum.Enum):
    Multinomial = sklearn.naive_bayes.MultinomialNB
    Gaussian = sklearn.naive_bayes.GaussianNB
    Categorical = sklearn.naive_bayes.CategoricalNB
    Bernoulli = sklearn.naive_bayes.BernoulliNB
    Complement = sklearn.naive_bayes.ComplementNB

class NaiveBayes(model.Model):
    @staticmethod
    def read_probability(prob: Probability) -> abc.ABCMeta:
        if not isinstance(prob, Probability):
            raise Exception("undefined probability type: '%s'" % prob)
        return prob.value
    
    def __init__(self, prob: Probability) -> None:
        self.model = NaiveBayes.read_probability(prob)()

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
