import sklearn.naive_bayes
import sklearn.metrics
import matplotlib.pyplot as plt
import model
import abc
import numpy

class NaiveBayes(model.Model):
    @staticmethod
    def type_str_to_klass(type: str) -> abc.ABCMeta:
        if type == "multinomial":
            return sklearn.naive_bayes.MultinomialNB
        elif type == "gaussian":
            return sklearn.naive_bayes.GaussianNB
        elif type == "categorical":
            return sklearn.naive_bayes.CategoricalNB
        elif type == "bernoulli":
            return sklearn.naive_bayes.BernoulliNB
        elif type == "complement":
            return sklearn.naive_bayes.ComplementNB
        else:
            raise Exception("invalid Naive Bayes type str: '%s'" % type)
    
    def __init__(self, type: str) -> None:
        self.model = NaiveBayes.type_str_to_klass(type)()

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
