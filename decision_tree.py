import sklearn.tree # DecisionTreeClassifier, plot_tree
import sklearn.metrics
import matplotlib.pyplot as plt
import model

class DecisionTree(model.Model):
    def __init__(self) -> None:
        self.model = sklearn.tree.DecisionTreeClassifier(random_state=42)

    def fit(self, trainset: dict) -> None:
        self.model.fit(trainset['x'], trainset['y'])

    def predict(self, inputs: list) -> list:
        return self.model.predict(inputs)

    def evaluate(self, testset: dict) -> tuple:
        y_pred = self.predict(testset['x'])
        cm = sklearn.metrics.confusion_matrix(testset['y'], y_pred)
        return (cm, cm.diagonal().sum() / cm.sum())

    def plot(self) -> None:
        fig, ax = plt.subplots(figsize=(150, 100))
        sklearn.tree.plot_tree(self.model, filled=True, ax=ax)
        plt.plot()

    def dump(self, path: str) -> None:
        raise Exception("not implemented")
