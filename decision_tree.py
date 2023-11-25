import sklearn.tree # DecisionTreeClassifier, plot_tree
import sklearn.metrics
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self, trainset: dict) -> None:
        self.model = sklearn.tree.DecisionTreeClassifier(random_state=42)
        self.model.fit(trainset['x'], trainset['y'])

    def plot(self) -> None:
        fig, ax = plt.subplots(figsize=(150, 100))
        sklearn.tree.plot_tree(self.model, filled=True, ax=ax)
        plt.plot()

    def evaluate(self, testset: dict) -> tuple:
        y_pred = self.model.predict(testset['x'])
        cm = sklearn.metrics.confusion_matrix(testset['y'], y_pred)
        return cm, cm.diagonal().sum() / cm.sum()
