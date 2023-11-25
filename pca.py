import numpy
import pandas
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def explain(scaled_data: pandas.DataFrame) -> None:
    pca = PCA().fit(scaled_data)
    plt.plot(scaled_data.columns, pca.explained_variance_ratio_, marker='o')
    plt.xlabel('PCA Components')
    plt.ylabel('Explained Variance')
    plt.xticks(rotation = 75)
    plt.title("PCA Results")
    plt.show()

def apply(scaled_data: pandas.DataFrame, n_components: int = 10) -> tuple[numpy.ndarray, numpy.ndarray, numpy.float64]:
    pca = PCA(n_components=n_components).fit(scaled_data)
    components = pca.components_
    data = pca.transform(scaled_data)
    explained_variance = sum(pca.explained_variance_ratio_)

    n_pcs = pca.components_.shape[0]
    most_important = [numpy.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    most_important_names = list(set([scaled_data.columns[most_important[i]] for i in range(n_pcs)]))

    return components, data, explained_variance, most_important_names

def visualize(components: numpy.ndarray, dataframe: pandas.DataFrame) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    for i, (x, y) in enumerate(zip(components[0, :], components[1, :])):
        ax.plot([0, x], [0, y], color='k')
        ax.text(x, y, dataframe.columns[i], fontsize='10')
