from __future__ import annotations
import pandas
import sklearn.preprocessing
import sklearn.model_selection
import dataframe_adapter

class ModelDataAdapter:
    def __init__(self, dataframe: pandas.DataFrame) -> None:
        self.dataframe: pandas.DataFrame = dataframe

    def split(self, target: str) -> tuple:
        Xs = dataframe_adapter.DataFrameAdapter(self.dataframe).skip([target]).with_categoricals().only_numericals().ok()
        Y = dataframe_adapter.DataFrameAdapter(self.dataframe).only([target]).with_categoricals().only_numericals().ok()
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Xs, Y, test_size=0.3, random_state=42)
        trainset = {'x': X_train, 'y': y_train}
        testset = {'x': X_test, 'y': y_test}
        return trainset, testset
