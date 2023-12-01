from __future__ import annotations
import numpy
import pandas
import sklearn.preprocessing
import sklearn.model_selection

class Adapter:
    def __init__(self, dataframe: pandas.DataFrame) -> None:
        self.dataframe: pandas.DataFrame = dataframe

    def skip(self, columns: list[str]) -> Adapter:
        new_dataframe = pandas.DataFrame(self.dataframe)
        return Adapter(new_dataframe.drop(labels=columns, axis=1))

    def only(self, labels: list[str]) -> Adapter:
        return Adapter(pandas.DataFrame(self.dataframe[labels]))

    def with_categoricals(self) -> Adapter:
        new_dataframe = pandas.DataFrame(self.dataframe)
        label_encoder = sklearn.preprocessing.LabelEncoder()
        for col in new_dataframe.select_dtypes(["object"]).columns:
            new_dataframe[col] = label_encoder.fit_transform(new_dataframe[col].astype("category"))
        return Adapter(new_dataframe)

    def only_numericals(self) -> Adapter:
        return Adapter(pandas.DataFrame(self.dataframe.select_dtypes(["int64", "float64"])))

    def scale(self) -> Adapter:
        return Adapter(pandas.DataFrame(sklearn.preprocessing.scale(self.dataframe),
                                        columns=self.dataframe.columns))

    def ok(self) -> pandas.DataFrame:
        return self.dataframe

    def split(self, target: str) -> tuple:
        Xs = self.skip([target]).with_categoricals().only_numericals().ok()
        Y = self.only([target]).with_categoricals().only_numericals().ok()
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(Xs, Y, test_size=0.3, random_state=42)
        trainset = {'x': X_train, 'y': y_train}
        testset = {'x': X_test, 'y': y_test}
        return trainset, testset
