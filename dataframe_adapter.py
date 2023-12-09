from __future__ import annotations
import pandas
import sklearn.preprocessing
import sklearn.model_selection

class DataFrameAdapter:
    def __init__(self, dataframe: pandas.DataFrame) -> None:
        self.dataframe: pandas.DataFrame = dataframe

    def skip(self, columns: list[str]) -> DataFrameAdapter:
        new_dataframe = pandas.DataFrame(self.dataframe)
        return DataFrameAdapter(new_dataframe.drop(labels=columns, axis=1))

    def only(self, labels: list[str]) -> DataFrameAdapter:
        return DataFrameAdapter(pandas.DataFrame(self.dataframe[labels]))

    def with_categoricals(self) -> DataFrameAdapter:
        new_dataframe = pandas.DataFrame(self.dataframe)
        label_encoder = sklearn.preprocessing.LabelEncoder()
        for col in new_dataframe.select_dtypes(["object"]).columns:
            new_dataframe[col] = label_encoder.fit_transform(new_dataframe[col].astype("category"))
        return DataFrameAdapter(new_dataframe)

    def only_numericals(self) -> DataFrameAdapter:
        return DataFrameAdapter(pandas.DataFrame(self.dataframe.select_dtypes(["int64", "float64"])))

    def scale(self) -> DataFrameAdapter:
        return DataFrameAdapter(pandas.DataFrame(sklearn.preprocessing.scale(self.dataframe),
                                                 columns=self.dataframe.columns))

    def ok(self) -> pandas.DataFrame:
        return self.dataframe
