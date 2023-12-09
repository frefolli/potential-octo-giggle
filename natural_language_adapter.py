from __future__ import annotations
import pandas
import nltk
import nltk.corpus
import nltk.tokenize
import nltk.stem
import string

# Download delle stopwords e del wordnet corpus di nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

class NaturalLanguageAdapter:
    def __init__(self, dataframe: pandas.DataFrame) -> None:
        self.dataframe: pandas.DataFrame = dataframe
    
    def lowercase(self, field: str) -> NaturalLanguageAdapter:
        self.dataframe[field] = self.dataframe[field].str.lower()
        return self

    def filter_pattern(self, field: str, pattern: str, regex: bool = True) -> NaturalLanguageAdapter:
        self.dataframe[field] = self.dataframe[field].str.replace(pattern, '', regex=regex)
        return self

    def filter_numbers(self, field: str) -> NaturalLanguageAdapter:
        return self.filter_pattern(field, r'\d+')

    def tokenize(self, field: str) -> NaturalLanguageAdapter:
        self.dataframe[field] = self.dataframe[field].apply(nltk.tokenize.word_tokenize)
        return self

    def filter_values(self, field: str, unwanted_values: set) -> NaturalLanguageAdapter:
        self.dataframe[field] = self.dataframe[field].apply(lambda x: [item for item in x if item not in unwanted_values])
        return self

    def filter_stopwords(self, field: str, language: str) -> NaturalLanguageAdapter:
        return self.filter_values(field, set(nltk.corpus.stopwords.words(language)))

    def filter_punctuation(self, field: str) -> NaturalLanguageAdapter:
        return self.filter_values(field, set(string.punctuation))

    def filter_token_length(self, field: str, threshold: int) -> NaturalLanguageAdapter:
        self.dataframe[field] = self.dataframe[field].apply(lambda x: [word for word in x if len(word) > threshold])
        return self

    def filter_variance(self, field: str, min_v: float, max_v: float) -> NaturalLanguageAdapter:
        raise Exception("not implemented")
        return self

    def filter_standard_deviation(self, field: str, min_v: float, max_v: float) -> NaturalLanguageAdapter:
        raise Exception("not implemented")
        return self

    def lemmatize(self, field: str) -> NaturalLanguageAdapter:
        lemmatizer = nltk.stem.WordNetLemmatizer()
        self.dataframe[field] = self.dataframe[field].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
        return self

    def join_text(self, field: str) -> NaturalLanguageAdapter:
        self.dataframe[field] = self.dataframe[field].apply(lambda x: ' '.join(x))
        return self
