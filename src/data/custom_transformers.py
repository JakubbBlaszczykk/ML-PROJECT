import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer, FunctionTransformer, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.stem import PorterStemmer

from sklearn import set_config
set_config(transform_output="pandas")


class PlaceholderToNullTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.numeric_cols:
            X_copy[col] = X_copy[col].replace(-1.0, np.nan)
        for col in self.categorical_cols:
            X_copy[col] = X_copy[col].replace("missing", np.nan)
        return X_copy
    


class TypeCorrector(BaseEstimator, TransformerMixin):
    def __init__(self, int_cols, float_cols, obj_cols):
        self.int_cols = int_cols
        self.float_cols = float_cols
        self.obj_cols = obj_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.int_cols:
            X_copy[col] = X_copy[col].astype(pd.Int64Dtype())
        for col in self.float_cols:
            X_copy[col] = X_copy[col].astype(float)
        for col in self.obj_cols:
            X_copy[col] = X_copy[col].astype(str).replace('<NA>', np.nan) # Ensure object
        return X_copy



class SearchCorpusGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def _flatten_names(self, list_of_structs):
        """Extract primaryName from list/array of person structs."""
        import numpy as np
        
        if list_of_structs is None:
            return ""
        if isinstance(list_of_structs, float) and np.isnan(list_of_structs):
            return ""
        
        if not isinstance(list_of_structs, (list, np.ndarray)):
            return ""
        
        names = []
        for person in list_of_structs:
            if person and isinstance(person, dict):
                name = person.get('primaryName')
                if name is not None:
                    names.append(name)
        
        return " ".join(names)

    def _normalize_text(self, text):
        """
        Normalizes text: lowercase, remove punctuation, stem using PorterStemmer.
        """
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        
        return " ".join(stemmed_words)

    def transform(self, X):
        X_copy = X.copy()
        
        X_copy['cast_text'] = X_copy['remainder__cast'].apply(self._flatten_names)
        X_copy['directors_text'] = X_copy['remainder__directors'].apply(self._flatten_names)
        X_copy['writers_text'] = X_copy['remainder__writers'].apply(self._flatten_names)
        
        X_copy['genres_text'] = X_copy['cat__genres'].fillna('').str.replace(',', ' ')
        
        X_copy['searchable_text'] = (
            X_copy['cat__title'].fillna('') + ' ' +
            X_copy['genres_text'] + ' ' +
            X_copy['cast_text'] + ' ' +
            X_copy['directors_text'] + ' ' +
            X_copy['writers_text']
        )
        
        X_copy['normalized_title'] = X_copy['cat__title'].fillna('').apply(self._normalize_text)
        
        return X_copy[['searchable_text', 'normalized_title', 'cat__title']]
    


class NumericalRankingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, log_cols, clip_cols, clip_quantile=0.99):
        self.log_cols = log_cols
        self.clip_cols = clip_cols
        self.clip_quantile = clip_quantile
        self.clip_values_ = {}
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        for col in self.clip_cols:
            self.clip_values_[col] = X[col].quantile(self.clip_quantile)
        
        X_transformed_for_fit = self.transform(X, fit_scaler=False)
        self.scaler.fit(X_transformed_for_fit)
        return self

    def transform(self, X, fit_scaler=True):
        X_copy = X.copy()
        indicator_cols = []
        
        for col in X_copy.columns:
            if X_copy[col].isnull().any():
                indicator_col_name = f'is_missing_{col}'
                X_copy[indicator_col_name] = X_copy[col].isnull().astype(int)
                indicator_cols.append(indicator_col_name)
                X_copy[col] = X_copy[col].fillna(0) 

        for col in self.log_cols:
            X_copy[f'{col}_log'] = np.log1p(X_copy[col])
            X_copy = X_copy.drop(columns=[col]) 
            
        for col in self.clip_cols:
            if col in self.clip_values_: 
                clip_val = self.clip_values_[col]
                X_copy[col] = X_copy[col].clip(upper=clip_val)

        if fit_scaler:
            X_scaled = self.scaler.transform(X_copy)
            X_copy = pd.DataFrame(X_scaled, columns=X_copy.columns, index=X_copy.index)
        
        return X_copy
    


class FeatureBinner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        year_bins = [1870, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, np.inf]
        year_labels = ['<1950', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
        X_copy['year_bin'] = pd.cut(X_copy['num__startYear'], bins=year_bins, labels=year_labels, right=False)
        
        runtime_bins = [-np.inf, 60, 90, 120, np.inf]
        runtime_labels = ['< 60 min', '60-90 min', '90-120 min', '120+ min']
        X_copy['runtime_bin'] = pd.cut(X_copy['num__runtimeMinutes'], bins=runtime_bins, labels=runtime_labels, right=False)
        
        rating_bins = [-np.inf, 4, 6, 8, np.inf]
        rating_labels = ['Low (0-4)', 'Medium (4-6)', 'High (6-8)', 'Excellent (8+)']
        X_copy['rating_bin'] = pd.cut(X_copy['num__averageRating'], bins=rating_bins, labels=rating_labels, right=False)

        bin_cols = ['year_bin', 'runtime_bin', 'rating_bin']
        return X_copy[bin_cols].astype(str)
    


class GenresBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        data_split = X.fillna('').str.split(',')
        self.mlb.fit(data_split)
        return self

    def transform(self, X):
        data_split = X.fillna('').str.split(',')
        return self.mlb.transform(data_split)
    
    def get_feature_names_out(self, input_features=None):
        return self.mlb.classes_
    

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

