import polars as pl
import pandas as pd
import numpy as np
import os
import joblib
import warnings
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer, FunctionTransformer, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.custom_transformers import (
    PlaceholderToNullTransformer,
    TypeCorrector,
    SearchCorpusGenerator,
    NumericalRankingTransformer,
    FeatureBinner,
    GenresBinarizer
)
from sklearn.ensemble import RandomForestRegressor

from sentence_transformers import SentenceTransformer

from sklearn import set_config
set_config(transform_output="pandas") 

warnings.filterwarnings('ignore', category=UserWarning)
pd.options.display.max_columns = 100

df = pl.read_parquet("data/imdb_us_movies_cleaned.parquet")

df_sample = df.sample(n=1000, seed=42)
print(f"Full dataset shape: {df.shape}")
print(f"Sample dataset shape: {df_sample.shape}")

df_sample_pd = df_sample.to_pandas()
print(f"\nSchema: {df_sample.schema}")
df_pd = df.to_pandas()
print(f"\nSchema: {df.schema}")

numeric_cols = [c for c in df_sample_pd.columns if c.startswith('num__')]
categorical_cols = [c for c in df_sample_pd.columns if c.startswith('cat__')]

int_cols = ['num__isAdult', 'num__startYear', 'num__numVotes']
float_cols = ['num__averageRating', 'num__runtimeMinutes']
obj_cols = categorical_cols

foundational_pipeline = Pipeline(steps=[
    ('placeholders_to_null', PlaceholderToNullTransformer(numeric_cols, categorical_cols)),
    ('type_corrector', TypeCorrector(int_cols, float_cols, obj_cols))
])

df_sample_clean = foundational_pipeline.fit_transform(df_sample_pd)
df_clean = foundational_pipeline.fit_transform(df_pd)

print("--- Schema Before ---")
df_sample_pd.info()
print("\n--- Schema After ---")
df_clean.info()

def parse_crew_and_cast(df: pl.DataFrame) -> pl.DataFrame:
    director_expr = pl.col("remainder__directors").list.first().struct.field("primaryName").alias("director_name")
    
    actor_names_expr = (
        pl.col("remainder__cast")
        .list.eval(pl.element().struct.field("primaryName"))
        .alias("actor_names_list")
    )

    writer_names_expr = (
        pl.col("remainder__writers")
        .list.eval(pl.element().struct.field("primaryName"))
        .alias("writer_names_list")
    )
    
    return df.select(pl.all(), director_expr, actor_names_expr, writer_names_expr)


df_parsed_pl = parse_crew_and_cast(df_sample)
df_sample_pd = df_parsed_pl.to_pandas()

numeric_cols = [c for c in df_sample_pd.columns if c.startswith('num__')]
categorical_cols = [c for c in df_sample_pd.columns if c.startswith('cat__')]

int_cols = ['num__isAdult', 'num__startYear', 'num__numVotes']
float_cols = ['num__averageRating', 'num__runtimeMinutes']
obj_cols = categorical_cols

foundational_pipeline = Pipeline(steps=[
    ('placeholders_to_null', PlaceholderToNullTransformer(numeric_cols, categorical_cols)),
    ('type_corrector', TypeCorrector(int_cols, float_cols, obj_cols))
])

df_clean = foundational_pipeline.fit_transform(df_pd)

print(f"df_clean shape: {df_clean.shape}")
print("Columns in df_clean:", df_clean.columns.tolist())

import string
import nltk
from nltk.stem import PorterStemmer

numeric_cols = ['num__isAdult', 'num__startYear', 'num__runtimeMinutes', 'num__averageRating', 'num__numVotes']
corpus_cols = ['cat__title', 'cat__genres', 'remainder__cast', 'remainder__directors', 'remainder__writers']
binner_cols = ['num__startYear', 'num__runtimeMinutes', 'num__averageRating']
onehot_cols = ['cat__types']
multilabel_cols = 'cat__genres'

full_preprocessor = ColumnTransformer(
    transformers=[
        ('pass_tconst', 'passthrough', ['cat__tconst']),
        ('search_corpus', SearchCorpusGenerator(), corpus_cols),
        
        ('ranking_numeric', NumericalRankingTransformer(log_cols=['num__numVotes'], clip_cols=['num__runtimeMinutes']), numeric_cols),

        ('ui_binned', Pipeline([
            ('binner', FeatureBinner()),
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), 
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), binner_cols),
        
        ('filter_onehot_types', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), 
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), onehot_cols),
        
        ('filter_multilabel_genres', GenresBinarizer(), multilabel_cols)
    ],
    remainder='drop' 
)

print("Full preprocessing ColumnTransformer ready with Text Normalization.")

print("--- Validating Full Preprocessing Pipeline on clean data... ---")

df_parsed_full_pl = parse_crew_and_cast(df)
df_parsed_full_pd = df_parsed_full_pl.to_pandas()

df_clean = foundational_pipeline.fit_transform(df_parsed_full_pd)

print(f"Input shape (df_clean): {df_clean.shape}")

X_processed = full_preprocessor.fit_transform(df_clean)

print(f"\nOutput shape (X_processed): {X_processed.shape}")
X_processed.head()

print("Saving processed data immediately...")
X_processed.to_parquet('data/imdb_us_movies_processed.parquet')
print("data/imdb_us_movies_processed.parquet saved early.")

tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', lowercase=True)

tfidf_vectors = tfidf_vectorizer.fit_transform(X_processed['search_corpus__searchable_text'])

print(f"TF-IDF sparse matrix shape: {tfidf_vectors.shape}")
print(tfidf_vectorizer.get_feature_names_out()[:50]) 

from sentence_transformers import SentenceTransformer
import numpy as np

print("--- Loading SBERT Model ---")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"Calculating embeddings for {len(X_processed)} movies...")

sbert_embeddings = sbert_model.encode(
    X_processed['search_corpus__searchable_text'].tolist(), 
    show_progress_bar=True
)

print(f"Sentence Embedding dense matrix shape: {sbert_embeddings.shape}")

from sklearn.metrics.pairwise import cosine_similarity
import timeit

print("--- R&D: Similarity Metric Showdown ---")

norms = np.linalg.norm(sbert_embeddings, axis=1)
print(f"1. Normalization Check (First 5 vectors): {norms[:5]}")
is_normalized = np.allclose(norms, 1, atol=1e-5)
print(f"   -> Are all vectors normalized? {is_normalized}")

query_vec = sbert_embeddings[0].reshape(1, -1)

scores_cosine = cosine_similarity(query_vec, sbert_embeddings)[0]
scores_dot = np.dot(query_vec, sbert_embeddings.T)[0]

are_identical = np.allclose(scores_cosine, scores_dot, atol=1e-5)
print(f"\n2. Accuracy Check")
print(f"   -> Do np.dot and cosine_similarity produce identical scores? {are_identical}")

print(f"\n3. Speed Test (running 1000 iterations)...")
t_cosine = timeit.timeit(lambda: cosine_similarity(query_vec, sbert_embeddings), number=1000)
t_dot = timeit.timeit(lambda: np.dot(query_vec, sbert_embeddings.T), number=1000)

print(f"   -> Cosine Similarity Time: {t_cosine:.4f} seconds")
print(f"   -> Dot Product Time:       {t_dot:.4f} seconds")
print(f"   -> WINNER: Dot Product is {t_cosine / t_dot:.2f}x faster!")

ranking_features = X_processed.filter(like='ranking_numeric')

plt.figure(figsize=(10, 8))
sns.heatmap(ranking_features.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Ranking Features')
plt.show()

y_target = df_clean['num__averageRating'].fillna(0) 
X_features = X_processed.drop(columns=['search_corpus__searchable_text']) 
model_features = [col for col in X_features.columns if 
                  col.startswith('ranking_numeric') or 
                  col.startswith('filter_') or
                  col.startswith('ui_binned')]

X_model_data = X_features[model_features]

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_model_data, y_target)

importances = pd.Series(rf.feature_importances_, index=X_model_data.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 12))
sns.barplot(x=importances.values, y=importances.index)
plt.title('Feature Importances for Predicting Average Rating')
plt.show()

print("--- Saving Artifacts ---")

joblib.dump(full_preprocessor, 'data/preprocessor.joblib')
print("1. preprocessor.joblib saved")

joblib.dump(tfidf_vectorizer, 'data/tfidf_vectorizer.joblib')
print("2. tfidf_vectorizer.joblib saved")

np.save('data/sbert_embeddings.npy', sbert_embeddings)
print("3. sbert_embeddings.npy saved")

X_processed.to_parquet('data/imdb_us_movies_processed.parquet')
print("4. imdb_us_movies_processed.parquet saved")
