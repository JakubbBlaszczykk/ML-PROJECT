# **Preprocessing and Feature Engineering Recommendations**


## **1.0 Summary**

This document outlines the recommended preprocessing and feature engineering pipeline to prepare the film dataset for a hybrid information retrieval (search) engine. The methodology is designed to serve two distinct functions:

1. **Search Corpus Generation:** To create a single, normalized, and searchable text document for each film.  
2. **Filter/Rank Feature Preparation:** To engineer clean, normalized numerical and categorical features for filtering (e.g., genre \= 'horror') and ranking (e.g., boost by averageRating).

The following steps are based on the findings from the initial data cleaning and exploratory data analysis (EDA) notebooks.

## **2.0 Foundational Cleansing and Type Conversion**

The initial step is to standardize data types and missing value representation.

* **Conversion of Placeholders to Nulls:** All non-standard missing value placeholders (e.g., \-1.0 in numerical columns, "missing" in categorical columns) must first be converted to a standard null (e.g., np.nan, None).  
  * **Rationale:** This intermediate step is critical. It provides a single, uniform representation of missingness, which is required to (a) reliably generate the indicator features in Step 3.0 and (b) enable the data type correction to nullable-integer formats.  
* **Data Type Correction:** Following the conversion of \-1.0 to null, it is recommended to convert num\_\_startYear and num\_\_isAdult from float to a **nullable integer type** (e.g., pd.Int64Dtype() or pl.Int64). This enforces data integrity and is more semantically correct.

## **3.0 Missing Data Strategy: Indicator Features**

Simple imputation (e.g., mean/median) is not recommended, as it can introduce significant bias and skew, especially in columns with high missingness (e.g., num\_\_averageRating at 38.3%).  
A more robust "Indicator Feature" (or "Missingness Flag") strategy is advised:

1. **Generate Indicator Columns:** For each numerical column with missing data (num\_\_startYear, num\_\_runtimeMinutes, num\_\_averageRating, num\_\_numVotes), a new binary indicator column (e.g., is\_missing\_startYear) will be created. This column will hold a 1 if the original value was null and 0 otherwise. This preserves the "missingness" as a potentially predictive signal for the model.  
2. **Fill Numerical Nulls:** After indicators are created, the null values in the *original* numerical columns will be filled with 0\. This is now safe, as the model can distinguish a true zero from a missing-value-zero by using the indicator column.  
3. **Fill Categorical Nulls:** Null values in categorical columns (cat\_\_types, cat\_\_genres) will be filled with a dedicated string literal, **"Unknown"**. This allows the value to be treated as a distinct category in subsequent encoding.

## **4.0 Feature Engineering**

### **4.1 Search Corpus Generation**

A master searchable\_text column must be engineered to serve as the target "document" for text-based queries.

1. **Flatten Nested Structures:** Text data from nested list-of-struct fields (remainder\_\_cast, remainder\_\_directors, remainder\_\_writers) will be extracted.  
   * **Nested Null Handling:** During extraction, any struct where the primaryName is null or an empty string will be explicitly skipped.  
   * **Corpus Purity (Rationale):** We will *only* extract primaryName. Other data in the structs (e.g., birthYear, deathYear) will be **intentionally discarded** from this search text. Including this data would pollute the *film's* search corpus with irrelevant numbers, leading to poor search results (e.g., a query for the year "1990" matching all actors born in 1990).  
   * The extracted names will be concatenated with spaces into new columns (e.g., cast\_text).  
2. **Clean Genre Text:** The cat\_\_genres column (e.g., "crime,drama") will be transformed by replacing commas with spaces (e.g., "crime drama") to create genres\_text.  
3. **Concatenate Corpus:** The final searchable\_text column will be a concatenation of all primary text fields: cat\_\_title, genres\_text, cast\_text, directors\_text, and writers\_text.

### **4.2 Numerical Feature Transformation (for Ranking Model)**

To prepare continuous numerical features for use as signals in a *learning-to-rank model*, the following transformations are necessary:

* **Logarithmic Transform:** Apply a log1p transformation to num\_\_numVotes to correct its extreme right skew, as identified in the EDA.  
* **Outlier Clipping (Winsorization):** num\_\_runtimeMinutes exhibits extreme outliers (e.g., 12,600). Values should be clipped at a high percentile (e.g., the 99th) to mitigate their disproportional effect on the ranking model.  
* **Scaling:** All numerical features (including the newly created binary indicators) must be scaled, preferably using StandardScaler or MinMaxScaler, to normalize their ranges for model consumption.

### **4.3 Categorical Feature Encoding (for Filtering)**

To enable faceted search and filtering, categorical data will be binarized:

* **cat\_\_types:** Use **One-Hot Encoding**, as this is a single-value, low-cardinality field.  
* **cat\_\_genres:** Use **Multi-Label Binarization**, as this is a multi-value field (comma-separated). This will create a binary column for each unique genre, allowing for multi-select filtering.

### **4.4 Feature Binning (for User-Facing Filters)**

To enhance the user's filtering capabilities, we will supplement the continuous features by binning them into new categorical features.

* **Methodology:** Apply logical cuts to key numerical fields to create new categorical columns.  
* **Target Columns:**  
  * num\_\_startYear: Bin into decades (e.g., "1980s", "1990s", "2000s", etc.).  
  * num\_\_runtimeMinutes: Bin into logical groups (e.g., "\< 60 min", "60-90 min", "90-120 min", "120+ min").  
  * num\_\_averageRating: Bin into quality tiers (e.g., "Low", "Medium", "High", "Excellent").  
* These new binned features can then be One-Hot Encoded for use as simple, user-friendly filters.

## **5.0 Text Vectorization Strategy**

The final step is to convert the searchable\_text corpus into numerical vectors for indexing. Two parallel strategies are **essential** to support a robust hybrid search.

### **5.1 Recommendation 1: Keyword Retrieval (Sparse Vectors)**

* **Methodology:** Use **TF-IDF** (or its modern equivalent, **BM25**). This approach is non-negotiable as it excels at matching specific keywords (e.g., actor names, titles), which is a primary user expectation.  
* **Recommended Library:** sklearn.feature\_extraction.text.TfidfVectorizer.  
* **Preprocessing:** This library should be configured to handle text normalization automatically (lowercase=True, stop\_words='english').

### **5.2 Recommendation 2: Semantic Retrieval (Dense Vectors)**

* **Methodology:** Use pre-trained **Sentence Embeddings**. This complements keyword search by capturing the semantic *meaning* of the text, allowing for conceptual queries (e.g., "funny space movie") that may not share keywords.  
* **Recommended Library:** sentence-transformers (e.g., model all-MiniLM-L6-v2).  
* **Preprocessing:** Manual preprocessing (like stopword removal) is **not** recommended, as it degrades contextual understanding. The raw searchable\_text should be passed directly to the model.

### **5.3 Implementation**

It is advised to generate *both* sparse vectors (for a keyword index like Elasticsearch/OpenSearch) and dense vectors (for a vector index like FAISS/Pinecone/Milvus). A production-grade search engine will query both and merge the results (e.g., using Reciprocal Rank Fusion \- RRF) to provide a comprehensive search experience that handles both literal and conceptual queries.

## **6.0 Post-Processing: Feature Selection**

As an optional final step before building a complex *ranking* model, we should analyze the utility of the feature set we have engineered.

* **Rationale:** We have created many new features (indicators, binned categories, scaled numericals). Some may be highly correlated (e.g., num\_\_averageRating and its binned version) or provide little predictive information.  
* **Methodology:**  
  1. **Correlation Analysis:** Generate a correlation matrix to identify and potentially remove redundant features.  
  2. **Feature Importance:** Train a simple baseline model (e.g., Random Forest) on a subset of the data to get an initial reading of feature importances. Features with near-zero importance could be candidates for removal.

This step will help reduce model complexity, decrease training time, and potentially improve the generalizability of the final ranking model.

## **7.0 Advanced Alternatives & Future Work (v2.0)**

This section documents advanced methodologies that were considered but are recommended for a future "v2.0" iteration, as they add significant complexity to the initial build.

### **7.1 Model-Based Imputation**

* **Concept:** Use a machine learning model (e.g., sklearn.impute.IterativeImputer) to predict missing values (like averageRating) based on all other features (startYear, genres, etc.).  
* **Recommendation:** This is **not recommended** for this project.  
  * **Inauthenticity:** With 38.3% of ratings missing, the model would be "inventing" ratings for over a third of the dataset. This is inauthentic for a search engine and damages user trust.  
  * **Bias:** The imputed values would be based on the model's own biases, not real-world data.  
  * The **Indicator Feature** strategy (Section 3.0) is a more honest and robust solution, as it treats "missing" as its own signal.

### **7.2 Entity-Based Search**

* **Concept:** The birthYear and deathYear data (Section 4.1) is valuable but should not be part of the *film's* search text. A "v2.0" system would handle this by creating two separate search indexes:  
  1. A film\_index (as planned).  
  2. An entity\_index containing data on actors, directors, etc. (e.g., {name: "Tom Cruise", birthYear: 1962}).  
* **Workflow:** A query like "Tom Cruise 1962" would first query the entity\_index to find "Tom Cruise", then use that entity's name to search the film\_index.  
* **Recommendation:** This is a powerful feature but adds significant architectural complexity (multiple indexes, query parsing, result merging). It should be deferred until the core film search is operational.

## **8.0 Recommended Project Structure**

A well-organized file structure is crucial for reproducibility. Something like this could be implemented:

/project-root  
├── models/  
│   ├── tfidf\_vectorizer.joblib     (Saved sklearn vectorizer)  
│   ├── standard\_scaler.joblib      (Saved sklearn scaler)  
│   └── sbert\_model/              (Cached sentence-transformer model)  
│  
├── notebooks/  
│   ├── 00\_cleaning.ipynb           (Done)  
│   ├── 01\_eda.ipynb                (Done)  
│   ├── 02\_preprocessing.ipynb      (Implements this report)  
│   ├── 03\_vectorization.ipynb      (Generates and saves vectors)  
│   └── 04\_modeling\_baseline.ipynb  (Builds baseline search models)  
│  
├── src/  
│   ├── preprocessing.py          (Reusable preprocessing pipeline/functions)  
│   └── vectorization.py          (Reusable vectorization functions)  
│  
├── app/  
│   └── main.py                   (Optional: A simple FastAPI/Flask app to serve results)  
│  
└── README.md                       (Project overview, setup, and run instructions)

## **9.0 Next Steps: Model Implementation**

Once preprocessing (Section 1-6) is complete, the following steps will build the search engine:

1. **Build Baseline Keyword Search:** Use the tfidf\_vectors.npz (or a BM25 index in OpenSearch/Elasticsearch) with cosine\_similarity to create the first baseline. This model will excel at matching exact keywords ("Tom Cruise").  
2. **Build Baseline Semantic Search:** Use the sbert\_embeddings.npy with a vector index (like FAISS) to create the second baseline. This model will excel at conceptual queries ("funny space movie").  
3. **Create Hybrid Search:** Combine the results from Step 1 and 2\. A simple, effective method is **Reciprocal Rank Fusion (RRF)**, which merges the two ranked lists without needing to tune weights.  
4. **Develop a Learning-to-Rank (LTR) Model:**  
   * **Goal:** To *re-rank* the top 100-200 results from the Hybrid Search (Step 3\) to produce a more relevant final ordering.  
   * **Features:** This model will use all the features we engineered in Section 4.0 (e.g., averageRating\_scaled, numVotes\_log\_scaled, is\_missing\_rating, bin\_year\_1990s).  
   * **Model:** XGBRanker or a RandomForestClassifier are excellent starting points.  
   * **Challenge:** This step requires **labeled training data** (i.e., (query, document, relevance\_score) tuples), which often must be manually created ("judgment lists") and is the most time-consuming part of building a production-grade LTR system.