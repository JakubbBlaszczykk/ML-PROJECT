#  **Proof of Concept Implementation Documentation**

## **Overview**
This document provides a detailed overview of the Proof of Concept (PoC) implementation for building a **basic search functionality** based on **TF-IDF (Term Frequency-Inverse Document Frequency) vectorization** and **cosine similarity**. The main goal of this PoC is to demonstrate the capability of searching a dataset of movie descriptions and effectively retrieving the most **relevant** results based on user queries. This mechanism serves as a foundation for more advanced recommendation systems or contextual search engines.

---

## **Key Implementation Components**

### 1. **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization**
* **Purpose:** To convert raw text (movie descriptions) into a **numerical representation** (vectors) in a feature space.
* **Technology:** The `TfidfVectorizer` class from the `sklearn.feature_extraction.text` library is utilized.
* **Details:** This process assigns a weight to each word in a document. This weight is proportional to the **frequency of the word in the specific document (TF)** but inversely proportional to the **frequency of that word across the entire corpus (IDF)**. This means rare words specific to a film receive higher weights, which is crucial for distinguishing documents in the feature space. The resulting TF-IDF vectors are the basis for similarity calculation.

### 2. **Cosine Similarity Calculation**
* **Purpose:** To determine the degree of **thematic similarity** between the user query vector and the vectors of individual movie descriptions.
* **Technology:** The `cosine_similarity` function from the `sklearn.metrics.pairwise` module is used.
* **Details:** Cosine similarity measures the **cosine of the angle** between two vectors in a multi-dimensional feature space. It's the preferred metric for text data as it focuses on the **orientation of the vectors** (content) rather than their magnitude, making it less sensitive to differences in document length. The result is a value between $[0, 1]$, where $1$ indicates maximum similarity.

### 3. **Search Functionality with Title Priority (`search_data_with_title_priority`)**
* **Purpose:** To implement the search logic that returns the most relevant results while **prioritizing** films whose titles directly match keywords from the query.
* **Implementation:**
    * The user query is converted into a **TF-IDF vector** (`query_vector`) using the fitted `TFIDF_VECTORIZER`.
    * **Cosine similarities** are calculated between the query vector and all movie description vectors.
    * A **prioritization mechanism** is introduced by adding the `title_match_priority` column. If the query string (`search_string`) is contained within the movie's title (`title_column`), the film receives a **high, fixed priority score (1000)**.
    * The final sorting is **two-tiered**: first by the **`title_match_priority`** column (descending), and then by the **`similarity`** score (cosine similarity, descending). This ensures that films with a title match always appear at the top, and their internal order is determined by their contextual similarity.

### 4. **Interactive Search Interface (`interactive_search`)**
* **Purpose:** To provide a simple, command-line interface (CLI) for entering queries and conveniently visualizing the results.
* **Implementation:** The function prompts the user for a query, calls the title-priority search logic, and displays the **top results (`top_n`)** in a clear tabular (Markdown) format. This includes both the movie title and the calculated **Cosine Similarity Score**, rounded to 4 decimal places.

---

## **Evaluation and PoC Conclusions**
The implementation of TF-IDF vectorization and cosine similarity, combined with the prioritization logic, is **fully functional** and meets the PoC objectives. The system effectively retrieves relevant movie titles based on user queries. Key takeaways:

* **Efficiency:** The use of TF-IDF correctly represents the content of the films, and cosine similarity enables precise ranking.
* **Result Accuracy:** The implemented `title_match_priority` mechanism is crucial—it significantly improves the **utility** of the results by guaranteeing that films whose titles contain the key search terms are presented first, which is often the desired behavior in search systems.

This PoC forms a **solid basis** for the further development of the search system and confirms that the selected machine learning techniques are appropriate for the information retrieval task within the context of movie data.



