# Information Retrieval - Search-Engine - Final Project

In our Information Retrieval project, we constructed a specialized search engine tailored for English Wikipedia. This engine swiftly retrieves the top 100 relevant articles by employing a scoring system we crafted. To enhance accuracy, we integrated various Similarity Measures into our search algorithm. These measures enable fine-tuning of search results based on relevance and similarity to the query. Our project aims to provide users with precise and efficient access to the vast wealth of information available on Wikipedia:

We use the following methods for scoring

* Inverted Index for each wiki section: title, body text, anchor text
* BM25
* page view
* page rank

In addition, we used cosine similarity tf-idf but get poor results versus BM25
  
  
## Description

Files Description:

- **search_frontend.py**: python file which contains the code to run server locally on a host by running Flask.

- **search_backend.py**: python file which contains backend functions and logic.

- **inverted_index_gcp.py**: python file which contains Inverted Index class with functions for writing, reading and updating inverted index and posting lists

- **PageRank_PageViews_Support_Dictionaries.ipynb**: Jupyter notebook which should contains the functions to build support files for the engine such as: pagerank calculation, pageviews, map between document and its length and more.

- **IndexBuilder_title.ipynb**: Jupyter notebook which used build the title inverted index.

- **IndexBuilder_anchor.ipynb**: Jupyter notebook was used to build the anchor inverted index.
- **IndexBuilder_text.ipynb** Jupyter notebook was used to build the body text inverted index.
  
## Credits

The project utilizes the following libraries:

- **Spark**: Employed as a robust data processing engine, facilitating distributed data processing operations
- **NumPy**: A library for scientific computing with Python that provides support for large, multi-dimensional arrays and matrices.
- **NLTK**: An indispensable toolkit for natural language processing, equipped with various utilities tailored for text data manipulation and analysis.


