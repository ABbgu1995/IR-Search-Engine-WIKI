from nltk.corpus import stopwords
from inverted_index_gcp import *
import numpy as np
import nltk
from contextlib import closing
import math
import re
NUM_OF_DOCUMENTS = 6348910
BM25_K1 = 3  
BM25_b = 0.13  


nltk.download('stopwords')


def tokenize(text):
    """
    Tokenizes the input text and filters out stopwords.

    Parameters:
        text (str): The text to tokenize and filter.

    Returns:
        list: A list of tokens with stopwords removed.
    """
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]
    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens

def search_BM25_with_normalized(query_tokens, index, docs_len_dict):
    """
    Calculate BM25 scores for all relevant documents to a given query.

    Params:
    query_tokens: Query tokens.
    index: Relevant index for search (textIndex).
    docs_len_dict: Dictionary {doc_id: document length (in tokens)}.

    Returns:
    Sorted dictionary by BM25 score {doc_id: BM25 score}.
    """
    bm25_doc_scores = {}  
    idf = calc_idf(query_tokens, index)
    average_doc_len = sum(docs_len_dict.values()) / NUM_OF_DOCUMENTS
    for term in query_tokens:
        try:
            term_frequencies = dict(index.read_a_posting_list("",term,"text311394365"))
        except:
            continue
        for doc_id, tf in term_frequencies.items():  # tuple of (doc_id, tf)
            if doc_id not in bm25_doc_scores.keys():
                bm25_doc_scores[doc_id] = 0
            freq = term_frequencies[doc_id]
            denominator = freq + BM25_K1 * (1 - BM25_b + BM25_b * docs_len_dict.get(doc_id) / average_doc_len)
            numerator = idf[term] * freq * (BM25_K1 + 1)
            bm25_doc_scores[doc_id] += (numerator / denominator)
    sorted_scores = {key: value for key, value in
                     sorted(bm25_doc_scores.items(), key=lambda item: item[1], reverse=True)}
    bm25_doc_scores_normalized = {}
    if len(sorted_scores) == 0:
        return  bm25_doc_scores_normalized
    bm25_score_normalized_min = min(list(sorted_scores.values())[:100])
    bm25_score_normalized_max = max(list(sorted_scores.values()))
    for doc_id, score in sorted_scores.items():
        x = sorted_scores[doc_id]
        bm25_doc_scores_normalized[doc_id] = (x - bm25_score_normalized_min) / (bm25_score_normalized_max - bm25_score_normalized_min)
    return bm25_doc_scores_normalized


def get_top_docs_amount_based(scores_dict, amount_top_docs):
    """
    Sort and return the top N documents based on their similarity scores.
    Generate a dictionary of similarity scores.

    Parameters:
    scores_dict: A dictionary containing similarity scores in the format: {doc_id: score}
    amount_top_docs: An integer specifying the number of documents to retrieve. 

    Returns:
    A ranked list of (doc_id, score) pairs with a length of amount_top_docs.
    """
    return sorted([(doc_id, score) for doc_id, score in scores_dict.items()], key=lambda x: x[1], reverse=True)[:amount_top_docs]


def BM25text_title_with_pagerank(query_tokens,title_inverted_index, text_inverted_index, docs_len_dict,docid_title_dict,pagerank_score_dict):
    """
    Retrieve the top-ranked documents based on BM25 scores for both document titles and body text, combined with PageRank scores.

    Parameters:
    query_tokens: A list of tokens representing the query.
    title_inverted_index: An inverted index for document titles.
    text_inverted_index: An inverted index for document body text.
    docs_len_dict: A dictionary containing the length of each document.
    docid_title_dict: A dictionary mapping document IDs to their titles.
    pagerank_score_dict: A dictionary containing PageRank scores for each document.

    Returns:
    A ranked list of tuples containing document IDs and their corresponding titles, sorted by relevance.
    """
    BM25_score_body_text = search_BM25_with_normalized(query_tokens, text_inverted_index, docs_len_dict)
    if BM25_score_body_text != {}:
        title_top = title_score_normalized(title_inverted_index, query_tokens)
        top100_text = get_top_docs_amount_based(BM25_score_body_text, 100)
        top100_title = get_top_docs_amount_based(title_top, 100)

        res_similarity_scores = weight_similarity_scores(dict(top100_title), dict(top100_text))
        total_score = merge_pr_scores(pagerank_score_dict, res_similarity_scores)

        total_amount_score = get_top_docs_amount_based(total_score, 100)
        top_doc_title_scored = [(str(item[0]), docid_title_dict.get(item[0], 'title not found')) for item in total_amount_score]
    else:
        top_doc_title_scored = []
    return top_doc_title_scored

def merge_pr_scores(pagerank_dict, similarity_scores):
    """
    Merge PageRank scores with similarity scores and return the combined scores for each document.

    Parameters:
    pagerank_dict: A dictionary containing PageRank scores for each document.
    similarity_scores: A dictionary containing similarity scores for each document.

    Returns:
    A dictionary containing the merged scores for each document, where the keys represent document IDs and the values represent the combined scores.
    """
    norm_factor = max(pagerank_dict.values())
    total_score_pr_indexes = {}
    for doc_id, score in similarity_scores.items():
        if doc_id in pagerank_dict.keys():
            total_score_pr_indexes[doc_id] = score + (pagerank_dict.get(doc_id) / norm_factor)
    return total_score_pr_indexes



def weight_similarity_scores(title_scores, BM25_scores, title_weight=0.5, bm25_weight=0.5):
    """
    Merge title_scores and BM25_scores dictionaries into one based on index weights

    Params:
    title_scores: Dict of title scores {doc_id: title_score}.
    BM25_scores: Dict of BM25 scores {doc_id: BM25_score}.
    title_weight: Weight for the title in the formula.
    bm25_weight: Weight for the BM25 score in the formula.

    Returns:
    Dict of merged scores {doc_id: merged_score}.
    """
    merged_results = {}

    for doc_id, score in title_scores.items():
        if doc_id not in merged_results.keys():
            merged_results[doc_id] = 0
        merged_results[doc_id] += score * title_weight

    for doc_id, score in BM25_scores.items():
        if doc_id not in merged_results.keys():
            merged_results[doc_id] = 0
        merged_results[doc_id] += score * bm25_weight

    merged_results = {key: value for key, value in
                      sorted(merged_results.items(), key=lambda item: item[1], reverse=True)}
    return merged_results



def title_score_normalized(index, query):
    """
    Identify the most relevant documents based on their titles and return a normalized sorted dictionary.

    Params:
    index: Title inverted index file.
    query: List of query tokens.
    Returns:
    Sorted dictionary in the format: {doc_id: query tokens found in doc's title / query length}. Sorted by values.
    """
    dict_query_docs = {}
    query_len = len(query)
    for term in np.unique(query): 
        try:
            term_frequencies = (index.read_a_posting_list("", term, "311394365"))
        except:
            continue
        for doc_id, tf in term_frequencies:
            if doc_id not in dict_query_docs.keys():
                dict_query_docs[doc_id] = 0
            dict_query_docs[doc_id] += 1 / query_len
    dict_query_docs = {key: value for key, value in
                       sorted(dict_query_docs.items(), key=lambda item: item[1], reverse=True)}
    return dict_query_docs


def calc_idf(query_tokens, inverted_index):
    """
    Compute IDF values using the BM25 IDF formula for query terms.

    Params:
    query_tokens: List of tokens representing the query
    inverted_index: Inverted index object holding document frequencies for terms.

    Returns:
    Dictionary of IDF scores for each query term. Format: {term: BM25 IDF score}.
    """
    idf_scores = {}
    for term in query_tokens:
        if term in inverted_index.df.keys():
            doc_freq_term = inverted_index.df[term]  
            idf_scores[term] = math.log(1 + (NUM_OF_DOCUMENTS - doc_freq_term + 0.5) / (doc_freq_term + 0.5))
        else:
            pass
    return idf_scores

