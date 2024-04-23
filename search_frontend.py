from flask import Flask, request, jsonify
from inverted_index_gcp import InvertedIndex
from search_backend import *
import pandas as pd
import pickle
import gzip
from urllib.parse import quote


textInvertedIndex = InvertedIndex().read_index("", 'dict_text_index', 'text311394365')

titleInvertedIndex = InvertedIndex().read_index("", 'dict_title_index', '311394365')

with gzip.open('pr_part-00000-e07ba620-73cf-4b24-82a8-73b28232722d-c000.csv.gz') as f:
    pr = pd.read_csv(f, header=None)
page_rank_score_dict = pr.set_index(0).to_dict()[1]

with (open('docs_to_title_dict.pkl', "rb")) as f:
    docID_title_dict = pickle.load(f)

with (open('pageviews-20231206-user.pkl', "rb")) as f:
    pv_dict = pickle.load(f)

with (open('DL_dict.pkl', "rb")) as f:
    docs_len_dict = pickle.load(f)
    
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_tokens = tokenize(query)
    return jsonify(BM25text_title_with_pagerank(query_tokens,titleInvertedIndex, textInvertedIndex,docs_len_dict, docID_title_dict,page_rank_score_dict))


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
