import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple

import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


### File IO and processing

class Document(NamedTuple):
    doc_id: int
    author: List[str]
    title: List[str]
    keyword: List[str]
    abstract: List[str]

    def sections(self):
        return [self.author, self.title, self.keyword, self.abstract]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  author: {self.author}\n" +
            f"  title: {self.title}\n" +
            f"  keyword: {self.keyword}\n" +
            f"  abstract: {self.abstract}")


def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

stopwords = read_stopwords('common_words')

stemmer = SnowballStemmer('english')

def read_rels(file):
    '''
    Reads the file of related documents and returns a dictionary of query id -> list of related documents
    '''
    rels = {}
    with open(file) as f:
        for line in f:
            qid, rel = line.strip().split()
            qid = int(qid)
            rel = int(rel)
            if qid not in rels:
                rels[qid] = []
            rels[qid].append(rel)
    return rels

def read_docs(file):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = [defaultdict(list)]  # empty 0 index
    category = ''
    with open(file) as f:
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                i = int(line[3:])
                docs.append(defaultdict(list))
            elif re.match(r'\.\w', line):
                category = line[1]
            elif line != '':
                for word in word_tokenize(line):
                    docs[i][category].append(word.lower())

    return [Document(i + 1, d['A'], d['T'], d['K'], d['W'])
        for i, d in enumerate(docs[1:])]

def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) for word in sec]
        for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, *[[word for word in sec if word not in stopwords]
        for sec in doc.sections()])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]



### Term-Document Matrix

class TermWeights(NamedTuple):
    author: float
    title: float
    keyword: float
    abstract: float

def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list):
    vec = defaultdict(float)
    for word in doc.author:
        vec[word] += weights.author
    for word in doc.keyword:
        vec[word] += weights.keyword
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.abstract:
        vec[word] += weights.abstract
    return dict(vec)  # convert back to a regular dict

def compute_tfidf(doc, doc_freqs, weights):
    vec = defaultdict(float)
    doc_num = max(doc_freqs.values())
    tf = compute_tf(doc, doc_freqs, weights)
    for word in doc.author:
        vec[word] += tf[word] * (np.log(doc_num / (1+doc_freqs[word])))
    for word in doc.keyword:
        vec[word] += tf[word] * (np.log(doc_num / (1+doc_freqs[word])))
    for word in doc.title:
        vec[word] += tf[word] * (np.log(doc_num / (1+doc_freqs[word])))
    for word in doc.abstract:
        vec[word] += tf[word] * (np.log(doc_num / (1+doc_freqs[word])))
    return dict(vec) #TODO: implement

def compute_boolean(doc, doc_freqs, weights):
    vec = defaultdict(float)
    for word in doc.author:
        if doc_freqs[word] > 1:
            vec[word] = 1
        else:
            vec[word] = 0
    for word in doc.keyword:
        if doc_freqs[word] > 1:
            vec[word] = 1
        else:
            vec[word] = 0
    for word in doc.title:
        if doc_freqs[word] > 1:
            vec[word] = 1
        else:
            vec[word] = 0
    for word in doc.abstract:
        if doc_freqs[word] > 1:
            vec[word] = 1
        else:
            vec[word] = 0
    return dict(vec) # TODO: implement



### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))

def dice_sim(x, y):
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (sum(list(x.values())) * sum(list(y.values())))  # TODO: implement

def jaccard_sim(x, y):
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / ((sum(list(x.values())) * sum(list(y.values()))) + 0.000001)  # TODO: implement

def overlap_sim(x, y):
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / min(sum(list(x.values())) , sum(list(y.values())))  # TODO: implement


### Precision/Recall

def interpolate(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b

def precision_at(recall: float, results: List[int], relevant: List[int]) -> float:
    '''
    This function should compute the precision at the specified recall level.
    If the recall level is in between two points, you should do a linear interpolation
    between the two closest points. For example, if you have 4 results
    (recall 0.25, 0.5, 0.75, and 1.0), and you need to compute recall @ 0.6, then do something like

    interpolate(0.5, prec @ 0.5, 0.75, prec @ 0.75, 0.6)

    Note that there is implicitly a point (recall=0, precision=1).

    `results` is a sorted list of document ids
    `relevant` is a list of relevant documents
    '''
    ranks = []
    
    for i in range(len(results)):
        if results[i] in relevant:
            ranks.append(i + 1)


    recalls = [0]
    precisions = [1]
    counter = 1
    for rank in ranks:
        recalls.append(float(counter) / float(len(relevant)))
        precisions.append(float(counter) / float(rank))
        counter += 1

    if recall in recalls:
        alreadythere = precisions[recalls.index(recall)]
        return alreadythere
    else:
        right_recall_id = np.searchsorted(recalls, recall, side='left', sorter=None)
        left_recall_id = right_recall_id - 1
        inter = interpolate(recalls[left_recall_id], precisions[left_recall_id],
                           recalls[right_recall_id], precisions[right_recall_id],
                           recall)
        return inter
            

def mean_precision1(results, relevant):
    return (precision_at(0.25, results, relevant) +
        precision_at(0.5, results, relevant) +
        precision_at(0.75, results, relevant)) / 3

def mean_precision2(results, relevant):
    for i in range(1,11):
        sum1 = sum([precision_at(i/10, results, relevant)])
    return sum1/10 # TODO: implement

def norm_recall(results, relevant):
    num_relevant = len(relevant)
    num_results = len(results)

    # Calculate the sum of ranks for relevant documents in the result list
    sum_rank_relevant = 0
    for i in range(1, num_relevant + 1):
        sum_rank_relevant += results.index(i)

    # Calculate the sum of ranks from 1 to the number of relevant documents
    sum_rank_all = 0
    for i in range(1, num_relevant + 1):
        sum_rank_all += i

    numerator = sum_rank_relevant - sum_rank_all
    denominator = num_relevant * (num_results - num_relevant)

    if denominator != 0:
        normalized_recall = 1 - (numerator / denominator)
    else:
        normalized_recall = 0

    return normalized_recall # TODO: implement

def norm_precision(results, relevant):
    num_relevant = len(relevant)
    num_results = len(results)

    # Calculate the sum of log2 ranks for relevant documents in the result list
    sum_log_rank_relevant = 0
    for i in range(1, num_relevant + 1):
        index_i = results.index(i)
        if index_i != 0:
            sum_log_rank_relevant += np.log2(index_i)

    # Calculate the sum of log2 ranks from 1 to the number of relevant documents
    sum_log_rank_all = 0
    for i in range(1, num_relevant + 1):
        if i != 0:
            sum_log_rank_all += np.log2(i)

    numerator = sum_log_rank_relevant - sum_log_rank_all
    denominator = num_results * np.log2(num_results) - (num_results - num_relevant) * np.log2(num_results - num_relevant) - num_relevant * np.log2(num_relevant) if num_results != num_relevant and num_results - num_relevant != 0 and num_relevant != 0 else 1

    normalized_precision = (1 - (numerator / denominator)) if denominator != 0 else 0
    return normalized_precision # TODO: implement


### Extensions

# TODO: put any extensions here

document_words = sorted({word for doc in read_docs('cacm.raw') for sec in doc.sections() for word in sec})

def reduce(document_vectors, query_vectors):
    doc_word_presence = [
        [vec[word] if word in vec else 0 for word in document_words]
        for vec in document_vectors
    ]
    doc_size = len(doc_word_presence)
    
    query_word_presence = [
        [vec[word] if word in vec else 0 for word in document_words]
        for vec in query_vectors
    ]

    tmp_matrix = np.asmatrix(doc_word_presence + query_word_presence)
    final_matrix, indices = calculate_svd(tmp_matrix, int(len(document_words) * 0.9))
    
    doc_matrix, query_matrix = final_matrix[:doc_size], final_matrix[doc_size:]
    
    new_doc_vectors = [
        {word: doc_matrix[i, j] for j, word in enumerate(document_words) if doc_matrix[i, j] != 0}
        for i in range(doc_matrix.shape[0])
    ]

    new_query_vectors = [
        {word: query_matrix[i, j] for j, word in enumerate(document_words) if query_matrix[i, j] != 0}
        for i in range(query_matrix.shape[0])
    ]

    return new_doc_vectors, new_query_vectors


def calculate_svd(input_matrix, num_components):
    num_rows, num_columns = input_matrix.shape
    mean_vector = np.mean(input_matrix, axis=0)
    centered_data = input_matrix - np.tile(mean_vector, (num_rows, 1))
    eigenvalues, eigenvectors = np.linalg.eig(np.cov(centered_data.T))
    sorted_indices = np.argsort(-eigenvalues)

    if num_components > num_columns:
        print("Number of components must be lower than the feature number")
        return

    selected_vectors = np.matrix(eigenvectors.T[sorted_indices[:num_components]])
    transformed_data = centered_data @ selected_vectors.T
    reconstructed_data = (transformed_data @ selected_vectors) + mean_vector

    return reconstructed_data, sorted_indices[:num_components]


def svdexp():
    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')
    
    term_funcs = {
        'tfidf': compute_tfidf,
    }

    sim_funcs = {
        'cosine': cosine_sim,
    }

    permutations = [
        term_funcs,
        [False, True],  # stem
        [False, True],  # remove stopwords
        sim_funcs,
        [TermWeights(author=1, title=1, keyword=1, abstract=1),
            TermWeights(author=1, title=3, keyword=4, abstract=1),
            TermWeights(author=1, title=1, keyword=1, abstract=4)]
    ]

    print('term', 'stem', 'removestop', 'sim', 'termweights', 'p_0.25', 'p_0.5', 'p_0.75', 'p_1.0', 'p_mean1', 'p_mean2', 'r_norm', 'p_norm', sep='\t')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]
        query_vec = [term_funcs[term](query, doc_freqs, term_weights) for query in processed_docs]
        query_ids = [query.doc_id for query in processed_queries]
        doc_vectors, query_vectors = reduce(doc_vectors, query_vec)
        metrics = []
        for index, query_vec in enumerate(query_vectors):
            results = search(doc_vectors, query_vec, sim_funcs[sim])
            rel = rels[query_ids[index]]
            metrics.append([
                precision_at(0.25, results, rel),
                precision_at(0.5, results, rel),
                precision_at(0.75, results, rel),
                precision_at(1.0, results, rel),
                mean_precision1(results, rel),
                mean_precision2(results, rel),
                norm_recall(results, rel),
                norm_precision(results, rel)
            ])

        averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}'
            for i in range(len(metrics[0]))]
        print(term, stem, removestop, sim, ','.join(map(str, term_weights)), *averages, sep='\t')

    return  # TODO: just for testing; remove this when printing the full table


def process_docs_and_queries(docs, queries, stem, removestop, stopwords):
    processed_docs = docs
    processed_queries = queries
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
        processed_queries = remove_stopwords(processed_queries)
    if stem:
        processed_docs = stem_docs(processed_docs)
        processed_queries = stem_docs(processed_queries)
    return processed_docs, processed_queries


def search(doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
    return results


def search_debug(docs, query, relevant, doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]

    print('Query:', query)
    print('Relevant docs: ', relevant)
    print()
    for doc_id, score in results_with_score[:10]:
        print('Score:', score)
        print(docs[doc_id - 1])
        print()


if __name__ == '__main__':
    svdexp()