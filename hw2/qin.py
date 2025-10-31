# %%
import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple

import numpy as np
from numpy.linalg import norm
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

# %%
nltk.download('punkt_tab')

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

def compute_tfidf(doc, doc_freqs, weights): # TODO
    # TF-IDF (Term Frequency-Inverse Document Frequency) vector
    N = len(doc_freqs)  # Total number of documents (estimate based on max doc frequency)
    tf_vector = compute_tf(doc, doc_freqs, weights)
    tfidf_vector = {}
    
    for term, tf in tf_vector.items():
        # Calculate inverse document frequency
        # Add 1 to avoid division by zero
        idf = np.log10(N / (doc_freqs[term] + 1))
        tfidf_vector[term] = tf * idf
        
    return tfidf_vector
    

def compute_boolean(doc, doc_freqs, weights): # TODO
    # Applies weights based on the section where terms appear.
    vec = defaultdict(float)
    
    # Add weighted boolean values for terms in each section
    for word in set(doc.author):
        vec[word] = weights.author
    
    for word in set(doc.title):
        vec[word] = max(vec[word], weights.title)
    
    for word in set(doc.keyword):
        vec[word] = max(vec[word], weights.keyword)
    
    for word in set(doc.abstract):
        vec[word] = max(vec[word], weights.abstract)
    
    return dict(vec)



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

def dice_sim(x, y): # TODO
    """
    Computes Dice's coefficient between two vectors x and y.
    Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
    """
    intersection = sum(min(x.get(key, 0), y.get(key, 0)) for key in set(x) & set(y))
    return (2 * intersection) / (sum(x.values()) + sum(y.values()))

def jaccard_sim(x, y):
    """
    Computes Jaccard similarity between two vectors x and y.
    Jaccard = |X ∩ Y| / |X ∪ Y|
    """
    # For the intersection, we need to consider minimum values
    intersection = sum(min(x.get(key, 0), y.get(key, 0)) for key in set(x) & set(y))
    
    # For the union, we need to consider maximum values
    union = sum(max(x.get(key, 0), y.get(key, 0)) for key in set(x) | set(y))
    
    if union == 0:
        return 0
    
    return intersection / union

def overlap_sim(x, y): # TODO
    """
    Computes overlap similarity between two vectors x and y.
    Overlap = |X ∩ Y| / min(|X|, |Y|)
    """
    intersection = sum(min(x.get(key, 0), y.get(key, 0)) for key in set(x) & set(y))
    
    x_magnitude = sum(x.values())
    y_magnitude = sum(y.values())
    
    denominator = min(x_magnitude, y_magnitude)
    
    if denominator == 0:
        return 0
    
    return intersection / denominator


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

    if not relevant:  # No relevant documents
        return 1.0
    
    num_relevant = len(relevant)
    relevant_set = set(relevant)
    
    # Calculate precision and recall at each position
    recalls = [0.0]  # Initial point (0 recall, 1 precision)
    precisions = [1.0]
    
    num_relevant_found = 0
    
    for i, doc_id in enumerate(results):
        position = i + 1  # 1-based indexing for position
        
        if doc_id in relevant_set:
            num_relevant_found += 1
        
        current_precision = num_relevant_found / position
        current_recall = num_relevant_found / num_relevant
        
        recalls.append(current_recall)
        precisions.append(current_precision)
    
    # Add final point (1.0 recall, final precision)
    if recalls[-1] < 1.0:
        recalls.append(1.0)
        precisions.append(precisions[-1])  # Use the last precision value
    
    # Find the precision at the specified recall level through interpolation
    for i in range(len(recalls) - 1):
        if recalls[i] <= recall <= recalls[i + 1]:
            if recalls[i] == recall:
                return precisions[i]
            if recalls[i + 1] == recall:
                return precisions[i + 1]
            
            # Interpolate between the two closest points
            return interpolate(recalls[i], precisions[i], recalls[i + 1], precisions[i + 1], recall)
    
    # If we don't find the exact recall level, return the precision at the closest recall point
    return precisions[-1]  # Should not reach here if implementation is correct
    # return 1  # TODO: implement

def mean_precision1(results, relevant):
    return (precision_at(0.25, results, relevant) +
        precision_at(0.5, results, relevant) +
        precision_at(0.75, results, relevant)) / 3

def mean_precision2(results, relevant): # TODO
    recall_levels = [i / 10 for i in range(11)]  # 0.0, 0.1, ..., 1.0
    
    return sum(precision_at(r, results, relevant) for r in recall_levels) / len(recall_levels)

def norm_recall(results, relevant): # TODO
    if not relevant:
        return 1.0  # No relevant documents, perfect recall
    
    # Create a set for O(1) lookups
    relevant_set = set(relevant)
    
    # Calculate the actual sum of ranks of relevant documents
    actual_sum = sum(i + 1 for i, doc_id in enumerate(results) if doc_id in relevant_set)
    
    # Calculate the best possible sum (relevant docs ranked first)
    best_sum = sum(range(1, len(relevant) + 1))
    
    # Calculate the worst possible sum (relevant docs ranked last)
    n = len(results)
    worst_sum = sum(range(n - len(relevant) + 1, n + 1))
    
    # Handle edge case where best and worst are the same
    if worst_sum == best_sum:
        return 1.0
    
    # Normalize the sum (1 for best ranking, 0 for worst)
    return (worst_sum - actual_sum) / (worst_sum - best_sum)

def norm_precision(results, relevant): # TODO
    """
    Computes normalized precision - the number of relevant documents retrieved
    normalized by the number of relevant documents that exist
    """
    if not relevant:
        return 1.0  # No relevant documents, perfect precision
    
    # Create a set for O(1) lookups
    relevant_set = set(relevant)
    
    # Count relevant documents in the results
    num_relevant_retrieved = sum(1 for doc_id in results if doc_id in relevant_set)
    
    # Normalize by total number of relevant documents
    return num_relevant_retrieved / len(relevant)


### Extensions

# TODO: put any extensions here


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


# %% [markdown]
# ## Experiment Function and Test

# %%
docs = read_docs('cacm.raw')
queries = read_docs('query.raw')
rels = read_rels('query.rels')
stopwords = read_stopwords('common_words')

term_funcs = {
    'tf': compute_tf,
    'tfidf': compute_tfidf,
    'boolean': compute_boolean
}

sim_funcs = {
    'cosine': cosine_sim,
    'jaccard': jaccard_sim,
    'dice': dice_sim,
    'overlap': overlap_sim
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

    metrics = []

    for query in processed_queries:
        query_vec = term_funcs[term](query, doc_freqs, term_weights)
        results = search(doc_vectors, query_vec, sim_funcs[sim])
        # results = search_debug(processed_docs, query, rels[query.doc_id], doc_vectors, query_vec, sim_funcs[sim])
        rel = rels[query.doc_id]

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

# TODO: just for testing; remove this when printing the full table


# %%



