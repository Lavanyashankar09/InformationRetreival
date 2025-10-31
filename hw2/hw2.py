import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple

import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
# import nltk
# nltk.download('punkt')

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
    # print("weights", weights)
    #weights TermWeights(author=1, title=1, keyword=1, abstract=1)
    for word in doc.author:
        vec[word] += weights.author
    for word in doc.keyword:
        vec[word] += weights.keyword
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.abstract:
        vec[word] += weights.abstract
    # print("vec", vec)
    return dict(vec)  # convert back to a regular dict

def compute_tfidf(doc, doc_freqs, weights):
    vec = defaultdict(float)
    doc_num = max(doc_freqs.values())
    # print("doc_num --------", doc_num)
    tf = compute_tf(doc, doc_freqs, weights)
    # print("tf --------", tf)

    for word in doc.author:
        vec[word] += tf[word] * (np.log(doc_num / (1+doc_freqs[word])))
    for word in doc.keyword:
        vec[word] += tf[word] * (np.log(doc_num / (1+doc_freqs[word])))
    for word in doc.title:
        vec[word] += tf[word] * (np.log(doc_num / (1+doc_freqs[word])))
    for word in doc.abstract:
        vec[word] += tf[word] * (np.log(doc_num / (1+doc_freqs[word])))

    '''
    doc_num -------- 3151
    weights TermWeights(author=1, title=1, keyword=1, abstract=1)
    tf -------- {'perlis': 1.0, ',': 2.0, 'a.': 1.0, 'j': 1.0, '.': 2.0, 'samelson': 1.0, 'k': 1.0, 'preliminary': 1.0, 'report-international': 1.0, 'algebraic': 1.0, 'language': 1.0}
    vec ------  {'perlis': 5.490525784295738, ',': -0.001269236881980585, 'a.': 3.044839847661019, 'j': 1.9440078022545961, '.': 0.022915403220331417, 'samelson': 6.263715672529219, 'k': 3.33697627046218, 'preliminary': 5.0597428682032835, 'report-international': 7.362327961197329, 'algebraic': 4.104231423175848, 'language': 2.2443341487805744})
    '''
    # print("vec ------", vec)
    return dict(vec) #TODO: implement

def compute_boolean(doc, doc_freqs, weights):
    vec = defaultdict(float)
    print("vec", vec)

    '''
    vec defaultdict(<class 'float'>, {'perlis': 1, ',': 1, 'a.': 1, 'j': 1, '.': 1, 'samelson': 1, 'k': 1, 'preliminary': 1, 'report-international': 0, 'algebraic': 1, 'language': 1})
    vec defaultdict(<class 'float'>, {'perlis': 1.0, ',': 2.0, 'a.': 1.0, 'j': 1.0, '.': 2.0, 'samelson': 1.0, 'k': 1.0, 'preliminary': 1.0, 'report-international': 1.0, 'algebraic': 1.0, 'language': 1.0})
    '''
    for word in doc.author:
        # print("word", word)
        # print("doc_freqs[word]", doc_freqs[word])
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

    # print("vec", vec)
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
    # print("x", x)
    # print("y", y)
    num = dictdot(x, y)
    #print("num", num)
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
    # print("x1", x1)
    # print("y1", y1)
    # print("x2", x2)
    # print("y2", y2)
    # print("x", x)
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
    #print("results", results)
    #print("len(results)", len(results))

    # print("relevant", relevant)
    # print("recall", recall)
    #len(results) 3204
    #relevant [1410, 1572, 1605, 2020, 2358]
    #recall 0.25
    ranks = []
    
    for i in range(len(results)):
        if results[i] in relevant:
            ranks.append(i + 1)

    # print("ranks", ranks)
    # print("len(ranks)", len(ranks))

    #ranks [26, 103, 210, 524, 1798]
    #len(ranks) 5

    recalls = [0]
    precisions = [1]
    counter = 1
    for rank in ranks:
        recalls.append(float(counter) / float(len(relevant)))
        precisions.append(float(counter) / float(rank))
        counter += 1

    

    print("recalls", recalls)
    print("precisions", precisions)

    # Plot precision-recall graph
    plt.plot(recalls, precisions, marker='o', linestyle='-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    

    # recalls [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # precisions [1, 0.038461538461538464, 0.019417475728155338, 0.014285714285714285, 0.007633587786259542, 0.0027808676307007787]
   
    if recall in recalls:
        # print("recall", recall)
       # print("recalls first", recalls)
        # print("recalls.index(recall)", recalls.index(recall))

        # recall 1.0
        # recalls [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # recalls.index(recall) 5

        #print("precisions", precisions)
        #precisions [1, 0.038461538461538464, 0.019417475728155338, 0.014285714285714285, 0.007633587786259542, 0.0027808676307007787]
        alreadythere = precisions[recalls.index(recall)]
        #print("alreadythere", alreadythere)
        #alreadythere 0.0027808676307007787
        return alreadythere
    else:
        right_recall_id = np.searchsorted(recalls, recall, side='left', sorter=None)
        # print("recalls second", recalls[:3])
        # print("recall", recall)
        # print("right_recall_id", right_recall_id)
        left_recall_id = right_recall_id - 1
        #print("left_recall_id", left_recall_id)
        inter = interpolate(recalls[left_recall_id], precisions[left_recall_id],
                           recalls[right_recall_id], precisions[right_recall_id],
                           recall)
        # print("recalls[left_recall_id]", recalls[left_recall_id])
        # print("recalls[right_recall_id]", recalls[right_recall_id])
        # print("precisions[left_recall_id]", precisions[left_recall_id])
        # print("precisions[right_recall_id]", precisions[right_recall_id])
        # print("inter", inter)
        # print("recalls", recalls)
        # print("precisions", precisions)
        # recalls [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # precisions [1, 0.038461538461538464, 0.019417475728155338, 0.014285714285714285, 0.007633587786259542, 0.0027808676307007787]

        '''
        recalls[left_recall_id] 0.2
        recalls[right_recall_id] 0.4
        precisions[left_recall_id] 0.038461538461538464
        precisions[right_recall_id] 0.019417475728155338
        inter 0.03370052277819269
        
        '''

        '''
        x1 0.2
        y1 0.038461538461538464
        x2 0.4
        y2 0.019417475728155338
        x 0.25
        inter 0.03370052277819269
        
        '''
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
    print("num_relevant", num_relevant)
    print("num_results", num_results)

    # Calculate the sum of ranks for relevant documents in the result list
    sum_rank_relevant = 0
    for i in range(1, num_relevant + 1):
        sum_rank_relevant += results.index(i)

    print("sum_rank_relevant", sum_rank_relevant)
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


### Search

def experiment():

    queries = read_docs('query.raw')

    docs = read_docs('cacm.raw')
    # print("docs", type(docs))
    # print("docs", docs[0])
    # print("docs", docs[1])
    # print("docs", docs[2])

    # docs <class 'list'>

    """

    .ID 1
    .Title
    Preliminary Report-International Algebraic Language
    .Author
    Perlis, A. J.
    Samelson,K.
    .I 2
    .T
    Extraction of Roots by Repeated Subtractions for Digital Computers
    .A
    Sugai, I.
    .I 3
    .T
    Techniques Department on Matrix Program Schemes
    .A
    Friedman, M. D.

    docs doc_id: 1
    author: ['perlis', ',', 'a.', 'j', '.', 'samelson', ',', 'k', '.']
    title: ['preliminary', 'report-international', 'algebraic', 'language']
    keyword: []
    abstract: []
    docs doc_id: 2
    author: ['sugai', ',', 'i', '.']
    title: ['extraction', 'of', 'roots', 'by', 'repeated', 'subtractions', 'for', 'digital', 'computers']
    keyword: []
    abstract: []
    docs doc_id: 3
    author: ['friedman', ',', 'm.', 'd', '.']
    title: ['techniques', 'department', 'on', 'matrix', 'program', 'schemes']
    keyword: []
    abstract: []

    """


    

    queries = read_docs('query.raw')
    # print("queries", type(queries))
    # print("queries", queries[0])
    # print("queries", queries[1])
    # print("queries", queries[2])

    '''
    .I 1
    .W - abstract
    What articles exist which deal with TSS (Time Sharing System), an
    operating system for IBM computers?


    .I 2
    .W
    I am interested in articles written either by Prieve or Udo Pooch
    .A
    Prieve, B.
    Pooch, U.


    .I 3
    .W
    Intermediate languages used in construction of multi-targeted compilers; TCOLL

    queries doc_id: 1
    author: []
    title: []
    keyword: []
    abstract: ['what', 'articles', 'exist', 'which', 'deal', 'with', 'tss', '(', 'time', 'sharing', 'system', ')', ',', 'an', 'operating', 'system', 'for', 'ibm', 'computers', '?']
    queries doc_id: 2
    author: ['prieve', ',', 'b', '.', 'pooch', ',', 'u', '.']
    title: []
    keyword: []
    abstract: ['i', 'am', 'interested', 'in', 'articles', 'written', 'either', 'by', 'prieve', 'or', 'udo', 'pooch']
    queries doc_id: 3
    author: []
    title: []
    keyword: []
    abstract: ['intermediate', 'languages', 'used', 'in', 'construction', 'of', 'multi-targeted', 'compilers', ';', 'tcoll']
    
    '''

    rels = read_rels('query.rels')
    # print("rels", rels)

    '''
    01 1410
    01 1572
    01 1605
    01 2020
    01 2358
    02 2434
    02 2863
    02 3078

    rels {1: [1410, 1572, 1605, 2020, 2358], 2: [2434, 2863, 3078], 3: [1134, 1613, 1807, 1947, 2290, 2923],

    query 1 is related to documents 1410, 1572, 1605, 2020, 2358 in cacm.raw
    
    my 1st query 
    .I 16
    .W
    find all descriptions of file handling in operating systems based on
    multiple processes and message passing.
    

    my related documents
    .I 1605
    .T
    An Experimental Comparison of Time Sharing and Batch Processing
    .W
    The effectiveness for program development
    of the MIT Compatible Time-Sharing System (CTSS)

    '''

    # there are just 33 queries
    # there are 448 related documents to this queries
    # there are 3204 documents



    stopwords = read_stopwords('common_words')
    # print("stopwords", stopwords)
    #stopwords {'also', 'that', 'because', 'second', 'however', 'necessary', 'que', 'hers',
    
    term_funcs = {
         #'tf': compute_tf,
         'tfidf': compute_tfidf,
        #'boolean': compute_boolean
    }
    
    print("term_funcs", term_funcs)
    
    sim_funcs = {
        'cosine': cosine_sim,
        # 'jaccard': jaccard_sim,
        # 'dice': dice_sim,
        # 'overlap': overlap_sim
    }

    permutations = [
        term_funcs,
        [False, True],  # stem
        [False, True],  # remove stopwords
        sim_funcs,
        [
            TermWeights(author=1, title=1, keyword=1, abstract=1),
            TermWeights(author=1, title=3, keyword=4, abstract=1),
            TermWeights(author=1, title=1, keyword=1, abstract=4)
        ]
    ]

    print('term', 'stem', 'removestop', 'sim', 'termweights', 'p_0.25', 'p_0.5', 'p_0.75', 'p_1.0', 'p_mean1', 'p_mean2', 'r_norm', 'p_norm', sep='\t')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):
        
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = []
        for doc in processed_docs:
            term_vector = term_funcs[term](doc, doc_freqs, term_weights)
            doc_vectors.append(term_vector)
        
        # print("processed_docs", type(processed_docs))
        # print("processed_docs", len(processed_docs))
        #print("first processed_docs", processed_docs[0])
        '''
        doc_id: 1
        author: ['per', ',', 'a.', 'j', '.', 'samelson', ',', 'k', '.']
        title: ['preliminari', 'report-intern', 'algebra', 'languag']
        keyword: []
        abstract: []

        '''
        
        #doc_freqs = compute_doc_freqs(processed_docs)
        
        
        # print("doc_freqs", type(doc_freqs))
        # print("doc_freqs:", list(doc_freqs.items())[0])  # Print the first element
        # print("doc_freqs:", type(doc_freqs))
        # # Iterate over all items in doc_freqs
        # for word, frequency in doc_freqs.items():
        #     if word == "perlis":
        #         print(f"Word: {word}, Frequency: {frequency}")

        # print("Number of elements in doc_freqs:", len(doc_freqs))  # Print the number of elements in doc_freqs
        
        
            

        # # Iterate over processed_docs and compute term vectors for each document
        # for doc in processed_docs:
        #     # Compute term vector for the current document
        #     #print("doc-----------------------", doc)
        #     #print("doc_freqs", doc_freqs)
        #     #print("term_weights-----------------------", term_weights)
           
        #     term_vector = term_funcs[term](doc, doc_freqs, term_weights)

        #     #print("term_vector-----------------------", term_vector)
        #     #print("term-----------------------", term)
        #     # Append the term vector to doc_vectors list
        #     doc_vectors.append(term_vector)
        #     #print("doc_vectors-----------------------", doc_vectors)

        
        # print("doc_vectors", type(doc_vectors))
        # print("doc_vectors", len(doc_vectors))


        #print("first doc_vectors", doc_vectors[0])
        '''
        first doc_vectors 
        {'perlis': 1.0, 
        ',': 2.0, 
        'a.': 1.0, 
        'j': 1.0, 
        '.': 2.0, 
        'samelson': 1.0, 'k': 1.0, 
        'preliminary': 1.0, 
        'report-international': 1.0, 
        'algebraic': 1.0, 
        'language': 1.0}
        '''
        metrics = []
        '''
        processed_docs <class 'list'>
        processed_docs 3204
        doc_freqs <class 'collections.Counter'>
        doc_freqs 13587
        doc_vectors <class 'list'>
        doc_vectors 3204
        '''
        for query in processed_queries:

            query_vec = term_funcs[term](query, doc_freqs, term_weights)
            #print("query_vec", query_vec)
            results = search(doc_vectors, query_vec, sim_funcs[sim])

            # results = search_debug(processed_docs, query, rels[query.doc_id], doc_vectors, query_vec, sim_funcs[sim])

            rel = rels[query.doc_id]

            # print("doc-id", query.doc_id)
            # print("rel", rel)
            # doc-id 1
            # rel [1410, 1572, 1605, 2020, 2358]
            # print("*"*10)
            # print("results", results[1:3])
            # print("rel", rel)
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
            #print("metrics", metrics)

        averages = []
        for i in range(len(metrics[0])):
            metric_values = []
            for metric in metrics:
                metric_values.append(metric[i])
            avg = np.mean(metric_values)
            avg_formatted = f'{avg:.4f}'
            averages.append(avg_formatted)

        #print(averages)
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
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    results_with_score = []
    for doc_id, doc_vec in enumerate(doc_vectors):
        similarity_score = sim(query_vec, doc_vec)
        results_with_score.append((doc_id + 1, similarity_score))

   # print("results_with_score", results_with_score[0:3])
    #print("*"*10)
    #[(1, 1.6143239751858961e-09), (2, 0.04160422521356425), (3, 5.026530810514811e-10), (4, 0), (5, 4.723298174257119e-10), (6, 0.05443891762226067), (7, 0), (8, 4.957962014851035e-10),
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    #print("results_with_score", results_with_score[0:3])
    results = [x[0] for x in results_with_score]
    #print("results", results[0:3])
    #results_with_score [(2319, 0.3566901441533025), (1033, 0.3341573600063747), (1523, 0.3281895412142857)]
    #results [2319, 1033, 1523]
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
    experiment()