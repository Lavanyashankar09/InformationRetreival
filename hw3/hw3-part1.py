from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple
import csv
import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

import itertools as it

### File IO and processing
class Document(NamedTuple):
    doc_id: int
    sen: List[str]
    output: int

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  sen: {self.sen}\n" +
            f"  output: {self.output}\n") 


def process_ambiguous_words(line):
    words = word_tokenize(line)
    # Find the index of the ambiguous word
    ambiguous_index = -1
    for i, word in enumerate(words):
        if len(word) > 3 and word.startswith(".X-"):
            ambiguous_index = i
            break
    # Add special adjacent tokens if an ambiguous word is found
    if ambiguous_index != -1:
        if ambiguous_index > 0:
            words[ambiguous_index - 1] = "L-" + words[ambiguous_index - 1]
        if ambiguous_index + 1 < len(words):
            words[ambiguous_index + 1] = "R-" + words[ambiguous_index + 1]
    return words



def process_documents(file_path, tokenization_mode):
    documents = []
    with open(file_path, "r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for row in tsv_reader:
            doc_id = int(row[0])
            text = row[2]
            label = int(row[1])

            if tokenization_mode == 1:
                tokens = word_tokenize(text)
            else:  # Special tokenization mode
                tokens = process_ambiguous_words(text)

            documents.append(Document(doc_id, tokens, label))
    return documents

def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])
    
stopwords = read_stopwords('common_words')

stemmer = SnowballStemmer('english')

def stem_doc(doc: Document):
    stemmed_sentence = [stemmer.stem(word) for word in doc.sen]
    return Document(doc.doc_id, stemmed_sentence, doc.output)


def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    pruned_sentence = [word for word in doc.sen if word not in stopwords]
    return Document(doc.doc_id, pruned_sentence, doc.output)


def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]



def index_ambiguous(sentence):
    for i, word in enumerate(sentence):
        if len(word) > 3 and word.startswith(".X-"):
            return i
    return -1

def uniform(sentence):
    return [1] * len(sentence)

def decay(sentence):
    weights = []
    pos_amb = index_ambiguous(sentence)

    if pos_amb == -1: return uniform(sentence)

    for i in range(len(sentence)):
        dist = abs(pos_amb - i)
        weights.append(0 if dist == 0 else 1 / dist)

    return weights


def stepped(sentence):
    weights = []
    pos_amb = index_ambiguous(sentence)
    
    if pos_amb == -1:
        return uniform(sentence)

    for i in range(len(sentence)):
        dist = abs(pos_amb - i)
        if dist == 0:
            weights.append(0)
        elif dist == 1:
            weights.append(6)
        elif 2 <= dist <= 3:
            weights.append(3)
        else:
            weights.append(1)
        
    return weights


def better(sentence):
    weights = []
    pos_amb = index_ambiguous(sentence)
    
    if pos_amb == -1:
        return uniform(sentence)
    
    for i in range(len(sentence)):
        dist = abs(pos_amb - i)
        if dist == 0:
            weights.append(0)
        elif dist == 1:
            weights.append(15)
        elif 2 <= dist <= 3:
            weights.append(12 - dist)
        else:
            weights.append(1)

    return weights

### Term-Document Matrix

class TermWeights(NamedTuple):
    uniform: bool
    decay: bool
    stepped: bool
    better: bool

def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for word in doc.sen:
            words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list):
    vec = defaultdict(float)
    
    computed_weights = uniform(doc.sen) if weights.uniform else \
                       decay(doc.sen) if weights.decay else \
                       stepped(doc.sen) if weights.stepped else \
                       better(doc.sen)

    for i, word in enumerate(doc.sen):
        vec[word] += computed_weights[i]  # scale TF weight
    return dict(vec)  # convert back to a regular dict



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

def stem_stop(docs, stem, removestop):
    processed_docs = docs
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
    if stem:
        processed_docs = stem_docs(processed_docs)
    return processed_docs

def label_docs(docs, label):
    return [doc for doc in docs if doc.output == label]

def compute_centroid(label_train_vectors):
    profile = {}
    occurrences = {}
    
    for vec in label_train_vectors:
        for key in vec:
            profile[key] = profile.get(key, 0) + vec[key]
            occurrences[key] = occurrences.get(key, 0) + 1
    
    for k in profile:
        profile[k] /= occurrences[k]
    
    return profile

def evaluate_dev_docs(dev_docs, dev_vectors, profile1, profile2, sim_funcs, sim):
    correct = 0
    incorrect = 0
    for doc, vec in zip(dev_docs, dev_vectors):
        sim1 = sim_funcs[sim](vec, profile1) 
        sim2 = sim_funcs[sim](vec, profile2)
        if sim1 >= sim2:
            correct += 1 if doc.output == 1 else 0
            incorrect += 1 if doc.output == 2 else 0
        else:
            correct += 1 if doc.output == 2 else 0
            incorrect += 1 if doc.output == 1 else 0

    return correct / (correct + incorrect)


def run_experiment(experiment_number, term_weights, distance_metric):
    print(f"\nROW {experiment_number}")
    all(False, False, term_weights, 1, distance_metric)

def experiment():
    term_weights_list = [
        TermWeights(True, False, False, False),
        TermWeights(False, True, False, False),
        TermWeights(False, True, False, False),
        TermWeights(False, True, False, False),
        TermWeights(False, False, True, False),
        TermWeights(False, False, False, True)
    ]
    distance_metrics = ['cosine', 'overlap']

    for i in range(1, 13):
        term_weights = term_weights_list[(i - 1) % len(term_weights_list)]
        distance_metric_index = (i - 1) // len(term_weights_list)
        distance_metric = distance_metrics[distance_metric_index]
        run_experiment(i, term_weights, distance_metric)

def all(stem, removestop, term_weights, tokenization_mode, sim):
    datasets = ['tank', 'plant', 'perplace', 'smsspam']
    term_funcs = {'tf': compute_tf}
    sim_funcs = {'cosine': cosine_sim, 'overlap': overlap_sim}
    
    permutations = [term_funcs, datasets]
    results = {}
    for term, dataset in it.product(*permutations):
        training_docs = process_documents(dataset + '-train.tsv', tokenization_mode)
        dev_docs = process_documents(dataset + '-dev.tsv', tokenization_mode)
        training_docs = stem_stop(training_docs, stem, removestop)
        dev_docs = stem_stop(dev_docs, stem, removestop)

        label_1_train_docs = label_docs(training_docs, 1)
        label_2_train_docs = label_docs(training_docs, 2)
        
        label_1_train_term_freq = compute_doc_freqs(label_1_train_docs)
        label_2_train_term_freq = compute_doc_freqs(label_2_train_docs)
        label_1_train_vectors = [term_funcs[term](doc, label_1_train_term_freq, term_weights) for doc in label_1_train_docs]
        label_2_train_vectors = [term_funcs[term](doc, label_2_train_term_freq, term_weights) for doc in label_2_train_docs]

        dev_term_freq = compute_doc_freqs(dev_docs)
        dev_vectors = [term_funcs[term](doc, dev_term_freq, term_weights) for doc in dev_docs]

        profile1 = compute_centroid(label_1_train_vectors)
        profile2 = compute_centroid(label_2_train_vectors)
        
        accuracy = evaluate_dev_docs(dev_docs, dev_vectors, profile1, profile2, sim_funcs, sim)
        results[dataset] = accuracy

    print(results)
    return

if __name__ == '__main__':
    experiment()
