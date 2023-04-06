import msgpack
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import json
import os
import pickle
import faiss
import numpy as np

from explanation_retrieval.ranker.bm25_v2 import BM25
from explanation_retrieval.ranker.relevance_score import RelevanceScore
from explanation_retrieval.ranker.explanatory_power_v2 import ExplanatoryPower
from explanation_retrieval.ranker.utils import Utils
from sentence_transformers import SentenceTransformer

#load utils
utils = Utils()
utils.init_explanation_bank_lemmatizer()

#Load facts bank
with open("entailmentbank/data/worldtree_corpus_sentences_extended.json", 'r') as f:
    knowledge_train = json.load(f)

#Load train and dev set (explanations corpus)
with open("entailmentbank/data/hypotheses_train.json", 'r') as f:
    hypotheses_train = json.load(f)

with open("entailmentbank/data/chains_train.json", 'r') as f:
    chains_train = json.load(f)

with open("entailmentbank/data/hypotheses_test.json", 'r') as f:
    hypotheses_test = json.load(f)

with open("entailmentbank/data/chains_test.json", 'r') as f:
    chains_test = json.load(f)


#load dense model
dense_model_name = './models/en_bank_nli'
dense_model = SentenceTransformer(dense_model_name)

######### BUILD THE FAISS INDEX ###########

#parameters
max_corpus_size = len(knowledge_train.items())
embedding_cache_path = 'embeddings-size-{}.pkl'.format(max_corpus_size)
# embedding_cache_path='embeddings-size-11941.pkl'
embedding_size = 768    #Size of embeddings
top_k_hits = 1000       #Output k hits
corpus_sentences = []
corpus_ids_original = []

#Defining our FAISS index
#Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N) - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
n_clusters =  282
#We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
quantizer = faiss.IndexFlatIP(embedding_size)
index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
#Number of clusters to explorer at search time. We will search for nearest neighbors in 10 clusters.
index.nprobe = 110

#Check if embedding cache path exists
if not os.path.exists(embedding_cache_path):
    #Check if the dataset exists. If not, extract
    for t_id, ts in tqdm(knowledge_train.items()):

        corpus_sentences.append(ts)
        corpus_ids_original.append(t_id)

    print("Encode the corpus. This might take a while")
    corpus_embeddings = dense_model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

    print("Store file on disc")
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({'ids': corpus_ids_original, 'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
else:
    print("Load pre-computed embeddings from disc")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_ids_original = cache_data['ids']
        corpus_sentences = cache_data['sentences']
        corpus_embeddings = cache_data['embeddings']

### Create the FAISS index
print("Start creating FAISS index")
# First, we need to normalize vectors to unit length
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]
# Then we train the index to find a suitable clustering
index.train(corpus_embeddings)
# Finally we add all embeddings to the index
index.add(corpus_embeddings)

print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))



######### MULTI-HOP EXPLANATION REGENERATION ###########

# open output files to save the final results
pred_q = open("entailmentbank/outputs/prediction_top50.txt", "w")
out_q = open("entailmentbank/outputs/retireval_top50.txt", "w")

# Parameters
K = 1000  # relevance limit
Q = 80 # similar hypotheses limit
QK = 70 # explanatory power limit
weights = [0.89, 0.11] # relevance and explanatory power weigths
#eb_dataset = hypotheses_test # test dataset to adopt for the experiment
# -------------------------------------------------------------
hypotheses_dataset = hypotheses_test # test hypotheses to adopt for the experiment

Iterations = 9 # number of iterations

# load and fit the sparse model
sparse_model = BM25()
facts_bank_lemmatized = []
explanations_corpus_lemmatized = []
ids = []
q_ids = []
# construct sparse index for the facts bank
for t_id, ts in tqdm(knowledge_train.items()):
    temp = []
    # facts  lemmatization
    for word in nltk.word_tokenize(ts):
        temp.append(utils.explanation_bank_lemmatize(word.lower()))
    lemmatized_fact = " ".join(temp)
    facts_bank_lemmatized.append(lemmatized_fact)
    ids.append(t_id)

# construct sparse index for the explanations corpus
for q_id, exp in tqdm(hypotheses_train.items()):
    temp = []
    # question lemmatization
    for word in nltk.word_tokenize(exp):
        temp.append(utils.explanation_bank_lemmatize(word.lower()))
    lemmatized_question = " ".join(temp)
    explanations_corpus_lemmatized.append(lemmatized_question)
    q_ids.append(q_id)
#fit the sparse model
sparse_model.fit(facts_bank_lemmatized, explanations_corpus_lemmatized, ids, q_ids)

#load relevance and explanatory power using the sparse model
RS = RelevanceScore(sparse_model)
PW = ExplanatoryPower(sparse_model, chains_train)

pre_dict={}
# Perform multi-hop inference for explanation regeneration and save the results
for q_id, exp in tqdm(hypotheses_dataset.items()):
    # initialize the partially constructed explanation as an empty list
    partial_explanation = []
    question=exp
    # lemmatization and stopwords removal
    temp = []
    for word in nltk.word_tokenize(exp):
        if not word.lower() in stopwords.words("english"):
          temp.append(utils.explanation_bank_lemmatize(word.lower()))
    lemmatized_question = " ".join(temp)

    # compute the explanatory power given the hypothesis
    explanatory_power = PW.compute(q_id, lemmatized_question, Q, QK)

    print("===========================================", file = out_q)

    # for each iteration
    for step in range(Iterations):
        #print the query
        print("---------------------------------------------", file = out_q)
        print("Query", step, question, file = out_q)
        print("---------------------------------------------", file = out_q)

        # Compute the relevance score using the sparse model
        relevance_scores_sparse = RS.compute(lemmatized_question, K)

        # Compute the relevance score using the dense model
        question_embedding = dense_model.encode(question)
        # FAISS works with inner product (dot product). When we normalize vectors to unit length, inner product is equal to cosine similarity
        question_embedding = question_embedding / np.linalg.norm(question_embedding)
        question_embedding = np.expand_dims(question_embedding, axis=0)
        # Search in FAISS. It returns a matrix with distances and corpus ids.
        distances, corpus_ids = index.search(question_embedding, top_k_hits)
        # We extract corpus ids and scores for the query
        hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        #save the relevance scores computed using the dense model
        relevance_scores_dense = {}
        for hit in hits[0:top_k_hits]:
            relevance_scores_dense[corpus_ids_original[hit['corpus_id']]] = hit['score']

        #compute the explanatory score for each element in the facts bank
        explanatory_scores = {}
        for t_id, ts in knowledge_train.items():
            if not t_id in explanatory_power:
                explanatory_power[t_id] = 0
            if not t_id in relevance_scores_sparse:
                relevance_scores_sparse[t_id] = 0
            if not t_id in relevance_scores_dense:
                relevance_scores_dense[t_id] = 0
            explanatory_scores[t_id] = weights[0] * (relevance_scores_sparse[t_id] + relevance_scores_dense[t_id]) + weights[1] * (explanatory_power[t_id])

        # select the best fact and save the partial explanation
        for fact in sorted(explanatory_scores, key=explanatory_scores.get, reverse=True):
            if not fact in partial_explanation:
                to_write = q_id + "\t" + fact
                print(to_write, file=pred_q)
                if fact in chains_test[q_id]:
                    print(knowledge_train[fact], "***", file = out_q)
                else:
                    print(knowledge_train[fact], file = out_q)
                # update the partial explanation
                partial_explanation.append(fact)
                break

        # update the query concatenating it with the partially constructed explanation
        question = hypotheses_dataset[q_id]
        for fact in partial_explanation:
            question += ". " +knowledge_train[fact]
        # lemmatization and stopwords removal
        temp = []
        for word in nltk.word_tokenize(question):
            if not word.lower() in stopwords.words("english"):
                temp.append(utils.explanation_bank_lemmatize(word.lower()))
        lemmatized_question = " ".join(temp)
    pre_list = partial_explanation
    # rank the remaining sentences in the facts bank
    print_count = 0
    for fact in sorted(explanatory_scores, key=explanatory_scores.get, reverse=True):
        if not fact in partial_explanation:
            to_write = q_id + "\t" + fact
            print(to_write, file=pred_q)
            if print_count < 41:
                if fact in chains_test[q_id]:
                    pre_list.append(fact)
                    print(knowledge_train[fact], "***", file = out_q)
                else:
                    pre_list.append(fact)
                    print(knowledge_train[fact], file = out_q)
            print_count += 1

    pre_dict[q_id]=pre_list

with open("entailmentbank/outputs/pre_test_top50.json",'w') as f:
    json.dump(pre_dict, f)

pred_q.close()
out_q.close()

