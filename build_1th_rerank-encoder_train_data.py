import jsonlines
from tqdm import tqdm
import json
import csv
import os
import nltk
from nltk.corpus import stopwords

from explanation_retrieval.ranker.bm25_v2 import BM25
from explanation_retrieval.ranker.relevance_score import RelevanceScore
from explanation_retrieval.ranker.explanatory_power_v2 import ExplanatoryPower
from explanation_retrieval.ranker.utils import Utils


#load utils
utils = Utils()
utils.init_explanation_bank_lemmatizer()

# Load facts bank
with open("entailmentbank/data/worldtree_corpus_sentences_extended.json", 'r') as f:
    knowledge_corpus = json.load(f)
# print(knowledge_corpus)


# Load train set (explanations corpus)
with open("entailmentbank/data/hypotheses_train.json", 'r') as f:
    hypotheses_train = json.load(f)


######### CHAINS EXTRACTION ###########

# open output files to save the final results
chains_output = open("./entailmentbank/train/rerank_train_1th.csv", "w")

# Parameters
K = len(knowledge_corpus.items())
Negative = 5

# load and fit the sparse model
sparse_model = BM25()
facts_bank_lemmatized = []
explanations_corpus_lemmatized = []
ids = []
q_ids = []
# construct sparse index for the facts bank
for t_id, ts in tqdm(knowledge_corpus.items()):
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

#load relevance using the sparse model
RS = RelevanceScore(sparse_model)


with jsonlines.open('entailmentbank/task_3/train.jsonl', "r") as rfd:
    for item in rfd:
        positive_examples = []
        hypothesis=item['hypothesis']
        wp=item['meta']['worldtree_provenance'] # list
        chains=[]
        for i in wp:
            chains.append(i['uuid'])
        # print(chains)
        neg_list=[]
        for i in range(25):
            gold_knowledge=knowledge_corpus[chains[i]]
            positive_examples.append(gold_knowledge)
            print(hypothesis + "\t" + gold_knowledge + "\t" + str(1), file=chains_output)
            temp = []
            for word in nltk.word_tokenize(gold_knowledge):
                if not word.lower() in stopwords.words("english"):
                    temp.append(utils.explanation_bank_lemmatize(word.lower()))
            lemmatized_gold_knowledge = " ".join(temp)


            
            # retrieve most similar negative facts
            relevance_scores_negative = RS.compute(lemmatized_gold_knowledge, K)
            count = 0
            for fact_negative in sorted(relevance_scores_negative, key=relevance_scores_negative.get, reverse=True):
                if not fact_negative in chains and not fact_negative in neg_list:
                    neg_knowledge=knowledge_corpus[fact_negative]
                    # save negative example
                    print(hypothesis + "\t" + neg_knowledge + "\t" + str(relevance_scores_negative[fact_negative]),
                          file=chains_output)
                    neg_list.append(fact_negative)
                    count += 1
                if count >= Negative:
                    break
        # break
chains_output.close()
