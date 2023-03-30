import json
import csv

with open('data/语料库/chains_train.json','r') as f:
    true_train=json.load(f)

with open('data/语料库/pre_train.json','r') as f:
    pre_train=json.load(f)
    
with open("data/语料库/worldtree_corpus_sentences_extended.json", 'r') as f:
    knowledge_corpus = json.load(f)

with open("data/语料库/hypotheses_train.json", 'r') as f:
    hypotheses_train = json.load(f)

output=open('data/语料库/hard_negative_100.csv','w')  

for i,j in pre_train.items():
    true_k=true_train[i] #true list k
    for k in j:
        if k in true_k:
            
            print(hypotheses_train[i] + "\t" + knowledge_corpus[k] + "\t" + str(1), file=output)
        else:
            print(hypotheses_train[i] + "\t" + knowledge_corpus[k] + "\t" + str(0), file=output)

    # print(i,j)
    # break