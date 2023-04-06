import json
import csv

with open('entailmentbank/data/chains_train.json','r') as f:
    true_train=json.load(f)

with open('entailmentbank/outputs/pre_train_top50.json','r') as f:
    pre_train=json.load(f)
    
with open("entailmentbank/data/worldtree_corpus_sentences_extended.json", 'r') as f:
    knowledge_corpus = json.load(f)

with open("entailmentbank/data/hypotheses_train.json", 'r') as f:
    hypotheses_train = json.load(f)

output=open('entailmentbank/train/cross_train_2th.csv','w')  

for i,j in pre_train.items():
    true_k=true_train[i] #true list k
    for k in j:
        if k in true_k:
            
            print(hypotheses_train[i] + "\t" + knowledge_corpus[k] + "\t" + str(1), file=output)
        else:
            print(hypotheses_train[i] + "\t" + knowledge_corpus[k] + "\t" + str(0), file=output)

    # print(i,j)
    # break