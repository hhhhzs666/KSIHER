import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from torch import nn
import numpy as np

cross_encoder = CrossEncoder('models/tanda_reranker_2th')

with open('entailmentbank/outputs/pre_test_top50.json','r') as f:
    pre_test=json.load(f)

with open('./entailmentbank/data/worldtree_corpus_sentences_extended.json','r') as f:
    corpus=json.load(f)

with open('./entailmentbank/data/hypotheses_test.json','r') as f:
    hypotheses_test=json.load(f)
dic={}
for query_id,knowledge_id in pre_test.items():
    cross_inp = [[hypotheses_test[query_id], corpus[k]] for k in knowledge_id]
    # cross_scores = cross_encoder.predict(cross_inp)
    cross_scores = cross_encoder.predict(cross_inp,apply_softmax=True)
    s=np.array(cross_scores)
    cross_scores=s[:,1]

    l=len(cross_scores)
    w=np.arange(1.0,1.0-l*0.02,-0.02)
    cross_scores=w*cross_scores

    # print(cross_scores)
    hits = sorted(zip(knowledge_id,cross_scores), key=lambda x: x[1], reverse=True)
    top_25=[]
    for hit in hits[0:25]:
        top_25.append(hit[0])
    dic[query_id]=top_25
    
    # break


with open("entailmentbank/outputs/pre_rerank_25.json",'w') as f:
    json.dump(dic, f)