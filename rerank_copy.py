import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from torch import nn
import numpy as np

import os

with open('./data/语料库/worldtree_corpus_sentences_extended.json','r') as f:
    corpus=json.load(f)

with open('./data/语料库/hypotheses_test.json','r') as f:
    hypotheses_test=json.load(f)

dir='data/语料库'
path_list=[]
for (dirpath, dirnames, filenames) in os.walk(dir):
    print(dirpath)
    for filename in sorted(
        filter(lambda f: "pre_test-50nli-" in f, filenames),
                    key=lambda x: int(x.split(".")[-2].split("-")[-1]),
            ):
        filepath = os.path.join(dirpath, filename)
        path_list.append(filepath)

cross_encoder = CrossEncoder('output/tanda+hard_negative_v2-2023-01-21_21-10-53')
# cross_encoder = CrossEncoder('output/training_cross-2023-01-11_23-23-07')


for i in range(len(path_list)):
    path=path_list[i]
    with open(path,'r') as f:
        pre_test=json.load(f)

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


    with open("data/语料库/pre_rerank_25_hard_weight_{}.json".format(i+1),'w') as f:
        json.dump(dic, f)