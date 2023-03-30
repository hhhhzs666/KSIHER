import json
# from sklearn.metrics import f1_score
from  tqdm import tqdm
import os
import csv

with open("data/语料库/chains_test.json", 'r') as f:
    true = json.load(f)

def f1_score(t, p):
    tp = 0
    for ture in p:
        if ture in t:
            tp = tp + 1
    if tp==0:
        return 0,0,0
    Precision=tp/25.0
    Recall=tp/25.0
    f1=2*Precision*Recall/(Precision+Recall)
    return f1,Precision,Recall


dir='data/语料库'
path_list=[]
for (dirpath, dirnames, filenames) in os.walk(dir):
    print(dirpath)
    for filename in sorted(
        filter(lambda f: "pre_rerank_25_hard_weight" in f, filenames),
                    key=lambda x: int(x.split(".")[-2].split("_")[-1]),
            ):
        filepath = os.path.join(dirpath, filename)
        path_list.append(filepath)

header = ['f1', 'Precision', 'Recall']

with open('25_rerank+hard+weight.csv', 'w', encoding='utf-8') as file_obj:
    writer = csv.writer(file_obj)
    writer.writerow(header)

    for path in path_list:

        with open(path, 'r') as f:
            pre = json.load(f)


        f1_s=0.0
        Pre_s=0.0
        Rec_s=0.0
        for i,j in tqdm(true.items()):
            T=true[i]
            P=pre[i]
            f1,Precision,Recall=f1_score(T,P)
            if f1==0:
                print(path,i)
            f1_s += f1
            Pre_s += Precision
            Rec_s += Recall
            

        s=f1_s/len(true)
        p_s=Pre_s/len(true)
        r_s=Rec_s/len(true)
        writer.writerow([s,p_s,r_s])
    