import json
# from sklearn.metrics import f1_score
from  tqdm import tqdm

with open("entailmentbank/data/chains_test.json", 'r') as f:
    true = json.load(f)

with open("entailmentbank/outputs/pre_rerank_25.json", 'r') as f:
    pre = json.load(f)

def f1_score(t, p):
    tp = 0
    for ture in p:
        if ture in t:
            tp = tp + 1
    fn=0
    for ture in t:
        if ture not in p:
            fn = fn + 1
    # fp = 25 - tp
    # tn = 25 - tp - fn - fp
    # f1 = 2 * tp / (25 + tp - tn)
    Precision=tp/25.0
    Recall=tp/25.0
    f1=2*Precision*Recall/(Precision+Recall)
    return f1

score=0.0
for i,j in tqdm(true.items()):
    T=true[i]
    P=pre[i]
    f1=f1_score(T,P)
    score += f1
    

s=score/len(true)
print(s)  