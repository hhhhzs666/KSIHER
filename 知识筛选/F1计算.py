import json
# from sklearn.metrics import f1_score
from  tqdm import tqdm

with open("data/语料库/chains_test.json", 'r') as f:
    true = json.load(f)

with open("data/语料库/pre_test.json", 'r') as f:
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
print(s)  # 0.49005702280912417  0.49347108843537424  0.49607142857142866  0.4973630452180872  0.49103901560624236
          # 5877  5920  5955  5946  5727
          # 0.5079680872348944  0.5150628251300517  0.5283428371348542
          # 0.43752921168467374  0.5350535214085637  0.5351833733493399
          # 0.5382491996798721  0.5408447378951582  5+95 roberta+tanda 0.5416657663065229
          # 0.5384942977190877 0.5391976790716292  0.5528717486994799  0.5529459783913564  1+39 0.5532220888355341
          # 0.5465114045618247 0.5469553821528613 4+36 0.5507323929571823
          # 0.5514259703881551  0.5514308723489394
          # 9+41 0.5522446978791513  12+38 0.5525208083233291  13+37 0.5532217887154858  20+30 0.5534520808323322
          # 20+30 0.5639399759903966  0.5658027210884352