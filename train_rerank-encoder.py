from torch.utils.data import DataLoader

from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator,CESoftmaxAccuracyEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import sys
import os
import csv
import torch



#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout




train_path='entailmentbank/train/rerank_train_1th.csv'   # Second training is rerank_train_2th.csv
dev_path='entailmentbank/train/rerank_dev_1th.csv'


#Define our Cross-Encoder
train_batch_size = 200
num_epochs = 12 # 2th train is 4
model_save_path = 'models/tanda_reranker_1th'  # Second training is tanda_reranker_2th

# The model we used can be downloaded here :
# https://d3t7erp6ge410c.cloudfront.net/tanda-aaai-2020/models/tanda_roberta_base_asnq.tar
model = CrossEncoder('models/tanda_roberta_base_asnq', num_labels=2,max_length=128)  # Second training is tanda_reranker_1th

# model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5])


# Read STSb dataset
logger.info("Read train dataset")

train_samples = []
with open(train_path, 'rt', encoding='utf8') as fIn:
    # reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    reader=csv.reader(fIn,delimiter='\t')
    for row in reader:
        # print(row)
        score = float(row[2])   # Normalize score to range 0 ... 1
        train_samples.append(InputExample(texts=[row[0], row[1]], label=score))
        

logger.info("Read dev dataset")
dev_samples = []

with open(dev_path, 'rt', encoding='utf8') as fIn:
    # reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    reader=csv.reader(fIn,delimiter='\t')
    for row in reader:
        # print(row)
        score = float(row[2])   # Normalize score to range 0 ... 1
        dev_samples.append(InputExample(texts=[row[0], row[1]], label=score))





# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


# We add an evaluator, which evaluates the performance during training
evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_samples, name='sts-dev')
# evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          save_best_model=True)
