# KSIHER
Code and data for the paper "A General Coarse-to-fine Approach for Knowledge Selection based on Iterative Hybrid Encoding and Re-ranking"

## Setup:

Install the [sentence-transformers](https://www.sbert.net/) package:

`pip install -U sentence-transformers`

Install the [faiss-gpu](https://pypi.org/project/faiss-gpu/) package:

`pip install faiss-gpu`

## Dense Encoder:

To reproduce our experiments, download the model and store it in `./models`.

**Training:**

If you want to train the dense encoder from scratch, you can use the released `train_bi-encoder.py` script. This will create a new [Sentence-BERT](https://www.sbert.net/) model (`bert-base-uncased`) and fine-tune it on the inference chains stored in `./entailmentbank/train/triplets_data.csv` via MultipleNegativesRankingLoss.

If needed, you can regenerate the training-set using the `build_bi-encoder_train_data.py` script.

##  Knowledge Selection based on Iterative Hybrid Encoding Experiment:

Put the trained models into the `.\models` folder, run the following command to start the experiment:

`python ./explanation_regeneration_experiment.py`

This will create the [FAISS](https://faiss.ai/) index and perform multi-hop inference using SCAR

