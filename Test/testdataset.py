import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

from BilinearLoss import BilinearLoss
from BilinearEvaluator import BilinearEvaluator

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

model_name ='bert-base-uncased'
model_path = 'input/training_add2_nli_bert-base-uncased-2024-06-04_18-01-47_L0-9/eval/epoch9_step-1_sim_evaluation_add_matrix.pth'

sentence_transformer_model = SentenceTransformer(model_name)
# model
test_model = BilinearLoss.load(model_path, sentence_transformer_model)

# Check if dataset exists. If not, download and extract  it
nli_dataset_path = "data/AllNLI.tsv.gz"
if not os.path.exists(nli_dataset_path):
    util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)


label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
test_samples = []
with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        if row["split"] == "test":
            label_id = label2int[row["label"]]
            test_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=label_id))


print(test_samples)
device = "cuda" if torch.cuda.is_available() else "cpu"

test_evaluator = BilinearEvaluator.from_input_examples(
    test_samples, 
    name="test", 
    similarity=test_model
)

