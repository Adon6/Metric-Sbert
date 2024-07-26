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

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Bilinear_loss_old.BilinearLoss import BilinearLoss
from Bilinear_loss_old.BilinearEvaluator import BilinearEvaluator


#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name ='bert-base-uncased'
model_path = "input/training_add2_nli_bert-base-uncased-2024-06-04_18-01-47_L0-9+B/eval/epoch9_step-1_sim_evaluation_add_matrix.pth"

model_save_path = (
    "test/test_mul_nli_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

sentence_transformer_model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model],
    device = device,
    )

# model
test_model = BilinearLoss.load(model_path, sentence_transformer_model)

test_batchsize = 64

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


print(len(test_samples))

device = "cuda" if torch.cuda.is_available() else "cpu"

test_evaluator = BilinearEvaluator.from_input_examples(
    test_samples, 
    batch_size = test_batchsize,
    name="test", 
    similarity=test_model
)

test_evaluator(sentence_transformer_model, output_path=model_save_path, steps = 10)