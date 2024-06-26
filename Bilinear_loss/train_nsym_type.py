"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
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

TEST = False

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

# Check if dataset exists. If not, download and extract  it
nli_dataset_path = "data/AllNLI.tsv.gz"
if not os.path.exists(nli_dataset_path):
    util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

#sts_dataset_path = "data/stsbenchmark.tsv.gz"
#if not os.path.exists(sts_dataset_path):
#    util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)


# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Read the dataset
train_batch_size = 32


model_save_path = (
    "output/training_mul_nli_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

checkpoint_save_path = (
    "output/training_mul_nli_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/checkpoint"
)

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model],
    device = device,
    )


# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read AllNLI train dataset")

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = []
count = 0

with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        if row["split"] == "train":
            label_id = label2int[row["label"]]
            train_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=label_id))
            count += 1
            if count > 500 and TEST:
                break

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = BilinearLoss(
    model=model, 
    num_labels=len(label2int),
    sentence_model_name = model_name,
    sim_method = "Nsym",
    device = device,
)


label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
dev_samples = []
with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        if row["split"] == "dev":
            label_id = label2int[row["label"]]
            dev_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=label_id))

dev_evaluator = BilinearEvaluator.from_input_examples(
    dev_samples, 
    batch_size=train_batch_size, 
    name="nsym", 
    similarity=train_loss
)


# Configure the training
num_epochs = 10

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=num_epochs,
    evaluation_steps=2000,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    #checkpoint_path=checkpoint_save_path,
    #checkpoint_save_steps= 2000,
)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

"""
test_samples = []
with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        if row["split"] == "test":
            score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
            test_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=score))

model = SentenceTransformer(model_save_path)
test_evaluator = BilinearEvaluator.from_input_examples(
    test_samples, batch_size=train_batch_size, name="sts-test"
)
test_evaluator(model, output_path=model_save_path)
"""
