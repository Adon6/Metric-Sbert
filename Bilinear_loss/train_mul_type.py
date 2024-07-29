"""
Usage:
python train_mul_type.py

OR
python train_mul_type pretrained_transformer_model_name batch_size
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
from BilinearLoss import BilinearLoss
from BilinearEvaluator import BilinearEvaluator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import  load_nil_data
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

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
train_batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
device = "cuda" if torch.cuda.is_available() else "cpu"


model_save_path = (
    "output/x_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

checkpoint_save_path = (
    "output/x_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/checkpoint"
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
train_samples, dev_samples, test_samples = load_nil_data(nli_dataset_path)
if TEST:
    train_samples = train_samples[:1000] 

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = BilinearLoss(
    model=model, 
    num_labels=len(label2int),
    sentence_model_name = model_name,
    sim_method = "MUL",
    device = device,
)

dev_evaluator = BilinearEvaluator.from_input_examples(
    dev_samples, 
    batch_size=train_batch_size, 
    name="x", 
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
# Load the stored model and evaluate
#
##############################################################################

test_model = BilinearLoss.load(model_save_path)

test_evaluator = BilinearEvaluator.from_input_examples(
    test_samples, 
    name="test", 
    similarity=test_model
)