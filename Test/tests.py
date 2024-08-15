import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
import logging
from datetime import datetime
import sys
import os
from datasets import load_dataset  # Import the datasets library

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Bilinear_loss.BilinearLoss import BilinearLoss
from Bilinear_loss.BilinearEvaluator import BilinearEvaluator
from utils import load_nil_data


# Setting up logging
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = 'bert-base-uncased'
model_path = "input/training_mul_nli_bert-base-uncased-2024-06-03_11-05-25_L0-9/eval/epoch9_step-1_sim_evaluation_mul_matrix.pth"
model_save_path = (
    "test/t__" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Load the pre-trained model
test_model = BilinearLoss.load(model_path)

test_batchsize = 64

# Load the copenlu/spanex dataset
logging.info("Loading copenlu/spanex dataset")
ds = load_dataset("copenlu/spanex", "snli")

# Convert the dataset into InputExample format
label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}

# Processing test samples
test_samples = [
    InputExample(
        texts=[example['premise'], example['hypothesis']],
        label=label2int[example['label']]
    )
    for example in ds['test']  # Assuming you want to use the test split
]

# Create the evaluator with the test samples
test_evaluator = BilinearEvaluator.from_input_examples(
    test_samples, 
    batch_size=test_batchsize,
    name="test", 
    similarity=test_model
)

# Evaluate the model
test_evaluator(test_model.model, output_path=model_save_path, steps=10)
