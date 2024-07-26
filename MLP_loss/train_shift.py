from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, models, util, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import os
import sys
from datetime import datetime

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from xsbert.models import ShiftingReferenceTransformer, XSTransformer
from xsbert.utils import load_nil_data

from MLPLoss import MLPLoss
from MLPEvaluator import MLPEvaluator


# training config
# model_name = 'sentence-transformers/all-mpnet-base-v2'
model_name = 'sentence-transformers/all-distilroberta-v1'
model_save_path = (
    "output/finetune_mlp_nli_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

#model_path = "input/training_add2_nli_sentence-transformers-all-mpnet-base-v2-2024-06-13_18-43-38/eval/epoch4_step-1_sim_evaluation_add_matrix.pth"
model_path = "data/training_mlp_nli_sentence-transformers-all-distilroberta-v1-2024-07-17_08-55-08/eval/epoch9_step-1_sim_evaluation_mlp_MLP_matrix.pth"
train_batch_size = 160
num_epochs = 5


if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

sentence_transformer_model = SentenceTransformer(model_name)
# model
mlp_loss = MLPLoss.load(model_path)

transformer_layer = mlp_loss.model[0]
save_path =  'transformer_temp'
transformer_layer.save(save_path)
embedding_model = ShiftingReferenceTransformer(save_path)

pooling_layer = mlp_loss.model[1]

model = XSTransformer(
    modules=[embedding_model, pooling_layer],
    device='cuda',
    sim_measure= "cos",
    )

# data
nli_dataset_path = "data/AllNLI.tsv.gz"
if not os.path.exists(nli_dataset_path):
    util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

# dataloading
train_samples, dev_samples, test_samples = load_nil_data(nli_dataset_path)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
evaluator = MLPEvaluator.from_input_examples(
    dev_samples, 
    batch_size=train_batch_size, 
    name="mlp-shift", 
    similarity=mlp_loss
)


# training
# If you want to train a model with a dot-product instead of cosine as a similarity-measure
# use the loss below instead.
# loss = DotSimilarityLoss(model=model)
model.fit(train_objectives=[(train_dataloader, mlp_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=4000,
          warmup_steps=math.ceil(len(train_dataloader) * num_epochs  * 0.1),
          output_path=model_save_path,
          )

# loading model checkpoint and running evaluation
model = XSTransformer(model_save_path,    
    device='cuda',
    sim_measure= "cos",
    )
test_evaluator = MLPEvaluator.from_input_examples(test_samples, name='mlp-shift')
test_evaluator(model, output_path=model_save_path)