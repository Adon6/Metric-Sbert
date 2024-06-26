from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, models, util, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import os
from BilinearLoss import BilinearLoss
from BilinearEvaluator import BilinearEvaluator

from xsbert.models import ShiftingReferenceTransformer, ReferenceTransformer, XSTransformer, DotSimilarityLoss
from xsbert.utils import load_nil_data
from datetime import datetime


# training config
# model_name = 'sentence-transformers/all-mpnet-base-v2'
model_name ='roberta-base'
#model_save_path = '../xs_models/droberta_bilinear'
model_save_path = (
    "output/finetune_add_nli_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
model_path = 'data\\training_add2_nli_roberta-base-2024-06-05_14-35-49_L0-9\eval\epoch9_step-1_sim_evaluation_add_matrix.pth'
train_batch_size = 16 
num_epochs = 5

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

sentence_transformer_model = SentenceTransformer(model_name)
# model
bilinear_loss = BilinearLoss.load(model_path, sentence_transformer_model)

transformer_layer = bilinear_loss.model[0]
save_path =  'transformer_layerxx'
transformer_layer.save(save_path)
embedding_model = ShiftingReferenceTransformer(save_path)

pooling_layer = bilinear_loss.model[1]

#model = XSRoberta(modules=[transformer, pooling], sim_measure= "bilinear", sim_mat= bilinear_loss.get_sim_mat())
model = XSTransformer(
    modules=[embedding_model, pooling_layer],
    device='cuda',
    sim_mat= bilinear_loss.get_sim_mat(),
    sim_measure= "bilinear",
    )

# data
nli_dataset_path = "data/AllNLI.tsv.gz"
if not os.path.exists(nli_dataset_path):
    util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

# dataloading
train_samples, dev_samples, test_samples = load_nil_data(nli_dataset_path)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
evaluator = BilinearEvaluator.from_input_examples(
    dev_samples, 
    batch_size=train_batch_size, 
    name="add-shift", 
    similarity=bilinear_loss
)


# training
# If you want to train a model with a dot-product instead of cosine as a similarity-measure
# use the loss below instead.
# loss = DotSimilarityLoss(model=model)
model.fit(train_objectives=[(train_dataloader, bilinear_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=4000,
          warmup_steps=math.ceil(len(train_dataloader) * num_epochs  * 0.1),
          output_path=model_save_path,
          )

# loading model checkpoint and running evaluation
model = XSTransformer(model_save_path)
test_evaluator = BilinearEvaluator.from_input_examples(test_samples, name='nil-test-shift')
test_evaluator(model, output_path=model_save_path)