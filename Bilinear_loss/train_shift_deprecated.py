from torch.utils.data import DataLoader
import math
from datetime import datetime
from sentence_transformers import SentenceTransformer, models, util, losses, InputExample
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BilinearLoss import BilinearLoss
from BilinearEvaluator import BilinearEvaluator

from xsbert.models import ShiftingReferenceTransformer, ReferenceTransformer, XSTransformer, DotSimilarityLoss
from xsbert.utils import load_nil_data


# training config
# model_name = 'sentence-transformers/all-mpnet-base-v2'
model_name = 'sentence-transformers/all-distilroberta-v1'
#model_save_path = '../xs_models/droberta_bilinear'
model_save_path = (
    "output/fo" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

#model_path = "input/training_add2_nli_sentence-transformers-all-mpnet-base-v2-2024-06-13_18-43-38/eval/epoch4_step-1_sim_evaluation_add_matrix.pth"
model_path = "input/training_nsym_nli_sentence-transformers-all-distilroberta-v1-2024-07-15_11-37-39osD/eval/epoch9_step-1_sim_evaluation_nsym_matrix.pth"
train_batch_size = 160
num_epochs = 5


if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# model
bilinear_loss = BilinearLoss.load(model_path)

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
    name="nsym-shift", 
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
model = XSTransformer(model_save_path,    
    device='cuda',
    sim_mat= bilinear_loss.get_sim_mat(),
    sim_measure= "bilinear",
    )
test_evaluator = BilinearEvaluator.from_input_examples(test_samples, name='nil-test-shift')
test_evaluator(model, output_path=model_save_path)