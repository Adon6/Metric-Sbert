import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import util, SentenceTransformer, models, InputExample
from SoftcosLoss import SoftcosLoss
from SoftcosEvaluator import SoftcosEvaluator
import os
import gzip
import csv
import datetime
import logging


TEST = True

# dataset allnli
class NLIDataset(Dataset):
    def __init__(self, datasplit = "train", device = "cuda" ):
        self.device = device 
        self.datasplit = datasplit 
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check if dataset exists. If not, download and extract it
        nli_dataset_path = "data/AllNLI.tsv.gz"
        if not os.path.exists(nli_dataset_path):
            util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)
        
        count = 0
        label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
        train_samples = []
        with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
            reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if row["split"] == self.datasplit:
                    train_samples.append((row["sentence1"], row["sentence2"],label2int[row["label"]]))
                    count+=1
                    if count > 50 and TEST:
                        break
        self.dataset = train_samples

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        premise, hypothesis, label  = self.dataset[idx]
        return InputExample(texts=[premise, hypothesis], label=label)
        #return premise, hypothesis, label

device = "cuda" if torch.cuda.is_available() else "cpu"

word_embedding_model = models.Transformer("bert-base-uncased", max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model],
    device = device,
    )

logging.info("Read AllNLI train dataset")

nli_dataset_path = "data/AllNLI.tsv.gz"
if not os.path.exists(nli_dataset_path):
    util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = []
with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        if row["split"] == "train":
            label_id = label2int[row["label"]]
            train_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=label_id))


train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)
#data = NLIDataset("train", device)
#dataloader = DataLoader(data, batch_size=32, shuffle=True)
logging.info("AllNLI train data loaded.")


Softcos_addtype = True
normalized = False
N_emb = model.get_sentence_embedding_dimension()
train_loss = SoftcosLoss(model, sentence_embedding_dimension = N_emb, num_labels= 3, normalized= normalized, ADD= Softcos_addtype, device= device)

logging.info("read AllNLI dev data.")

testdata = NLIDataset("dev", device)
testdataloader = DataLoader(testdata, batch_size=1, shuffle=True)
evaluator = SoftcosEvaluator.from_input_examples(testdataloader, train_loss, name = "Add" if Softcos_addtype else "Mul")

# Tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)], 
    epochs=2, 
    warmup_steps=10,
    checkpoint_path = "SModel",
    evaluator=evaluator,
    evaluation_steps=10,
    )