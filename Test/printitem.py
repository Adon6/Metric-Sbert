from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
import os
import gzip
import csv


# Check if dataset exists. If not, download and extract  it
nli_dataset_path = "data/AllNLI.tsv.gz"
if not os.path.exists(nli_dataset_path):
    util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

count = 0

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
test_samples = []
with gzip.open(nli_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        if row["split"] == "test":
            label_id = label2int[row["label"]]
            test_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=label_id))
            if count <1000:
                print(row)
            count += 1

print(len(test_samples))
