from datasets import load_dataset

ds = load_dataset("copenlu/spanex", "snli")
print(len(ds))
input()
print(ds)