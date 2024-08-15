
python train_add_type.py sentence-transformers/all-distilroberta-v1 160
python train_add_type.py sentence-transformers/all-mpnet-base-v2 64 cuda:1
python train_nsym_type.py sentence-transformers/all-distilroberta-v1 160 cuda:2
python train_nsym_type.py sentence-transformers/all-mpnet-base-v2 64 cuda:3