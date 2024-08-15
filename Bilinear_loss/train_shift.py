"""
Usage:
python train_shift.py

OR
python train_shift.py pretrained_transformer_model_name batch_size
"""
def main():
    import torch
    from torch.utils.data import DataLoader
    import math
    from sentence_transformers import models, losses
    from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
    import logging
    from datetime import datetime
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from BilinearLoss import BilinearLoss
    from BilinearEvaluator import BilinearEvaluator

    from xsbert.models import ShiftingReferenceTransformer, XSTransformer
    from xsbert.utils import load_nil_data

    TEST = False

    #### Just some code to print debug information to stdout
    logging.basicConfig(
        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
    )
    #### /print debug information to stdout

    # Check if dataset exists. If not, download and extract it
    nli_dataset_path = "data/AllNLI.tsv.gz"
    if not os.path.exists(nli_dataset_path):
        util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

    # You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
    train_batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    device = sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu")

    model_save_path = (
        "output/ef+_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"_matrix.pth"
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
        sim_method = "ADD",
        device = device,
    )

    train_loss.save(model_save_path)


    # training config

    #model_path = "input/training_add2_nli_sentence-transformers-all-mpnet-base-v2-2024-06-13_18-43-38/eval/epoch4_step-1_sim_evaluation_add_matrix.pth"
    model_load_path = model_save_path
    num_epochs = 5


    #model_save_path = '../xs_models/droberta_bilinear'
    model_save_path = (
        "output/f+" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # model
    bilinear_loss = BilinearLoss.load(model_load_path)
    bilinear_loss.device = device

    transformer_layer = bilinear_loss.model[0]
    save_path =  'transformer_layertemp'
    transformer_layer.save(save_path)
    embedding_model = ShiftingReferenceTransformer(save_path)

    pooling_layer = bilinear_loss.model[1]

    #model = XSRoberta(modules=[transformer, pooling], sim_measure= "bilinear", sim_mat= bilinear_loss.get_sim_mat())
    xsmodel = XSTransformer(
        modules=[embedding_model, pooling_layer],
        device=device,
        sim_mat= bilinear_loss.get_sim_mat(),
        sim_measure= "bilinear",
        )

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
    xsmodel.fit(train_objectives=[(train_dataloader, bilinear_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=4000,
            warmup_steps=math.ceil(len(train_dataloader) * num_epochs  * 0.1),
            output_path=model_save_path,
            )

    # loading model checkpoint and running evaluation
    xsmodel2 = XSTransformer(model_save_path,    
        device=device,
        sim_mat= bilinear_loss.get_sim_mat(),
        sim_measure= "bilinear",
        )
    test_evaluator = BilinearEvaluator.from_input_examples(test_samples, name='nil-test-shift')
    test_evaluator(xsmodel2, output_path=model_save_path)


if __name__ == "__main__":
    main()