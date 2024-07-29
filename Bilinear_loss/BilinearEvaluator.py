from contextlib import nullcontext
import torch
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import SentenceEvaluator
import logging
import os
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List


logger = logging.getLogger(__name__)


class BilinearEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        outputlabel,
        similarity ,
        batch_size: int = 16,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.outputlabel = outputlabel
        self.write_csv = write_csv

        assert len(self.sentences1) == len(self.sentences2)

        self.similarity = similarity
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = (
            "sim_evaluation"
            + ("_" + name if name else "")
            + "_results.csv"
        )

        self.csv_headers = [
            "epoch",
            "steps",
            "f1",
            "accuracy",
            "precision",
            "recall",
        ]

        self.evaluator_file = (
            "sim_evaluation"
            + ("_" + name if name else "")
            + "_matrix.pth"
        )
    @classmethod
    def from_input_examples(cls, examples: List[InputExample], similarity, **kwargs):
        sentences1 = []
        sentences2 = []
        labels = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            labels.append(example.label)
        return cls(sentences1, sentences2, labels, similarity, **kwargs)

    def __call__(self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(f"BilinearEvaluator: Evaluating the model on the {self.name} dataset {out_txt}:")
        logger.info(f"output path: {output_path}")

        with nullcontext():
            embeddings1 = model.encode(
                self.sentences1,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
            )
            embeddings2 = model.encode(
                self.sentences2,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
            )
        
        similarities = self.similarity.sim(embeddings1,embeddings2)

        predicted_label = torch.argmax(similarities, dim=1)

        accuracy = accuracy_score(self.outputlabel, predicted_label)
        precision = precision_score(self.outputlabel, predicted_label, average='macro')
        recall = recall_score(self.outputlabel, predicted_label, average='macro')
        f1 = f1_score(self.outputlabel, predicted_label, average='macro')

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                        f1,
                        accuracy,
                        precision,
                        recall,
                    ]
                )

            # Save matrix
            if steps < 0 and epoch in [4,9]:
                evaluator_path = os.path.join(
                    output_path, 
                    "epoch"+str(epoch) +"_step" +str(steps) +"_" + self.evaluator_file
                )
                self.similarity.save(evaluator_path)
            
        return accuracy
