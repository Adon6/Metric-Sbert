import torch
from torch import nn, Tensor
from typing import Iterable, Dict, Callable, Optional
from sentence_transformers import SentenceTransformer
import logging


logger = logging.getLogger(__name__)


class BilinearLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        sentence_embedding_dimension: int,
        num_labels: int,
        loss_fct: Callable = nn.CrossEntropyLoss(),
        normalized : bool= False,
        ADD : bool = True,
        device : Optional[str] = None,
    ):
        """
        This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
        model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

        """
        super(BilinearLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.normalized = normalized
        self.model_name = ("ADD" if ADD else "MUL") + ("_N" if normalized else "")
        self.add = ADD
        self.embedding_dim = sentence_embedding_dimension
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.Us = nn.ParameterList([nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim).to(self.device)) for _ in range(num_labels)]).to(self.device)
        logger.info("Bilinear loss: #Labels: {}".format(num_labels))
        
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        e1, e2 = reps
        e1, e2 = Tensor(e1).to(self.device), Tensor(e2).to(self.device)
        self.Us = self.Us.to(self.device)
        
        if self.normalized:
            e1 = nn.functional.normalize(e1, p=2)# example normalization
            e2 = nn.functional.normalize(e2, p=2)# example normalization

        if self.add:
            qs = [torch.sum(e1 * (e2 @ ((U + U.t())/2) ), dim=1) for U in self.Us] # W = W.T
        else:
            qs = [torch.sum(e1 * (e2 @ (U @ U.t()) ), dim=1) for U in self.Us] # W = W.T
        #output = qs
        output = torch.stack(qs, dim= 1)

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output

    def sim(self, e1, e2):
        eval_device = 'cpu'
        e1, e2 = Tensor(e1).to(eval_device), Tensor(e2).to(eval_device)
        self.Us = self.Us.to(eval_device)
        if self.normalized:
            e1 = nn.functional.normalize(e1, p=2)# example normalization
            e2 = nn.functional.normalize(e2, p=2)# example normalization

        if self.add:
            qs = [torch.sum(e1 * (e2 @ (U + U.t()) ), dim=1) for U in self.Us] # W = W.T
        else:
            qs = [torch.sum(e1 * (e2 @ (U @ U.t()) ), dim=1) for U in self.Us] # W = W.T
        
        output = torch.stack(qs, dim= 1)

        return output
