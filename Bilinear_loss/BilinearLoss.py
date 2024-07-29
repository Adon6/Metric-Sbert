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
        num_labels: int = 3,
        loss_fct: Callable = nn.CrossEntropyLoss(),
        sentence_model_name: str = None,
        normalization : str= "",
        sim_method : str = "ADD",
        device :Optional[str] = None,
    ):
        """
        This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
        model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

        """
        super(BilinearLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.normalization = normalization
        self.model_name = sim_method + "_" + normalization + "_"
        self.sentence_model_name = sentence_model_name
        self.sim_method = sim_method
        self.embedding_dim = model.get_sentence_embedding_dimension()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.Us = nn.ParameterList([nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim).to(self.device)) for _ in range(num_labels)])
        logger.info("Bilinear loss: #Labels: {}".format(num_labels))
        
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor = None):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        e1, e2 = reps
        e1, e2 = Tensor(e1).to(self.device), Tensor(e2).to(self.device)
        e1, e2 = self.get_norm_emb(e1, e2)
    
        Ms = self.get_sim_mat(device = self.device)
        output = torch.stack([torch.sum(e1 * (e2 @ M) , dim=1) for M in Ms], dim= 1 )

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output

    def sim(self, e1, e2):
        eval_device = 'cpu'
        e1, e2 = Tensor(e1).to(eval_device), Tensor(e2).to(eval_device)
        e1, e2 = self.get_norm_emb(e1, e2)

        #self.Us = self.Us.to(eval_device)
        Ms = self.get_sim_mat(device = eval_device)

        output = torch.stack([torch.sum(e1 * (e2 @ M) , dim=1) for M in Ms], dim= 1 )

        return output
    
    def get_norm_emb(self, e1 ,e2):
        if self.normalization:
            e1 = nn.functional.normalize(e1, p=2)# example normalization
            e2 = nn.functional.normalize(e2, p=2)# example normalization
        return e1, e2

    def get_sim_mat(self, device = None):
        if not device:
            device = self.device
             
        if self.sim_method == "ADD":
            Ms = torch.stack( [(U + U.t())/2 for U in self.Us] )# W = W.T
        elif self.sim_method == "MUL":
            Ms = torch.stack( [ U @ U.t() for U in self.Us] )# W = W.T
        else:
            Ms = torch.stack( [ U for U in self.Us] )# W = W.T

        return Ms.to(device)
 

    def save(self, path):
       torch.save({
            'model_state_dict': self.state_dict(),
            'metadata': {
                'sentence_model_name' : self.sentence_model_name,
                'sim_method': self.sim_method,
                'normalization': self.normalization,
                'num_labels': self.num_labels,
                'embedding_dim': self.embedding_dim,
                'loss_fct' : self.loss_fct,
            },
        }, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        metadata = checkpoint.pop("metadata")
        smodel = SentenceTransformer.load(metadata['sentence_model_name'])
        
        modelbili = cls(
            model = smodel,
            num_labels = metadata['num_labels'],
            loss_fct = metadata['loss_fct'],
            sentence_model_name = metadata['sentence_model_name'],
            normalization = metadata['normalization'],
            sim_method = metadata['sim_method'],
        )
        
        modelbili.load_state_dict(checkpoint['model_state_dict'])
        
        return modelbili