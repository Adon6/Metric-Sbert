import torch
from torch import nn, Tensor
from typing import Iterable, Dict, Callable
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
        device :str | None = None,
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
            qs = [torch.sum(e1 * (e2 @ (U + U.t()) ), dim=1) for U in self.Us] # W = W.T
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
            qs = [torch.sum(e1 * (e2 @ ((U + U.t())/2) ), dim=1) for U in self.Us] # W = W.T
        else:
            qs = [torch.sum(e1 * (e2 @ (U @ U.t()) ), dim=1) for U in self.Us] # W = W.T
        
        output = torch.stack(qs, dim= 1)

        return output
    
    def get_sim_mat(self):
        if self.add:
            Ms = torch.stack( [(U + U.t())/2 for U in self.Us] )# W = W.T
        else:
            Ms =  torch.stack( [ U @ U.t() for U in self.Us] )# W = W.T

        return Ms

    def save(self, path):
       torch.save({
            'model_state_dict': self.model.state_dict(),
            'sim_mat': self.sim_mat
        }, path)

    @classmethod
    def loadcls(cls, path, sentence_transformer_model):
        checkpoint = torch.load(path)
        model = sentence_transformer_model
        model.load_state_dict(checkpoint['model_state_dict'])
        sim_mat = checkpoint['sim_mat']
        return cls(model, sim_mat)
    
    @staticmethod
    def load(model_path, sentence_transformer_model):
        model_dict = torch.load(model_path)

        #print(model_dict.keys())
        #print(model_dict["metadata"])
        
        metadata = model_dict.pop('metadata')
        model_type = metadata.get('model_type',"ADD")
        normalized = metadata.get('normalized', False)
        embedding_dimension = metadata.get('embedding_dimension',768)

        bilinear_loss = BilinearLoss(
            model=sentence_transformer_model,
            sentence_embedding_dimension= embedding_dimension,  # 根据你的模型进行调整
            num_labels= sum(1 for key in model_dict if key.startswith("Us")), 
            normalized=normalized,
            ADD= True if model_type == "ADD" else False 
        )

        bilinear_loss.load_state_dict(model_dict)

        return bilinear_loss