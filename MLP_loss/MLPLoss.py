import torch
from torch import nn, Tensor
from typing import Iterable, Dict, Callable, Optional
from sentence_transformers import SentenceTransformer
import logging


logger = logging.getLogger(__name__)


class MLPLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        num_labels: int = 3,
        loss_fct: Callable = nn.CrossEntropyLoss(),
        sentence_model_name: str = None,
        device :Optional[str] = None,
    ):
        """
        This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
        model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

        """
        super(MLPLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.model_name =  "MLP_"
        self.sentence_model_name = sentence_model_name
        self.embedding_dim = model.get_sentence_embedding_dimension()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")


        self.M1 = nn.ParameterList([nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim).to(self.device)) for _ in range(num_labels)])
        self.M2 = nn.ParameterList([nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim).to(self.device)) for _ in range(num_labels)])

        logger.info("MLP loss: #Labels: {}".format(num_labels))

        self.loss_fct = loss_fct


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor = None):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        e1, e2 = reps
        e1, e2 = Tensor(e1).to(self.device), Tensor(e2).to(self.device)
        self.M1, self.M2 = self.M1.to(self.device), self.M2.to(self.device)

        cosine_similarities = []
        for M1, M2 in zip(self.M1, self.M2):
            e1_transformed = torch.matmul(e1, M1)
            e2_transformed = torch.matmul(e2, M2)

            cos_sim = nn.functional.cosine_similarity(e1_transformed, e2_transformed, dim=-1)
            cosine_similarities.append(cos_sim)

        output = torch.stack(cosine_similarities, dim=1)

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output
   
    def sim(self, e1, e2):
        eval_device = 'cpu'
        e1, e2 = Tensor(e1).to(eval_device), Tensor(e2).to(eval_device)
        self.M1, self.M2 = self.M1.to(eval_device), self.M2.to(eval_device)

        cosine_similarities = []
        for M1, M2 in zip(self.M1, self.M2):
            e1_transformed = torch.matmul(e1, M1)
            e2_transformed = torch.matmul(e2, M2)

            cos_sim = nn.functional.cosine_similarity(e1_transformed, e2_transformed, dim=-1)
            cosine_similarities.append(cos_sim)

        output = torch.stack(cosine_similarities, dim=1)

        return output

    def save(self, path):
       torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_labels': self.num_labels,
            'embedding_dim': self.embedding_dim,
            'loss_fct' : self.loss_fct,
            'sentence_model_name' : self.sentence_model_name
        }, path)

    @classmethod
    def load(cls, path):
      
        checkpoint = torch.load(path)
        model = SentenceTransformer.load(checkpoint['sentence_model_name'])
        model.load_state_dict(checkpoint['model_state_dict'])
                
        modelmlp = cls(
            model = model,
            num_labels = checkpoint['num_labels'],
            loss_fct = checkpoint['loss_fct'],
            sentence_model_name = checkpoint['sentence_model_name'],
        )
          
        return modelmlp
        
