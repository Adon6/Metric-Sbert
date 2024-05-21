
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import os
import tqdm

class EmbeddingModel(nn.Module):
    def __init__(self, smodel = 'all-MiniLM-L6-v2', normalized = False, device = "cuda"):
        super(EmbeddingModel, self).__init__()
        self.device = device

        self.normalized = normalized
        # 加载 SentenceBERT 模型
        self.sbert = SentenceTransformer(smodel).to(self.device)
        self.sbert.eval()

        self.embedding_dim = self.sbert.get_sentence_embedding_dimension()

    def forward(self, x):
        embeddings = self.sbert.encode(x, convert_to_tensor=True, show_progress_bar = False)
        if self.normalized:
            embeddings = nn.functional.normalize(embeddings, p=2)# example normalization

        return embeddings.to(self.device)
        
        
class SiameseNetworkBase(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class SiameseNetworkMUL(nn.Module):

    def __init__(self, num_classes, device= "cuda", smodel = 'all-MiniLM-L6-v2', normalized = False):
        super(SiameseNetworkMUL, self).__init__()
        self.normalized = normalized
        self.embedding_model = smodel
        self.model_name = self.__class__.__name__


        self.E = EmbeddingModel(smodel, normalized = normalized,device = device)
        self.emb_dim = self.E.embedding_dim
        #self.U = nn.Parameter(torch.randn(input_dim, input_dim)) 
        self.Us = nn.ParameterList([nn.Parameter(torch.randn(self.emb_dim, self.emb_dim)) for _ in range(num_classes)])


    def forward(self, x1, x2):
        e1 = self.E(x1)
        e2 = self.E(x2)
        qs = [torch.sum(e1 * (e2 @ (U @ U.t()) ), dim=1) for U in self.Us] # W = W.T
        q = torch.stack(qs, dim= 1)
        #out = self.classifier(q)
        return q
    

class SiameseNetworkADD(nn.Module):
    def __init__(self, num_classes, device= "cuda", smodel = 'all-MiniLM-L6-v2', normalized = False):
        super(SiameseNetworkADD, self).__init__()
        self.normalized = normalized
        self.embedding_model = smodel
        self.model_name = self.__class__.__name__


        self.E = EmbeddingModel(smodel, normalized = normalized,device = device)
        self.emb_dim = self.E.embedding_dim
        #self.U = nn.Parameter(torch.randn(input_dim, input_dim)) 
        self.Us = nn.ParameterList([nn.Parameter(torch.randn(self.emb_dim, self.emb_dim)) for _ in range(num_classes)])


    def forward(self, x1, x2):
        e1 = self.E(x1)
        e2 = self.E(x2)
        qs = [torch.sum(e1 * (e2 @ (U + U.t()) ), dim=1) for U in self.Us] # W = W.T
        q = torch.stack(qs, dim= 1)
        #out = self.classifier(q)
        return q
