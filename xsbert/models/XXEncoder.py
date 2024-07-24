from . import XSTransformer
from torch import nn, Tensor, einsum   
from typing import Optional, Iterable, Dict, Union

class XXEncoder(XSTransformer):
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        modules: Optional[Iterable[nn.Module]] = None,
        device: Optional[str] = None,
        prompts: Optional[Dict[str, str]] = None,
        default_prompt_name: Optional[str] = None,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        truncate_dim: Optional[int] = None,
        sim_measure: Optional[str] = None,
        sim_mat: Optional[Tensor] = None,
    ):
        super.__init__(
            model_name_or_path,
            modules,
            device,
            prompts,
            default_prompt_name,
            cache_folder,
            trust_remote_code,
            revision,
            token, 
            use_auth_token,
            truncate_dim,
        )
        self.set_similarity(sim_measure= sim_measure, sim_mat= sim_mat)
    

    def explain_similarity_gen(
        self, 
        text_a: str, 
        text_b: str, 
        return_lhs_terms: bool = False,
        move_to_cpu: bool = False,
        verbose: bool = True,
        ):


        self.intermediates.clear()
        device = self[0].auto_model.embeddings.word_embeddings.weight.device

        emb_a, ref_a = self.get_sentence_embedding(text_a, device)
        interm_a = self.intermediates[0]
        J_a = self._compute_integrated_jacobian(emb_a, interm_a, move_to_cpu=move_to_cpu, verbose=verbose)
        da = interm_a[0] - interm_a[-1]# da size (Sa, Da)

        # repeat for b 
        emb_b, ref_b = self.get_sentence_embedding(text_b, device)
        interm_b = self.intermediates[1]
        J_b = self._compute_integrated_jacobian(emb_b, interm_b, move_to_cpu=move_to_cpu, verbose=verbose)
        db = interm_b[0] - interm_b[-1]


        """
        da, Ja.T, Jb, db.T -->
        sa-da,d-sa-da,d-db-sb,db-sb
        
        == i:Sa,j:Da, p:Sb,q:Db
        IJ = Ja.T,Jb ==> ijD,Dpq->ijpq
        da IJ db.T ==> ijpq*ijpq*ijpq->ijpq
        sum j and q==> ip
        ==
        da IJ db.T ==> ij,ijpq,pq -> iq
                
        """
        # comput J
        if not self.sim_mat:
            A = einsum('ij, Dij, Dpq, pq -> Liq', da, J_a, J_b, db)
        else:
            A = einsum('ij, Dij, Lip, Dpq, pq -> Liq', da, J_a, self.sim_mat, J_b, db) # to check

        A = A.detach().cpu()

        tokens_a = self.tokenize_text(text_a)
        tokens_b = self.tokenize_text(text_b)

        if return_lhs_terms:
            if move_to_cpu:
                ref_a = ref_a.detach().cpu()
                ref_b = ref_b.detach().cpu()

            score, ref_emb_a, ref_emb_b, ref_ref = self.sim_fun(emb_a[0], emb_b[0], ref_a, ref_b, self.sim_mat)

            return A, tokens_a, tokens_b, score, ref_emb_a, ref_emb_b, ref_ref
        else:
            return A, tokens_a, tokens_b
     
    
    def forward(self, features: dict):
        features = super().forward(features)
        return features