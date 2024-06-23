import torch
from BilinearLoss import BilinearLoss
from sentence_transformers import SentenceTransformer
from xsbert.models import XSBert,XSRoberta,ReferenceTransformer
import xsbert.utils as utils


# load model
model_path = 'data\\training_add2_nli_sentence-transformers-all-distilroberta-v1-2024-06-10_14-52-37\eval\epoch9_step-1_sim_evaluation_add_matrix.pth'
sentence_transformer_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

"""
transformer_layer = sentence_transformer_model[0]
print("Transformer Layer:\n", transformer_layer)

# 打印 Transformer 层的配置信息
print("\nTransformer Configuration:\n", transformer_layer.auto_model.config)

# 打印 Transformer 层的参数
print("\nTransformer Parameters:")
for name, param in transformer_layer.auto_model.named_parameters():
    print(f"Parameter: {name}, Size: {param.size()}")
input()
"""

bilinear_loss = BilinearLoss.load(model_path, sentence_transformer_model)

# 假设 bilinear_loss 是已经初始化的对象
transformer_layer = bilinear_loss.model[0]

# 保存 transformer 层到文件
save_path =  'transformer_layerxx'
transformer_layer.save(save_path)
transformer = ReferenceTransformer.load(save_path)

pooling = bilinear_loss.model[1]
test_sbert = XSRoberta(modules=[transformer, pooling], sim_measure= "bilinear", sim_mat= bilinear_loss.get_sim_mat())

#test_sbert = XSRoberta(modules = bilinear_loss.model, sim_measure= "bilinear", sim_mat= Ms)

print("Load model successfully.")
print(test_sbert)
print("Start to calculate.")

test_sbert.to(torch.device('cuda'))
test_sbert.reset_attribution()
test_sbert.init_attribution_to_layer(idx=5, N_steps=60)

texta = 'The coffee is bad.'
textb = 'This is not a good coffee.'

print("Sim.")

# Notice A is multifolds 
A, tokens_a, tokens_b, score, ra, rb, rr = test_sbert.explain_similarity_gen(
    texta, 
    textb, 
    move_to_cpu=False,
    return_lhs_terms=True
)

print("A.shape: ", A.shape)


f = utils.plot_attributions_multi(
        A, 
        tokens_a, 
        tokens_b, 
        size=(5, 5),
        show_colorbar=True, 
        shrink_colorbar=.5,
        dst_path= "figsave_all",
        labels=["Entailment", "Neutral", "Contradiction"]
    )


print(A.sum(dim=(1, 2)), score - ra - rb + rr)

"""
for i in range(len(A)):
    f = utils.plot_attributions(
        A[i], 
        tokens_a, 
        tokens_b, 
        size=(5, 5),
        range=.3,
        show_colorbar=True, 
        shrink_colorbar=.5,
        dst_path= "figsave" +str(i)
    )
"""