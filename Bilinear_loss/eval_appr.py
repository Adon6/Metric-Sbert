
import sys
import os

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from Bilinear_loss.BilinearLoss import BilinearLoss
from xsbert.models import XSRoberta, ReferenceTransformer
from xsbert.utils import plot_attributions_multi

# Load model
model_path = "data/training_add2_nli_sentence-transformers-all-distilroberta-v1-2024-06-10_14-52-37+sD/eval/epoch9_step-1_sim_evaluation_add_matrix.pth"
sentence_transformer_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
model_name = "t+D9_"
bilinear_loss = BilinearLoss.load(model_path)

# Path for the Excel file
excel_path = f'figures/res_{model_name}.xlsx'
os.makedirs(os.path.dirname(excel_path), exist_ok=True)
# Initialize DataFrame to store results
results_df = pd.DataFrame(columns=['texta', 'textb', 'A_sum', 'score', 'ra', 'rb', 'rr', 'sum_true', 'loss', 'true_label', 'pred_label'])


# Assume bilinear_loss is already initialized
transformer_layer = bilinear_loss.model[0]

# Save transformer layer to file
save_path = 'temp_transformer'
transformer_layer.save(save_path)
transformer = ReferenceTransformer.load(save_path)

pooling = bilinear_loss.model[1]
test_sbert = XSRoberta(modules=[transformer, pooling], sim_measure="bilinear", sim_mat=bilinear_loss.get_sim_mat())

print("Load model successfully.")
print(test_sbert)
print("Start to calculate.")

test_sbert.to(torch.device('cuda'))
test_sbert.reset_attribution()
test_sbert.init_attribution_to_layer(idx=5, N_steps=100)

# Multiple pairs of text

true_labels = ['contradiction', 'entailment', 'neutral']  # Add corresponding true labels
from utils import text_pairs

for i, (texta, textb, label) in enumerate(text_pairs):
    if i > 300: 
        break
    print(f"Processing pair {i+1}:")
    print(f"texta: {texta}")
    print(f"textb: {textb}")
    print(f"label: {label}")
    
    A, tokens_a, tokens_b, score, ra, rb, rr = test_sbert.explain_similarity_gen(
        texta, 
        textb, 
        move_to_cpu=False,
        return_lhs_terms=True
    )

    #print("A.shape: ", A.shape)

    A_sum = A.sum(dim=(1, 2)).detach().cpu().numpy()
    score_np = score.detach().cpu().numpy()
    ra_np = ra.detach().cpu().numpy()
    rb_np = rb.detach().cpu().numpy()
    rr_np = rr.detach().cpu().numpy()

    sum_true = score_np - ra_np - rb_np + rr_np
    loss = A_sum - sum_true

    pred_label = true_labels[sum_true.argmax()]

    result_values = f"Asum{A_sum}com{sum_true}_loss{loss}_{label}_{pred_label}"

    fig_save_path = f"figures/fig_{model_name}_{i+1}_{result_values}.png"
    os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)

    plot_attributions_multi(
        A, 
        tokens_a, 
        tokens_b, 
        size=(5, 5),
        show_colorbar=True, 
        shrink_colorbar=.5,
        dst_path=fig_save_path,
        labels=["Contradiction", "Entailment", "Neutral"]
    )

    print(f"Saved figure to {fig_save_path}")
    print(A_sum, sum_true)

    # Create a new DataFrame for the current result
    new_data = pd.DataFrame({
        'texta': [texta],
        'textb': [textb],
        'A_sum': [A_sum.tolist()],
        'score': [score_np.tolist()],
        'ra': [ra_np.tolist()],
        'rb': [rb_np.tolist()],
        'rr': [rr_np.tolist()],
        'sum_true': [sum_true.tolist()],
        'loss': [loss.tolist()],
        'true_label': [label],
        'pred_label': [pred_label]
    })

    # Concatenate the new data with the existing DataFrame
    results_df = pd.concat([results_df, new_data], ignore_index=True)


    # Save DataFrame to Excel after each iteration
    results_df.to_excel(excel_path, index=False)

print("All pairs processed.")
