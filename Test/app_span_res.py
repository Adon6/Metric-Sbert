import sys
import os
import numpy as np
import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from xsbert.models import XSMPNet, XSRoberta, ReferenceTransformer
from xsbert.utils import plot_attributions_multi
from Bilinear_loss.BilinearLoss import BilinearLoss

DEBUG = False
in_device = torch.device('cuda')

# Load model
model_path = "data/o_sentence-transformers-all-mpnet-base-v2-2024-08-15_06-25-38/eval/epoch9_step-1_sim_evaluation_o_matrix.pth"
model_name = "oM"
#sentence_transformer_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
bilinear_loss = BilinearLoss.load(model_path)

# Assume bilinear_loss is already initialized
transformer_layer = bilinear_loss.model[0]
save_path = 'transformer_layermx'
transformer_layer.save(save_path)
transformer = ReferenceTransformer.load(save_path)
pooling = bilinear_loss.model[1]

#create sbert for test and initialize
test_sbert = XSMPNet(modules=[transformer, pooling], sim_measure="bilinear", sim_mat=bilinear_loss.get_sim_mat())
test_sbert.to(in_device)
test_sbert.reset_attribution()
test_sbert.init_attribution_to_layer(idx=11, N_steps=100)

# Release bilinear_loss from memory if no longer needed
del bilinear_loss

# Define labels
label2index = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
labellist = list(label2index.keys())

# excel
# Path for the Excel file
output_prefix = f'{model_name}out_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
result_IJ_path = output_prefix + f'res_IJ.xlsx'
os.makedirs(os.path.dirname(result_IJ_path), exist_ok=True)

# Initialize DataFrame to store results
results_df = pd.DataFrame(columns=['texta', 'textb', 'A_sum', 'score', 'ra', 'rb', 'rr', 'sum_true', 'loss', 'true_label', 'pred_label'])

# Initialize dictionary to store relation contributions
relation_contributions = {label: {} for label in label2index.keys()}

# Function to update relation contributions
def update_relation_contributions(relation_contributions, relation, label, avg_contribution):
    if relation not in relation_contributions[label]:
        relation_contributions[label][relation] = []
    relation_contributions[label][relation].append(avg_contribution.item())

def calculate_tids(text, tokens):
    """
    Calculate the mapping of character indices to token indices for a given text and tokens.
    """
    
    if DEBUG:
        print(text)
        print(tokens)
    
    tids = []
    current_token_idx = 0
    current_token_text = tokens[current_token_idx].replace("▁", "")  # Remove special token indicators
    char_idx_in_token = 0
    
    for char_idx, char in enumerate(text):
        if char_idx_in_token >= len(current_token_text):
            current_token_idx += 1
            if current_token_idx < len(tokens):
                current_token_text = tokens[current_token_idx].replace("▁", "")
                char_idx_in_token = 0
            else:
                break
        
        if char.isspace():
            if len(tids) > 0:
                tids.append(tids[-1])
            continue
        
        tids.append(current_token_idx)
        char_idx_in_token += 1
    
    return tids

def char_to_token_idx(tids, start_char_idx, end_char_idx):
    """
    Extract the relevant submatrix from the contribution matrix using character indices and token indices.
    """
    # Convert character indices to token indices using the precomputed tids
    start_token_idx = tids[start_char_idx] if start_char_idx < len(tids) else tids[-1]
    end_token_idx = tids[end_char_idx-1] + 1 if end_char_idx-1 < len(tids) else tids[-1] + 1
    
    return start_token_idx, end_token_idx

def draw_interpretation(pack_ex, texta, textb, label, idx):
    A, tokens_a, tokens_b, score, ra, rb, rr = pack_ex

    A_sum = A.sum(dim=(1, 2)).detach().cpu().numpy()
    score_np = score.detach().cpu().numpy()
    ra_np = ra.detach().cpu().numpy()
    rb_np = rb.detach().cpu().numpy()
    rr_np = rr.detach().cpu().numpy()

    sum_true = score_np - ra_np - rb_np + rr_np
    loss = A_sum - sum_true

    pred_label = labellist[score_np.argmax()]

    result_values = f"Asum{A_sum}com{sum_true}_loss{loss}_{label}_{pred_label}"

    fig_save_path = output_prefix + f"figures/fig_{model_name}_{idx}_{result_values}.png"
    os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)

    plot_attributions_multi(
        A, 
        tokens_a, 
        tokens_b, 
        size=(5, 5),
        show_colorbar=True, 
        shrink_colorbar=.5,
        dst_path=fig_save_path,
        labels=labellist,
    )

    global true_labels, pred_labels, results_df
    pred_labels.append(pred_label)
    true_labels.append(label)

    if DEBUG:
        print(f"Saved figure to {fig_save_path}")
        print(f"Predicted label: {pred_label}, true label {label}")
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
    results_df.to_csv(result_IJ_path, index=False)

def save_results_and_metrics(true_labels, pred_labels):
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    classification_rep = classification_report(true_labels, pred_labels, target_names=labellist)

    # Save metrics to a text file
    output_file = output_prefix + f'final/{model_name}_metrics.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write(f"F1-Score (weighted): {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{classification_rep}\n")

def compute_token_embedding(texta, textb):
    global test_sbert, in_device

    emb_a = test_sbert.get_token_embeddings(texta, device = in_device)
    emb_b = test_sbert.get_token_embeddings(textb, device = in_device)

    # move to cpu
    if emb_a.is_cuda:
        emb_a = emb_a.cpu()
    if emb_b.is_cuda:
        emb_b = emb_b.cpu()

    emb_a_np = emb_a.detach().numpy()  # shape: (len(tokens_a), embedding_dim)
    emb_b_np = emb_b.detach().numpy()  # shape: (len(tokens_b), embedding_dim)

    cossim_M = cosine_similarity(emb_a_np, emb_b_np)
    return cossim_M.tolist()

def process_example(idx, example):
    global results_df, output_prefix
    
    # Process relations
    relation_details_path = output_prefix + f'relation/{idx}_details.json'
    os.makedirs(os.path.dirname(relation_details_path), exist_ok=True)
    
    example_forsave = {
        'example_idx': idx,
        'premise': "",
        'hypothesis': "",
        'label': "",
        'relations': [],
    }

    try:
        premise = example['premise']
        hypothesis = example['hypothesis']
        label = example['label']
        relations = example['relations']
        
        if DEBUG:
            print(f"\nExample {idx}")
            print(f"Premise: {premise}")
            print(f"Hypothesis: {hypothesis}")
            print(f"Label: {label}")

        example_forsave['example_idx'] = idx
        example_forsave['premise'] = premise
        example_forsave['hypothesis'] = hypothesis
        example_forsave['label'] = label

        # Explain similarity using the SBERT model
        pack_ex = test_sbert.explain_similarity_gen(premise, hypothesis, move_to_cpu=False, return_lhs_terms=True)
        A, tokens_a, tokens_b, score, ra, rb, rr = pack_ex
        
        example_forsave['A'] = A.tolist()
        example_forsave['Cossim_M'] = compute_token_embedding(premise, hypothesis)
        example_forsave['tokens_A'] = tokens_a
        example_forsave['tokens_B'] = tokens_b

        # Draw interpretation and save figure
        draw_interpretation(pack_ex, premise, hypothesis, label, idx)
        
        # Convert tokens to TIDs
        tokens_a = tokens_a[1:-1]
        tokens_b = tokens_b[1:-1]
        
        tids_a = calculate_tids(premise, tokens_a)
        tids_b = calculate_tids(hypothesis, tokens_b)
        
        for relation in relations:
            rel_res = process_relation(relation, tids_a, tids_b, premise, hypothesis, A, tokens_a, tokens_b)
            example_forsave['relations'].append(rel_res) 
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"Out of memory error at index {idx}. Skipping this example...")
            torch.cuda.empty_cache()
        else:
            raise e
    except Exception as e:
        print(f"Error processing example {idx}: {e}")

    with open(relation_details_path, 'w') as f:
        json.dump(example_forsave, f, indent=4)

def process_relation(relation, tids_a, tids_b, premise, hypothesis, A,  tokens_a, tokens_b):
    # Process relation and update contributions
    relation_info = {
        'most_frequent_relation': "",
        'premise_char_indices': (0, 0),
        'hypothesis_char_indices': (0, 0),
        'premise_token_indices': (0, 0),
        'hypothesis_token_indices': (0, 0),
        'premise_char_text': "",
        'hypothesis_char_text': "",
        'extracted_premise_tokens': "",
        'extracted_hypothesis_tokens': "",
        'contributions_per_label': []
    }

    try:
        most_frequent_relation = find_most_frequent_relation(relation['labels'])
        
        if DEBUG:
            print(f"Most Frequent Relation: {most_frequent_relation}")

        start_premise_char = relation['start_premise']
        end_premise_char = relation['end_premise']
        start_hypothesis_char = relation['start_hypothesis']
        end_hypothesis_char = relation['end_hypothesis']

        # Convert character indices to token indices
        start_premise_token, end_premise_token = char_to_token_idx(tids_a, start_premise_char, end_premise_char)
        start_hypothesis_token, end_hypothesis_token = char_to_token_idx(tids_b, start_hypothesis_char, end_hypothesis_char)
        extracted_premise_tokens = " ".join(tokens_a[start_premise_token:end_premise_token])
        extracted_hypothesis_tokens = " ".join(tokens_b[start_hypothesis_token:end_hypothesis_token])
        if DEBUG:
            # Print the token index ranges
            print(f"Premise Char Indices: {start_premise_char}-{end_premise_char}, Token Indices: {start_premise_token}-{end_premise_token}")
            print(f"Hypothesis Char Indices: {start_hypothesis_char}-{end_hypothesis_char}, Token Indices: {start_hypothesis_token}-{end_hypothesis_token}")
            print(f"Premise Char Text: '{premise[start_premise_char:end_premise_char]}', Extracted Premise Text from Tokens: '{extracted_premise_tokens}'")
            print(f"Hypothesis Char Text: '{hypothesis[start_hypothesis_char:end_hypothesis_char]}', Extracted Hypothesis Text from Tokens: '{extracted_hypothesis_tokens}'")
            

        # Skip if tokenization failed
        if start_premise_token is None or end_premise_token is None or start_hypothesis_token is None or end_hypothesis_token is None:
            print("Skipping due to tokenization failure.")
            return

        relation_info['most_frequent_relation'] = most_frequent_relation
        relation_info['premise_char_indices'] = (start_premise_char, end_premise_char)
        relation_info['hypothesis_char_indices'] = (start_hypothesis_char, end_hypothesis_char)
        relation_info['premise_token_indices'] = (start_premise_token, end_premise_token)
        relation_info['hypothesis_token_indices'] = (start_hypothesis_token, end_hypothesis_token)
        relation_info['premise_char_text'] = premise[start_premise_char:end_premise_char]
        relation_info['hypothesis_char_text'] = hypothesis[start_hypothesis_char:end_hypothesis_char]
        relation_info['extracted_premise_tokens'] = extracted_premise_tokens
        relation_info['extracted_hypothesis_tokens'] = extracted_hypothesis_tokens
        relation_info['contributions_per_label'] = {}
        

        # Update contributions
        for label_name, label_index in label2index.items():
            contribution_matrix = A[label_index].detach().cpu().numpy()
            submatrix = contribution_matrix[start_premise_token:end_premise_token, start_hypothesis_token:end_hypothesis_token]
            avg_contribution = np.mean(submatrix)

            if DEBUG:
                # Print the extracted submatrix
                print(f"Extracted Submatrix Shape for Label '{label_name}': {submatrix.shape}")
                print(f"Extracted Submatrix for Label '{label_name}':\n{submatrix}")
                print(f"Average Contribution for Relation '{most_frequent_relation}' under Label '{label_name}': {avg_contribution}")
            
            # Log the submatrix and contribution for this label
            relation_info['contributions_per_label'][label_name] = {
                'submatrix_shape': submatrix.shape,
                'submatrix_values': submatrix.tolist(),
                'average_contribution': avg_contribution.item()
            }

            update_relation_contributions(relation_contributions, most_frequent_relation, label_name, avg_contribution)

        return relation_info
    except Exception as e:
        print(f"Error processing relation: {e}")

        return relation_info

def find_most_frequent_relation(relation_labels):
    relation_count = {}
    for rel_label in relation_labels:
        if rel_label not in relation_count:
            relation_count[rel_label] = 0
        relation_count[rel_label] += 1
    return max(relation_count, key=relation_count.get)

def plot_histo(relation_contributions):
    output_dir = output_prefix + '/hist'
    os.makedirs(output_dir, exist_ok=True)
    # Plot the histogram of contributions for each relation across different labels
    for label, relations in relation_contributions.items():
        plt.figure(figsize=(10, 6))
        
        # Plot histograms for each relation in the same label
        for relation, contributions in relations.items():
            plt.hist(contributions, bins=30, alpha=0.5, label=relation)
        
        plt.title(f"Contribution Distribution per Relation for Label: {label}")
        plt.xlabel("Contribution")
        plt.ylabel("Frequency")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()

        # Save the figure
        file_path = os.path.join(output_dir, f"contribution_distribution_{label}.png")
        plt.savefig(file_path)
        plt.close()

def plot_violin(relation_contributions):
    # Plot the histogram of contributions for each relation across different labels
    output_dir = output_prefix + '/violin'
    os.makedirs(output_dir, exist_ok=True)

    # Plotting for each label
    for label, relations in relation_contributions.items():
        plt.figure()
        
        # Prepare data for violin plot
        data = []
        relation_labels = []
        for relation, contributions in relations.items():
            data.extend(contributions)  # Append all contribution values
            relation_labels.extend([relation] * len(contributions))  # Corresponding relation labels

        # Plot violin plot
        sns.violinplot(x=relation_labels, y=data, inner='quartile', palette='muted')
        
        # Set plot title and labels
        plt.title(f"Distribution of Average Contributions for Label: {label}")
        plt.xlabel("Relation")
        plt.ylabel("Average Contribution")
        
        # Show the plot
        plt.tight_layout()

        # Save the figure
        file_path = os.path.join(output_dir, f"contribution_distribution_{label}.png")
        plt.savefig(file_path)
        plt.close()

true_labels = []
pred_labels = []

# Load the spanex dataset
ds = load_dataset("copenlu/spanex", "snli")
samples = ds['test']  # Assuming we use the test split

for idx, example in enumerate(samples):
    if DEBUG and idx > 20:
            break
    process_example(idx, example)
    if idx % 100 == 99:
        torch.cuda.empty_cache()
# Save final results and metrics
save_results_and_metrics(true_labels, pred_labels)

# Path for the Excel file
contribution_path = output_prefix + f'res_contribution.json'
os.makedirs(os.path.dirname(contribution_path), exist_ok=True)
with open(contribution_path, 'w') as f:
    json.dump(relation_contributions, f, indent=4)

# plot violin
plot_violin(relation_contributions)
