import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from xsbert.models import XSRoberta, ReferenceTransformer
from Bilinear_loss.BilinearLoss import BilinearLoss

DEBUG = True

# Load model
model_path = "data/+_sentence-transformers-all-distilroberta-v1-2024-08-15_06-25-33/eval/epoch9_step-1_sim_evaluation_add_matrix.pth"
sentence_transformer_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
bilinear_loss = BilinearLoss.load(model_path)

# Assume bilinear_loss is already initialized
transformer_layer = bilinear_loss.model[0]
save_path = 'transformer_layerztx'
transformer_layer.save(save_path)
transformer = ReferenceTransformer.load(save_path)
pooling = bilinear_loss.model[1]
test_sbert = XSRoberta(modules=[transformer, pooling], sim_measure="bilinear", sim_mat=bilinear_loss.get_sim_mat())
test_sbert.to(torch.device('cuda'))
test_sbert.reset_attribution()
test_sbert.init_attribution_to_layer(idx=5, N_steps=30)

# Define labels
label2index = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

# Initialize dictionary to store relation contributions
relation_contributions = {label: {} for label in label2index.keys()}

# Function to update relation contributions
def update_relation_contributions(relation_contributions, relation, label, avg_contribution):
    if relation not in relation_contributions[label]:
        relation_contributions[label][relation] = []
    relation_contributions[label][relation].append(avg_contribution)


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


# Load the spanex dataset
ds = load_dataset("copenlu/spanex", "snli")
samples = ds['test']  # Assuming we use the test split

for idx, example in enumerate(samples):
    if DEBUG and idx > 10:
            break

    premise = example['premise']
    hypothesis = example['hypothesis']
    label = example['label']
    relations = example['relations']
    
    if DEBUG:
        print(f"\nExample {idx + 1}")
        print(f"Premise: {premise}")
        print(f"Hypothesis: {hypothesis}")
        print(f"Label: {label}")

    # Find the most frequent relation
    relation_count = {}
    for relation in relations:
        for rel_label in relation['labels']:
            if rel_label not in relation_count:
                relation_count[rel_label] = 0
            relation_count[rel_label] += 1
    
    # Get the most frequent relation label
    most_frequent_relation = max(relation_count, key=relation_count.get)
    if DEBUG:
        print(f"Most Frequent Relation: {most_frequent_relation}")

    # Explain similarity using the SBERT model
    A, tokens_a, tokens_b, score, ra, rb, rr = test_sbert.explain_similarity_gen(
        premise, 
        hypothesis, 
        move_to_cpu=False,
        return_lhs_terms=True
    )
        
    # Adjust tokens for CLS and EOS tokens
    tokens_a = tokens_a[1:-1]
    tokens_b = tokens_b[1:-1]

    tids_a = calculate_tids(premise, tokens_a)
    tids_b = calculate_tids(hypothesis, tokens_b)
    
    # Process each relation in the example
    for relation in relations:
        if most_frequent_relation in relation['labels']:
            start_premise_char = relation['start_premise']
            end_premise_char = relation['end_premise']
            start_hypothesis_char = relation['start_hypothesis']
            end_hypothesis_char = relation['end_hypothesis']
            
            # Convert character indices to token indices using tokens from SBERT
            start_premise_token, end_premise_token = char_to_token_idx(tids_a, start_premise_char, end_premise_char)
            start_hypothesis_token, end_hypothesis_token = char_to_token_idx(tids_b, start_hypothesis_char, end_hypothesis_char)
            
            if DEBUG:
                # Print the token index ranges
                print(f"Premise Char Indices: {start_premise_char}-{end_premise_char}, Token Indices: {start_premise_token}-{end_premise_token}")
                print(f"Hypothesis Char Indices: {start_hypothesis_char}-{end_hypothesis_char}, Token Indices: {start_hypothesis_token}-{end_hypothesis_token}")
                print(f"Premise Char Text: '{premise[start_premise_char:end_premise_char]}', Extracted Premise Text from Tokens: '{" ".join(tokens_a[start_premise_token:end_premise_token])}'")
                print(f"Hypothesis Char Text: '{hypothesis[start_hypothesis_char:end_hypothesis_char]}', Extracted Hypothesis Text from Tokens: '{" ".join(tokens_b[start_hypothesis_token:end_hypothesis_token])}'")
             
            # Skip if tokenization failed
            if start_premise_token is None or end_premise_token is None or start_hypothesis_token is None or end_hypothesis_token is None:
                print("Skipping due to tokenization failure.")
                continue

            # Get the contribution matrix for all labels
            for label_name, label_index in label2index.items():
                contribution_matrix = A[label_index].detach().cpu().numpy()
                
                # Extract the relevant submatrix based on token indices
                submatrix = contribution_matrix[start_premise_token:end_premise_token, start_hypothesis_token:end_hypothesis_token]
                
                if DEBUG:
                    # Print the extracted submatrix
                    print(f"Extracted Submatrix Shape for Label '{label_name}': {submatrix.shape}")
                    print(f"Extracted Submatrix for Label '{label_name}':\n{submatrix}")
                
                # Calculate the average contribution for the relation under this label
                avg_contribution = np.mean(submatrix)
                if DEBUG:
                    print(f"Average Contribution for Relation '{most_frequent_relation}' under Label '{label_name}': {avg_contribution}")
                
                # Update the relation contributions dictionary for the current label
                update_relation_contributions(relation_contributions, most_frequent_relation, label_name, avg_contribution)


# Plot the histogram of contributions for each relation across different labels
output_dir = 'violin'
os.makedirs(output_dir, exist_ok=True)

'''histo
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
'''

# Plotting for each label
for label, relations in relation_contributions.items():
    plt.figure(figsize=(10, 6))
    
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