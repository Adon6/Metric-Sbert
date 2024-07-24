import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, Any
from os import PathLike
import torch
import gzip, csv
from os import PathLike
from sentence_transformers.readers import InputExample


def plot_attributions(A, tokens_a, tokens_b, 
    size: Tuple = (7, 7), 
    dst_path: Optional[PathLike] = None,
    show_colorbar: bool = False,
    cmap: str = 'RdBu',
    range: Optional[float] = None,
    shrink_colorbar: float = 1.,
    bbox = None
):
    if isinstance(A, torch.Tensor):
        A = A.numpy()
    assert isinstance(A, np.ndarray)
    Sa, Sb = A.shape
    assert len(tokens_a) == Sa and len(tokens_b) == Sb, 'size mismatch of tokens and attributions'
    if range is None:
        range = np.max(np.abs(A))
    f = plt.figure(figsize=size)
    plt.imshow(A, cmap=cmap, vmin=-range, vmax=range)
    plt.yticks(np.arange(A.shape[0]), labels=tokens_a)
    plt.xticks(np.arange(A.shape[1]), labels=tokens_b, rotation=50, ha='right')
    if show_colorbar:
        plt.colorbar(shrink=shrink_colorbar)
    if dst_path is not None:
        plt.savefig(dst_path, bbox_inches=bbox)
        plt.close()
    else:
        return f

from typing import Tuple, Optional, List, Union
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_attributions_multi(A: Union[np.ndarray, torch.Tensor], 
                      tokens_a: List[str], 
                      tokens_b: List[str], 
                      labels: List[Any],
                      size: Tuple[int, int] = (7, 7), 
                      dst_path: Optional[Union[str, PathLike]] = None,
                      show_colorbar: bool = False,
                      cmap: str = 'RdBu',
                      ranges: Optional[float] = None,
                      shrink_colorbar: float = 1.,
                      bbox = None):
    if isinstance(A, torch.Tensor):
        A = A.numpy()
    assert isinstance(A, np.ndarray)
    
    # Get the shape of A
    dim, Sa, Sb = A.shape
    
    assert len(tokens_a) == Sa and len(tokens_b) == Sb, 'size mismatch of tokens and attributions'
    
    if ranges is None:
        ranges = np.max(np.abs(A))
        
    fig, axes = plt.subplots(1, dim, figsize=(size[0] * dim, size[1]))
    
    if dim == 1:
        axes = [axes]
    
    for i in range(dim):
        ax = axes[i]
        ax.imshow(A[i], cmap=cmap, vmin=-ranges, vmax=ranges)
        ax.set_yticks(np.arange(Sa))
        ax.set_yticklabels(tokens_a)
        ax.set_xticks(np.arange(Sb))
        ax.set_xticklabels(tokens_b, rotation=50, ha='right')
        ax.set_title(labels[i]) 


        if show_colorbar:
            plt.colorbar(ax.imshow(A[i], cmap=cmap, vmin=-ranges, vmax=ranges), ax=ax, shrink=shrink_colorbar)
    
    if dst_path is not None:
        plt.savefig(dst_path, bbox_inches=bbox)
        plt.close()
    else:
        return fig



def input_to_device(inpt: dict, device: torch.device):
    for k, v in inpt.items():
        if isinstance(v, torch.Tensor):
            inpt[k] = v.to(device)


def load_sts_data(path: PathLike):
    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open(path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)
    return train_samples, dev_samples, test_samples

def load_nil_data(path: PathLike):
    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}

    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open(path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            label_id = label2int[row["label"]]
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id)

            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)
    return train_samples, dev_samples, test_samples