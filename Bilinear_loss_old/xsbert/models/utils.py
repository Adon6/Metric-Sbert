import zipfile
import os
from os.path import join, exists
from os import PathLike
import wget
import sys
from torch import cosine_similarity, dot, Tensor, einsum
#from . import XSMPNet, XSRoberta


def load_checkpoint(model_name: str):

    path = join('checkpoints', model_name)
    if not exists(path):
        zip_path = f'checkpoints/{model_name}.zip'
        if not zipfile.is_zipfile(zip_path):
            print('fetching checkpoint from LFS')
            os.system(f'git lfs pull --exclude="" --include="{zip_path}"')
        print('unzipping')
        with zipfile.ZipFile(zip_path, 'r') as f:
            f.extractall('checkpoints/')

    print('initializing')
    if model_name.endswith('roberta'):
        model = XSRoberta(path)
    elif model_name.endswith('mpnet'):
        model = XSMPNet(path)

    return model


def progress(current, total, width):
  progress_message = f"downloading model: {int(current / total * 100)}%"
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()

def load_model(name: str, model_dir: PathLike = '../xs_models/'):
    assert name in ['mpnet_cos', 'mpnet_dot','droberta_cos', 'droberta_dot'], \
        'available models are: mpnet_cos, mpnet_dot, droberta_cos, droberta_dot'
    if not exists(model_dir):
        os.makedirs(model_dir)
    path = join(model_dir, name)
    if not exists(path):
        zip_path = path + '.zip'
        if not exists(zip_path):
            url = f'https://www2.ims.uni-stuttgart.de/data/xsbert/{name}.zip'
            wget.download(url, zip_path, bar=progress)
            print()
        print('unzipping')
        with zipfile.ZipFile(zip_path, 'r') as f:
            f.extractall(model_dir)
    print('initializing')
    if name.startswith('droberta'):
        model = XSRoberta(path)
    elif name.startswith('mpnet'):
        model = XSMPNet(path)
    return model

def cossim(emb_a: Tensor, emb_b: Tensor, ref_a: Tensor, ref_b: Tensor, sim_mat: Tensor):
    score = cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
    ref_emb_a = cosine_similarity(emb_a.unsqueeze(0), ref_b.unsqueeze(0)).item()
    ref_emb_b = cosine_similarity(emb_b.unsqueeze(0), ref_a.unsqueeze(0)).item()
    ref_ref = cosine_similarity(ref_a.unsqueeze(0), ref_b.unsqueeze(0)).item()
    return score, ref_emb_a, ref_emb_b, ref_ref

def dotsim(emb_a: Tensor, emb_b: Tensor, ref_a: Tensor, ref_b: Tensor, sim_mat: Tensor):
    score = dot(emb_a, emb_b).item()
    ref_emb_a = dot(emb_a, ref_b).item()
    ref_emb_b = dot(emb_b, ref_a).item()
    ref_ref = dot(ref_a, ref_b).item()
    return score, ref_emb_a, ref_emb_b, ref_ref

def bilinear_sim(emb_a: Tensor, emb_b: Tensor, ref_a: Tensor, ref_b: Tensor, sim_mat: Tensor):
    score = einsum('i, Lip, p -> L', emb_a, sim_mat, emb_b)
    ref_emb_a = einsum('i, Lip, p -> L', emb_a, sim_mat, ref_a)
    ref_emb_b = einsum('i, Lip, p -> L', emb_b, sim_mat, ref_b)
    ref_ref = einsum('i, Lip, p -> L', ref_a, sim_mat, ref_b)
    return score, ref_emb_a, ref_emb_b, ref_ref

def gencos_sim(emb_a: Tensor, emb_b: Tensor, ref_a: Tensor, ref_b: Tensor, sim_mat: Tensor):
    pass

def softcos_sim(emb_a: Tensor, emb_b: Tensor, ref_a: Tensor, ref_b: Tensor, sim_mat: Tensor):
    pass