import torch
import time
import numpy as np
from torchmetrics import HammingDistance


def cosine(a, b):
    return torch.cosine_similarity(a, b)

def euclidean(a, b):
    return torch.cdist(a, b)

def clip(a, b):
    logits_per_text = torch.matmul(a, b.t())
    probs = logits_per_text.softmax(dim=1)
    return probs.argmax(1)

def band(a, b):
    return a & b

def bmul(a, b):
    return a * b

def bxor(a, b):
    return torch.bitwise_xor(a, b)

def numpy_nonzero(a, b):
    return np.count_nonzero(a != b)

def normal_test():
    a = torch.rand((1, 768), dtype=torch.float32)
    b = torch.rand((5000, 768), dtype=torch.float32)
    euc = []
    cos = []
    cl = []
    for i in range(1010):
        t = time.time()
        euclidean(a, b)
        if i >= 10:
            euc.append(time.time() - t)
        t = time.time()
        cosine(a, b)
        if i >= 10:
            cos.append(time.time() - t)
        t = time.time()
        clip(a, b)
        if i >= 10:
            cl.append(time.time() - t)

    print(f'Euclidean Distance: {np.mean(euc)}')
    print(f'Cosine Similarity: {np.mean(cos)}')
    print(f'CLIP Contrastive: {np.mean(cl)}')


def binary_test():
    a = (torch.rand((1, 2048)) > 0.5).byte()
    b = (torch.rand((5000, 2048)) > 0.5).byte()
    ba = []
    bm = []
    bx = []
    for i in range(1010):
        t = time.time()
        band(a, b)
        if i >= 10:
            ba.append(time.time() - t)
        t = time.time()
        bmul(a, b)
        if i >= 10:
            bm.append(time.time() - t)
        t = time.time()
        bxor(a, b)
        if i >= 10:
            bx.append(time.time() - t)

    print(f'Binary And: {np.mean(ba)}')
    print(f'Binary Multiplcative: {np.mean(bm)}')
    print(f'Binary XOR: {np.mean(bx)}')


if __name__ == '__main__':
    normal_test()
    binary_test()
