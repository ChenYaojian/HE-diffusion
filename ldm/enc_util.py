#Author: Yaojian Chen

import tenseal as ts

def enc_sum(enc_a):
    shape = enc_a.shape
    dim = len(shape)
    enc_sum = enc_a.sum()
    for i in range(dim-1):
        enc_sum = enc_sum.sum()
    return enc_sum
