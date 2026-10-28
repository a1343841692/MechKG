"""Unchanged seed-site feature definitions from trusted_pipeline/sequence_evidence.py."""
import re
import numpy as np

def count_overlapping(seq, motif):
    count = 0; start = 0
    while True:
        where = seq.find(motif, start)
        if where < 0: return count
        count += 1; start = where + 1

def site_features(mirna, utr):
    if not mirna or not utr or len(mirna) < 8 or re.search('[^ACGT]', mirna):
        return np.zeros(6, np.float32)
    complement = str.maketrans('ACGT', 'TGCA')
    six = mirna[1:7].translate(complement)[::-1]
    seven = mirna[1:8].translate(complement)[::-1]
    n6 = count_overlapping(utr, six); m8 = count_overlapping(utr, seven)
    a1 = count_overlapping(utr, six + 'A'); n8 = count_overlapping(utr, seven + 'A')
    exclusive = [n8, m8 - n8, a1 - n8, n6 - m8 - a1 + n8]
    assert min(exclusive) >= 0
    return np.array([*np.log1p(exclusive), np.log1p(len(utr)), 1.], dtype=np.float32)
