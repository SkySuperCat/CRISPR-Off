import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.utils import Bunch

fpath = 'CIRCLE.csv'
fname = 'CIRCLE'

class Encoder:
    def __init__(self, target_rna, off_target_dna):
        tlen = 24
        self.target_rna = "-" * (tlen - len(target_rna)) + target_rna
        self.off_target_dna = "-" * (tlen - len(off_target_dna)) + off_target_dna
        self.encoded_dict_indel = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0],
                                   'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 0]}
        self.direction_dict = {'A': 5, 'G': 4, 'C': 3, 'T': 2, '_': 1}
        self.encode_on_off_dim4()

    def encode_on_off_dim4(self):
        encoded_list = []
        for char in self.off_target_dna:
            encoded_list.append(self.encoded_dict_indel[char][:4])
        self.on_off_code_4 = np.array(encoded_list)[1:]

def save_to_pickle(data, dim, filename):
    fname = f"encoded{dim}x23{filename}.pkl"
    with open(fname, "wb") as pickle_out:
        pkl.dump(data, pickle_out)


def load_GUIDE_data(fname, fpath):
    dfguideSeq = pd.read_csv(fpath, sep=',')

    guide_labels = []

    guide_codes_4x23 = []

    for idx, row in dfguideSeq.iterrows():
        target_rna = row['sgRNA_seq']
        off_target_dna = row['off_seq']
        label = row['label']
        en = Encoder(target_rna=target_rna, off_target_dna=off_target_dna)

        guide_codes_4x23.append(en.on_off_code_4)

        guide_labels.append(label)

    # Convert guide_labels to numpy ndarray
    guide_labels = np.array(guide_labels)


    save_to_pickle(Bunch(target=guide_labels, images=np.array(guide_codes_4x23).reshape(-1, 23, 4, order='C')), 4, fname)

load_GUIDE_data(fname, fpath)