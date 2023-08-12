import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.utils import Bunch

class Encoder:
    def __init__(self, target_rna, off_target_dna):
        tlen = 24
        self.target_rna = "-" * (tlen - len(target_rna)) + target_rna
        self.off_target_dna = "-" * (tlen - len(off_target_dna)) + off_target_dna
        self.encoded_dict_indel = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0],
                                   'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 0]}
        self.direction_dict = {'A': 5, 'G': 4, 'C': 3, 'T': 2, '_': 1}
        self.encode_on_off_dim4()
        self.encode_on_off_dim6()
        self.encode_on_off_dim7()
        self.encode_on_off_dim8()
        self.encode_on_off_dim9()
        self.encode_on_off_dim14()

    def encode_on_off_dim4(self):
        encoded_list = []
        for char in self.off_target_dna:
            encoded_list.append(self.encoded_dict_indel[char][:4])
        self.on_off_code_4 = np.array(encoded_list)[1:]

    def encode_on_off_dim6(self):
        encoded_dict = self.encoded_dict_indel
        on_bases = list(self.target_rna)
        off_bases = list(self.off_target_dna)
        on_off_dim6_codes = []

        for i in range(len(on_bases)):
            on_b = on_bases[i]
            off_b = off_bases[i]

            # Convert the base encodings to arrays for arithmetic operations
            on_b_array = np.array(encoded_dict[on_b][:4])
            off_b_array = np.array(encoded_dict[off_b][:4])

            # If the bases are the same, simply use the encoding and append [0, 0] for directionality
            if on_b == off_b:
                combined_code = np.append(on_b_array, [0, 0])
            else:
                # Use bitwise OR for differences
                diff_code = np.bitwise_or(on_b_array, off_b_array)

                # Encode the directionality
                dir_code = np.array([0, 0])
                if self.direction_dict[on_b] > self.direction_dict[off_b]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1

                combined_code = np.concatenate((diff_code, dir_code))

            on_off_dim6_codes.append(combined_code)

        self.on_off_code_6 = np.array(on_off_dim6_codes)[1:]

    def encode_on_off_dim7(self):
        encoded_dict = self.encoded_dict_indel
        on_bases = list(self.target_rna)
        off_bases = list(self.off_target_dna)
        on_off_dim7_codes = []
        for i in range(len(on_bases)):
            on_b = on_bases[i]
            off_b = off_bases[i]
            diff_code = np.bitwise_or(encoded_dict[on_b], encoded_dict[off_b])
            dir_code = np.zeros(2)
            if on_b == "-" or off_b == "-" or self.direction_dict[on_b] == self.direction_dict[off_b]:
                pass
            else:
                if self.direction_dict[on_b] > self.direction_dict[off_b]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            on_off_dim7_codes.append(np.concatenate((diff_code, dir_code)))
        self.on_off_code_7 = np.array(on_off_dim7_codes)[1:]

    def encode_on_off_dim8(self):
        on_encoded = np.array([self.encoded_dict_indel[base][:4] for base in self.target_rna])
        off_encoded = np.array([self.encoded_dict_indel[base][:4] for base in self.off_target_dna])
        self.on_off_code_8 = np.concatenate((on_encoded, off_encoded), axis=1)[1:]


    def encode_on_off_dim9(self):
        self.encode_on_off_dim8()
        position = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
        position_encoded = np.array(position).reshape(-1, 1)
        self.on_off_code_9 = np.concatenate((self.on_off_code_8, position_encoded), axis=1)

    def encode_on_off_dim14(self):
        self.encode_on_off_dim9()
        mismatch_encoded = np.zeros((self.on_off_code_9.shape[0], 5))
        for i in range(self.on_off_code_9.shape[0]):
            on_base = self.on_off_code_9[i, 0:4]
            off_base = self.on_off_code_9[i, 4:8]
            if not np.array_equal(on_base, off_base):
                mismatch_encoded[i, :4] = on_base + off_base
                if (np.array_equal(on_base + off_base, [1,1,0,0])) or (np.array_equal(on_base + off_base, [0,0,1,1])):
                    mismatch_encoded[i, 4] = 1
                else:
                    mismatch_encoded[i, 4] = -1
        self.on_off_code_14= np.concatenate((self.on_off_code_9, mismatch_encoded), axis=1)


def save_to_pickle(data, dim, filename):
    if dim == 9:
        fname = f"encodedposition{dim}x23{filename}.pkl"
    elif dim == 14:
        fname = f"encodedmismatchtype{dim}x23{filename}.pkl"
    else:
        fname = f"encoded{dim}x23{filename}.pkl"

    with open(fname, "wb") as pickle_out:
        pkl.dump(data, pickle_out)


def load_CIRCLE_data():
    filename = 'CIRCLE'
    fpath = 'CIRCLE.csv'
    dfCircleSeq = pd.read_csv(fpath, sep=',')

    circle_labels = []

    circle_codes_4x23 = []
    circle_codes_6x23 = []
    circle_codes_7x23 = []
    circle_codes_8x23 = []
    circle_codes_9x23 = []
    circle_codes_14x23 = []

    for idx, row in dfCircleSeq.iterrows():
        target_rna = row['sgRNA_seq']
        off_target_dna = row['off_seq']
        label = row['label']
        en = Encoder(target_rna=target_rna, off_target_dna=off_target_dna)

        circle_codes_4x23.append(en.on_off_code_4)
        circle_codes_6x23.append(en.on_off_code_6)
        circle_codes_7x23.append(en.on_off_code_7)
        circle_codes_8x23.append(en.on_off_code_8)
        circle_codes_9x23.append(en.on_off_code_9)
        circle_codes_14x23.append(en.on_off_code_14)

        circle_labels.append(label)

    # Convert circle_labels to numpy ndarray
    circle_labels = np.array(circle_labels)

    save_to_pickle(Bunch(target=circle_labels, images=np.array(circle_codes_4x23).reshape(-1, 23, 4, order='C')), 4, filename)
    save_to_pickle(Bunch(target=circle_labels, images=np.array(circle_codes_6x23).reshape(-1, 23, 6, order='C')), 6, filename)
    save_to_pickle(Bunch(target=circle_labels, images=np.array(circle_codes_7x23).reshape(-1, 23, 7, order='C')), 7, filename)
    save_to_pickle(Bunch(target=circle_labels, images=np.array(circle_codes_8x23).reshape(-1, 23, 8, order='C')), 8, filename)
    save_to_pickle(Bunch(target=circle_labels, images=np.array(circle_codes_9x23).reshape(-1, 23, 9, order='C')), 9, filename)
    save_to_pickle(Bunch(target=circle_labels, images=np.array(circle_codes_14x23).reshape(-1, 23, 14, order='C')), 14, filename)


load_CIRCLE_data()

