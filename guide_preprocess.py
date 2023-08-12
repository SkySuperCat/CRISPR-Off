import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.utils import Bunch

fpath = 'SITE.csv'
fname = 'SITE'

class Encoder:
    def __init__(self, target_rna, off_target_dna):
        self.target_rna = target_rna
        self.off_target_dna = off_target_dna
        self.encoded_dict_indel = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0],
                                   'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}

        self.direction_dict = {'A': 1, 'T': 2, 'G': 3, 'C': 4}
        self.encode_on_off_dim4()
        self.encode_on_off_dim6()
        self.encode_on_off_dim7()

    def encode_on_off_dim4(self):
        if len(self.target_rna) != len(self.off_target_dna):
            raise ValueError("The length of sgRNA and DNA are not matched!")

        pair_code = []
        for i in range(len(self.target_rna)):
            if self.target_rna[i] == 'N':
                self.target_rna[i] = self.off_target_dna[i]

            gRNA_base_code = self.encoded_dict_indel[self.target_rna[i]]
            DNA_based_code = self.encoded_dict_indel[self.off_target_dna[i]]
            pair_code.append(list(np.bitwise_or(gRNA_base_code, DNA_based_code)))

        self.on_off_code_4 = np.array(pair_code).reshape(1, 1, 23, 4)

    def encode_on_off_dim6(self):
        target_seq_code = np.array([self.encoded_dict_indel[base] for base in list(self.target_rna)])
        off_target_seq_code = np.array([self.encoded_dict_indel[base] for base in list(self.off_target_dna)])
        on_off_dim6_codes = []

        for i in range(len(self.target_rna)):
            diff_code = np.bitwise_or(target_seq_code[i], off_target_seq_code[i])
            dir_code = np.zeros(2)
            if self.direction_dict[self.target_rna[i]] == self.direction_dict[self.off_target_dna[i]]:
                diff_code = diff_code*-1
                dir_code[0] = 1
                dir_code[1] = 1
            elif self.direction_dict[self.target_rna[i]] < self.direction_dict[self.off_target_dna[i]]:
                dir_code[0] = 1
            elif self.direction_dict[self.target_rna[i]] > self.direction_dict[self.off_target_dna[i]]:
                dir_code[1] = 1
            else:
                raise Exception("Invalid seq!", self.target_rna, self.off_target_dna)
            on_off_dim6_codes.append(np.concatenate((diff_code, dir_code)))

        self.on_off_code_6 = on_off_dim6_codes

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
            on_off_dim7_codes.append(np.concatenate((diff_code, dir_code, [0])))  # Added an extra zero to make it 23x7
        self.on_off_code_7 = np.array(on_off_dim7_codes)


def save_to_pickle(data, dim, filename):
    fname = f"encoded{dim}x23{filename}.pkl"
    with open(fname, "wb") as pickle_out:
        pkl.dump(data, pickle_out)


def load_GUIDE_data(fname, fpath):
    dfguideSeq = pd.read_csv(fpath, sep=',')

    guide_labels = []

    guide_codes_4x23 = []
    guide_codes_6x23 = []
    guide_codes_7x23 = []

    for idx, row in dfguideSeq.iterrows():
        target_rna = row['sgrna']
        off_target_dna = row['otdna']
        label = row['label']
        en = Encoder(target_rna=target_rna, off_target_dna=off_target_dna)

        guide_codes_4x23.append(en.on_off_code_4)
        guide_codes_6x23.append(en.on_off_code_6)
        guide_codes_7x23.append(en.on_off_code_7)

        guide_labels.append(label)

    # Convert guide_labels to numpy ndarray
    guide_labels = np.array(guide_labels)


    save_to_pickle(Bunch(target=guide_labels, images=np.array(guide_codes_4x23).reshape(-1, 23, 4, order='C')), 4, fname)
    save_to_pickle(Bunch(target=guide_labels, images=np.array(guide_codes_6x23).reshape(-1, 23, 6, order='C')), 6, fname)
    save_to_pickle(Bunch(target=guide_labels, images=np.array(guide_codes_7x23).reshape(-1, 23, 7, order='C')), 7, fname)


load_GUIDE_data(fname, fpath)

dfGuideSeq = pd.read_csv(fpath, sep=',')

def one_hot_encode_seq(data):
    """One-hot encoding of the sequences."""
    # define universe of possible input values
    alphabet = 'AGCT'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in data]

    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)

    # invert encoding
    inverted = int_to_char[np.argmax(onehot_encoded[0])]

    return onehot_encoded


def flatten_one_hot_encode_seq(seq):
    """Flatten one hot encoded sequences."""
    return np.asarray(seq).flatten(order='C')

# we store the image in im
im = np.zeros((len(dfGuideSeq), 8, 23))

cnt = 0
for n in range(len(dfGuideSeq)):
    arr1 = one_hot_encode_seq(dfGuideSeq.loc[n, 'sgrna'])
    arr1 = np.asarray(arr1).T
    arr2 = one_hot_encode_seq(dfGuideSeq.loc[n, 'otdna'])
    arr2 = np.asarray(arr2).T
    arr = np.concatenate((arr1, arr2))
    im[n] = arr
    cnt += 1

# we put the results in bunch
guideseq8x23 = Bunch(
    # target_names=dfGuideSeq['name'].values,
    target=dfGuideSeq['label'].values,
    images=im)

iswritepkl = True
if iswritepkl is True:
    # we create the pkl file for later use
    pickle_out = open("encoded8x23"+fname+".pkl", "wb")
    pkl.dump(guideseq8x23, pickle_out)
    pickle_out.close()

#9x23
new9x23 = np.zeros((im.shape[0],9,23))
position = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
for n in range(im.shape[0]):
    for m in range(im.shape[1]):
        for k in range(im.shape[2]):
            if im[n,m,k] != 0:
                new9x23[n,m,k] = 1
    new9x23[n,8] = position

new9x23 = new9x23.transpose((0,2,1))

new_coding = Bunch(
    # target_names=dfGuideSeq['name'].values,
    target=dfGuideSeq['label'].values,
    images=new9x23
)

iswritepkl = True
if iswritepkl is True:
    # we create the pkl file for later use
    pickle_out = open("encodedposition9x23"+fname+".pkl", "wb")
    pkl.dump(new_coding, pickle_out)
    pickle_out.close()

#14x23
encoded_list = np.zeros((im.shape[0],5,23))
for n in range(im.shape[0]):
    for m in range(im.shape[2]):
        arr1 = im[n,0:4,m].tolist()
        # print(arr1)
        arr2 = im[n,4:8,m].tolist()
        # print(arr2)
        arr = []
        if arr1 == arr2:
            arr = [0,0,0,0,0]
        else:
            arr = np.add(arr1,arr2).tolist()
            if (arr == [1,1,0,0]) or (arr == [0,0,1,1]):
                arr.append(1)
            else:
                arr.append(-1)
        encoded_list[n,:,m] = arr


new9x23 = np.zeros((im.shape[0],14,23))
position = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
for n in range(im.shape[0]):
    for m in range(im.shape[1]):
        for k in range(im.shape[2]):
            if im[n,m,k] != 0:
                new9x23[n,m,k] = 1
    new9x23[n,8:13] = encoded_list[n]
    new9x23[n,13] = position


new9x23 = new9x23.transpose((0,2,1))

new_coding = Bunch(
    target=dfGuideSeq['label'].values,
    images=new9x23
)

iswritepkl = True
if iswritepkl is True:
    pickle_out = open("encodedmismatchtype14x23"+fname+".pkl", "wb")
    pkl.dump(new_coding, pickle_out)
    pickle_out.close()