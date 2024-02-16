import csv
import pickle as pkl
from sklearn.utils import Bunch
from mittens import GloVe
import numpy as np
import gc
import os

from tensorflow.keras import backend as K
K.clear_session()

os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'


# Co-occurrence
def countCOOC(cooccurrence, window, coreIndex):
    for index in range(len(window)):
        if index == coreIndex:
            continue
        else:
            cooccurrence[window[coreIndex]][window[index]] = cooccurrence[window[coreIndex]][window[index]] + 1
    return cooccurrence

filename = 'SITE'
loaddata = pkl.load(open('encoded8x23'+filename+'.pkl','rb'),encoding='latin1')

images = loaddata.images
target = loaddata.target

MATCH_ROW_NUMBER1 = {"AA": 1, "AC": 2, "AG": 3, "AT": 4, "CA": 5, "CC": 6, "CG": 7, "CT": 8, "GA": 9,
                    "GC": 10, "GG": 11, "GT": 12, "TA": 13, "TC": 14, "TG": 15, "TT": 16}
crispor = []
alphabet = 'AGCT'
print(target[0])
for i in range(images.shape[0]):
    arr = []
    arr.append(target[i])
    for j in range(images.shape[2]):
        temp = ''
        indexlist = []
        templist = list(images[i,:,j])
        for index,num in enumerate(templist):
            if num != 0:
                indexlist.append(index)
        # print(indexlist)
        temp = temp + alphabet[indexlist[0]]
        temp = temp + alphabet[indexlist[1]-4]
        arr.append(MATCH_ROW_NUMBER1[temp]-1)
        if i==0:
            print(temp)
    crispor.append(arr)

print(crispor[0])

tableSize = 16
coWindow = 5
vecLength = 100  # The length of the matrix
max_iter = 10000  # Maximum number of iterations
display_progress = 1000
cooccurrence = np.zeros((tableSize, tableSize), "int64")
print("An empty table had been created.")
print(cooccurrence.shape)

# Start statistics
flag = 0
for item in crispor:
    itemInt = [int(x) for x in item]
    for core in range(1, len(item)):
        if core <= coWindow + 1:
            window = itemInt[1:core + coWindow + 1]
            coreIndex = core - 1
            cooccurrence = countCOOC(cooccurrence, window, coreIndex)

        elif core >= len(item) - 1 - coWindow:
            window = itemInt[core - coWindow:(len(item))]
            coreIndex = coWindow
            cooccurrence = countCOOC(cooccurrence, window, coreIndex)

        else:
            window = itemInt[core - coWindow:core + coWindow + 1]
            coreIndex = coWindow
            cooccurrence = countCOOC(cooccurrence, window, coreIndex)

    flag = flag + 1
    if flag % 20 == 0:
        print("%s pieces of data have been calculated" % flag)

# del crispor, window
gc.collect()

coocPath = "cooccurrence_"+filename+"_%s.csv" % (coWindow)

f = open(coocPath,'w',newline='')

writer = csv.writer(f)
for item in cooccurrence:
    writer.writerow(item)


# GloVe
print("Start GloVe calculation")
coocMatric = np.array(cooccurrence, "float32")


glove_model = GloVe(n=vecLength, max_iter=max_iter,
                    display_progress=display_progress)
embeddings = glove_model.fit(coocMatric)

print(embeddings)
print(embeddings.shape)

del cooccurrence, coocMatric
gc.collect()

# Output calculation result
dicIndex = 0
# result=[]
# nowTime = tic.getNow().strftime('%Y%m%d_%H%M%S')
GlovePath = "keras_GloVeVec_"+filename+"_%s_%s_%s.csv" % (coWindow, vecLength,max_iter)

f = open(GlovePath,'w',newline='')

writer = csv.writer(f)

for embeddingsItem in embeddings:
    item = np.array([dicIndex])
    item = np.append(item, embeddingsItem)
    writer.writerow(item)
    dicIndex = dicIndex + 1
print(dicIndex)
print("Finished!")

f.close()
data = []
for i in range(len(crispor)):
    data.append(crispor[i][1:])

print(data[0])

new_coding = Bunch(
    target=loaddata.target,
    images=data
)

iswritepkl = True
if iswritepkl is True:
    pickle_out = open("encoded_CnnCrispr_"+filename+".pkl", "wb")
    pkl.dump(new_coding, pickle_out)
    pickle_out.close()