import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\..')))
from jffpy.data import load_features
from jffpy.printing import format_time, purify_name
import faiss
import numpy as np
import pickle
from tqdm import tqdm
from numba import njit, jit
# Print current date
from datetime import datetime


now = datetime.now()
print(now.strftime("%d/%m/%Y %H:%M:%S"))

gallerypath = "C:\\Users\\Joao\\Documents\\Repos\\Griaule\\Dados\\tse1kflat_sd27"
querypath = "C:\\Users\\Joao\\Documents\\Repos\\Griaule\\Dados\\sd27\\latent"
qcache = "sd27_latent"

print(gallerypath)
print(querypath)

# Load all the features
features, shapes, filenames = load_features(gallerypath, log_info = True, progress_bar=True, cache=True, cache_name="tse1kflat_sd27")

result = np.repeat(np.arange(len(shapes)), shapes)
for i in range(len(filenames)):
    filenames[i] = purify_name(filenames[i])

queries, qshapes, qfilenames = load_features(querypath, log_info = True, progress_bar = False, cache=True, cache_name=qcache)

# Create two arrays to store mean and stddev of all features from gallery
# Each mean/stddev is computed between accumulated and accumulated + shapes[i]

# Preallocate arrays
mean = np.zeros((len(shapes), features.shape[1]))
stddev = np.ones((len(shapes), features.shape[1]))

accumulated = 0
for i in range(len(shapes)):
    f = features[accumulated:accumulated + shapes[i]]
    mean[i] = np.mean(f, axis=0)
    stddev[i] = np.std(f, axis=0)
    # Apply a shift for all gallery features (features - mean) / stddev
    features[accumulated:accumulated + shapes[i]] = (f - mean[i]) / stddev[i]

    accumulated += shapes[i]

################################################################################################

# Otimizar a partir daqui usando numba
@njit(fastmath=True)
def sequential_search(features, shapes, queries, qshapes, mean, stddev, result, filenames, max_queries):
    all_D = []
    all_I = []

    acc = 0
    for p in range(max_queries):
        query = queries[acc:acc + qshapes[p]]
        acc += qshapes[p]

        qidxs = []
        qdist = []
        for qf in query:  # Query Feature
            # Preallocate for all distances
            distances = np.zeros(len(result))
            accumulated = 0
            for i in range(len(shapes)):
                sf = features[accumulated:accumulated + shapes[i]]  # Shifted Features
                sqf = (qf - mean[i]) / stddev[i]  # Shifted Query Feature

                # Compute the distance between the query and all gallery features
                distances[accumulated:accumulated + shapes[i]] = np.sum((sf - sqf) ** 2, axis=1)
                accumulated += shapes[i]
            # Get the indices of the k closest and the distances
            k = 16
            idxs = np.argsort(distances)[:k]
            dd = distances[idxs]

            qidxs.append(idxs)
            qdist.append(dd)

        all_I.append(qidxs)
        all_D.append(qdist)

    return all_I, all_D

max_queries = 2
# Perform the sequential search using Numba
# time sequential_search
start = datetime.now()
all_I, all_D = sequential_search(features, shapes, queries, qshapes, mean, stddev, result, filenames, max_queries)
print("Sequential search time:", datetime.now() - start)

p = 0
for I, D in zip(all_I, all_D):
    I, D = np.array(I), np.array(D)
    print()
    print(f"Query {p}", qfilenames[p], "(queries:" + str(qshapes[p]) + ")")
    for i in range(I.shape[0]):
        idxs = I[i]
        for j, idx in enumerate(idxs):
            print(filenames[result[idx]], ":", f"{D[i][j]:.4f}", sep="", end="  ")
        print()
    p += 1

with open(qcache + '_results_numba.pkl', 'wb') as f:
    pickle.dump((all_I, all_D), f)