import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\..')))
from jffpy.data import load_features
from jffpy.printing import format_time, purify_name
import faiss
import numpy as np
import pickle
from tqdm import tqdm
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

all_D = []
all_I = []
times = []

max_queries = 2
progress_bar = tqdm(range(min(max_queries, len(qshapes))), desc="Processing queries")
old_time = 0

acc = 0
for p in progress_bar:
    query = queries[acc:acc + qshapes[p]]
    acc += qshapes[p]

    print()
    print(f"Query {p}", qfilenames[p], "(queries:" + str(qshapes[p]) + ")")
    # Make a sequential search for the closest features

    accumulated = 0
    qidxs = []
    qdist = []
    for qf in query: # Query Feature
        # Preallocate for all distances
        distances = np.zeros(len(result))
        accumulated = 0
        for i in range(len(shapes)):
            sf = features[accumulated:accumulated + shapes[i]] # Shifted Features
            sqf = (qf - mean[i]) / stddev[i] # Shifted Query Feature

            # Compute the distance between the query and all gallery features
            distances[accumulated:accumulated + shapes[i]] = np.sum( (sf - sqf)**2, axis=1)
            accumulated += shapes[i]
            
        # Get the indices of the k closest and the distances
        k = 16
        idxs = np.argsort(distances)[:k]
        dd = distances[idxs]

        qidxs.append(idxs)
        qdist.append(dd)
        
        for idx in idxs:
            print(filenames[result[idx]], ":", f"{distances[idx]:.4f}", sep="", end="  ")
        print()
    
    all_I.append(qidxs)
    all_D.append(qdist)

    elapsed_time = progress_bar.format_dict['elapsed']
    times.append(elapsed_time - old_time)
    print(f"Elapsed time after query {p}: {format_time(times[-1])}")
    old_time = elapsed_time