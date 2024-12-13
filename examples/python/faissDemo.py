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
print()

# Building the index
d = 32 # dimension
index = faiss.IndexFlatL2(d)

# Load all the features
features, shapes, filenames = load_features(gallerypath, log_info = True, progress_bar=True, cache=True, cache_name="tse1kflat_sd27")
print()

queries, qshapes, qfilenames = load_features(querypath, log_info = True, progress_bar = False, cache=True, cache_name=qcache)
print()

# Add the features to the index
print(index.is_trained) # Should return True
index.add(features)
print("Added", index.ntotal, "to index.")
print()

print("=== END OF HEADER ===")

for i in range(len(filenames)):
    filenames[i] = purify_name(filenames[i])

# Listas para armazenar D e I
all_D = []
all_I = []
times = []

max_queries = np.inf
progress_bar = tqdm(range(min(max_queries, len(qshapes))), desc="Processing queries")
old_time = 0

accumulated = 0
for i in progress_bar:
    query = queries[accumulated:accumulated + qshapes[i]]
    accumulated += qshapes[i]

    # Query the index
    k = 16
    D, I = index.search(query, k)
    all_D.append(D)
    all_I.append(I)

    result = np.repeat(np.arange(len(shapes)), shapes)

    print()
    print(f"Query {i}", qfilenames[i], "(queries:" + str(qshapes[i]) + ")")
    for i in range(I.shape[0]):
        idxs = I[i]
        for j, idx in enumerate(idxs):
            print(filenames[result[idx]], ":", f"{D[i][j]:.4f}", sep="", end="  ")
        print()
    
    elapsed_time = progress_bar.format_dict['elapsed']
    times.append(elapsed_time - old_time)
    print(f"Elapsed time after query {i}: {format_time(times[-1])}")
    old_time = elapsed_time


with open(qcache + '_results.pkl', 'wb') as f:
    pickle.dump((all_I, all_D, times), f)