import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\..')))
from jffpy.data import load_features
import faiss
import numpy as np
from tqdm import tqdm
# Print current date
from datetime import datetime
now = datetime.now()
print(now.strftime("%d/%m/%Y %H:%M:%S"))

gallerypath = "C:\\Users\\Joao\\Documents\\Repos\\Griaule\\Dados\\tse1kflat_sd27"
querypath = "C:\\Users\\Joao\\Documents\\Repos\\Griaule\\Dados\\sd27\\latent"

print(gallerypath)
print(querypath)
print()

# Building the index
d = 32 # dimension
index = faiss.IndexFlatL2(d)

# Load all the features
features, shapes, filenames = load_features(gallerypath, log_info = True, progress_bar=True, cache=True, cache_name="tse1kflat_sd27")
print()

queries, qshapes, qfilenames = load_features(querypath, log_info = True, progress_bar = False, cache=True, cache_name="sd27_latent")
print()

# Add the features to the index
print(index.is_trained) # Should return True
index.add(features)
print("Added", index.ntotal, "to index.")
print()

def purify_name(name):
    # 006022331520_E25916112409340610_dedo3.tpt
    # -> 31520_40610_d3
    if '-' in name:
        return name.split(".")[0].rjust(15)
    else:
        parts = name.split("_")
        frankenstein = parts[0][-5:] + "_" + parts[1][-5:] + "_d"
        fingerno = parts[2].split("dedo")[1].split(".")[0]
        if int(fingerno) < 10:
            frankenstein += "0"
        frankenstein += fingerno
        return frankenstein.rjust(15)


for i in range(len(filenames)):
    filenames[i] = purify_name(filenames[i])

max_queries = 10
progress_bar = tqdm(range(min(max_queries, len(qshapes))), desc="Processing queries")

accumulated = 0
for i in progress_bar:
    query = queries[accumulated:accumulated + qshapes[i]]
    accumulated += qshapes[i]

    # Query the index
    k = 16
    D, I = index.search(query, k)

    result = np.repeat(np.arange(len(shapes)), shapes)

    print()
    print(f"Query {i}", qfilenames[i], "(queries:" + str(qshapes[i]) + ")")
    for i in range(I.shape[0]):
        idxs = I[i]
        for j, idx in enumerate(idxs):
            print(filenames[result[idx]], ":", f"{D[i][j]:.4f}", sep="", end="  ")
        print()
    
    elapsed_time = progress_bar.format_dict['elapsed']
    print(f"Elapsed time after query {i}: {elapsed_time:.2f} seconds")