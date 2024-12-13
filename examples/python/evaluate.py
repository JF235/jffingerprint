import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\..')))
from jffpy.data import load_features
from jffpy.printing import format_time, purify_name
from jffpy.plotting import plot_time
from jffpy.benchmark import eval_rank

plot_time("sd27_latent_results.pkl")
with open("tse1kflat_sd27.pkl", "rb") as f:
    feature_shapes, filenames = pickle.load(f)

for i in range(len(filenames)):
    filenames[i] = purify_name(filenames[i])

with open("sd27_latent.pkl", "rb") as f:
    _, qfilenames = pickle.load(f)

for i in range(len(qfilenames)):
    qfilenames[i] = purify_name(qfilenames[i]).lstrip()

with open("sd27_latent_results.pkl", "rb") as f:
    result = pickle.load(f)

eval_rank(result, filenames, qfilenames, feature_shapes, n = 128)


