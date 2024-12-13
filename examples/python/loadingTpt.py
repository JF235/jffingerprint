import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\..')))
from jffpy.data import load_features

filepath = "C:\\Users\\Joao\\Documents\\Repos\\Griaule\\Dados\\tse1kflat_sd27"

features, shapes, filenames = load_features(filepath, log_info = True, progress_bar=True, cache=True, cache_name="tse1kflat_sd27")