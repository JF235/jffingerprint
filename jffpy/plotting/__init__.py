import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import List, Callable, Optional, Dict


def plot_distance_distribution(features: List[np.ndarray], distance_func: Callable[[np.ndarray, np.ndarray], float], k: Optional[int] = None, query: Optional[np.ndarray] = None, ax: Optional[plt.Axes] = None, hist_kwargs: Optional[Dict] = None):
    """
    Plots the distance distribution of features with respect to an exemplar element.

    :param features: A list of feature vectors.
    :type features: List[np.ndarray]
    :param distance_func: A function to compute the distance between two feature vectors.
    :type distance_func: Callable[[np.ndarray, np.ndarray], float]
    :param k: The index of the exemplar element. If None, a random element is chosen.
    :type k: int, optional
    :param query: An exemplar element. If None, the element at index k or a random element is chosen.
    :type query: np.ndarray, optional
    :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
    :type ax: plt.Axes, optional
    :param hist_kwargs: Dictionary of keyword arguments to pass to the histogram plot.
    :type hist_kwargs: Dict, optional
    """
    if query is None:
        if k is not None:
            query = features[k]
        else:
            query = features[np.random.randint(len(features))]

    distances = [distance_func(query, feature) for feature in features]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if hist_kwargs is None:
        hist_kwargs = {'bins': 30, 'edgecolor': 'black'}

    ax.hist(distances, **hist_kwargs)

def plot_time(filename: str):
    # Carregar os dados salvos
    with open(filename, 'rb') as f:
        all_I, all_D, times = pickle.load(f)

    # Calcular o tempo total e o tempo médio por query
    total_times = 1e3 * np.array(times)
    avg_times = []
    for i in range(len(times)):
        avg_times.append(1e6 * times[i] / all_D[i].shape[0])

    # Plotar os histogramas
    plt.figure(figsize=(12, 5))

    # Check if total_times shold be adjusted to milliseconds/microseconds/seconds based on the average of each list
    if len(total_times) > 0:
        avg_total_times = sum(total_times) / len(total_times)
        if avg_total_times < 1:
            total_times = [t * 1000 for t in total_times]
            total_units = ' (µs)'
        elif avg_total_times > 1000:
            total_times = [t / 1000 for t in total_times]
            total_units = ' (s)'
        else:
            total_units = ' (ms)'

    if len(avg_times) > 0:
        avg_avg_times = sum(avg_times) / len(avg_times)
        if avg_avg_times > 1000:
            avg_times = [t / 1000 for t in avg_times]
            avg_units = ' (ms)'
        else:
            avg_units = ' (µs)'

    REMOVE_OUTLIERS = True
    if REMOVE_OUTLIERS:
        # If is 3x the standard deviation, remove it
        total_times = [t for t in total_times if abs(t - np.mean(total_times)) < 2 * np.std(total_times)]
        avg_times = [t for t in avg_times if abs(t - np.mean(avg_times)) < 2 * np.std(avg_times)]

    plt.subplot(1, 2, 1)
    plt.hist(total_times, bins=20, color='skyblue', edgecolor='black')
    plt.title('Histograma de Tempo Total' + total_units)
    plt.xlabel('Tempo Total' + total_units)
    plt.ylabel('Frequência')

    plt.subplot(1, 2, 2)
    plt.hist(avg_times, bins=40, color='salmon', edgecolor='black')
    plt.title('Histograma de Tempo Médio' + avg_units)
    plt.xlabel('Tempo Médio' + avg_units)
    plt.ylabel('Frequência')

    plt.tight_layout()
    plt.show()