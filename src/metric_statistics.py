import argparse
import os
from typing import Mapping, Tuple
from pathlib import Path
import json
from sys import set_coroutine_origin_tracking_depth
import numpy as np
import pandas as pd
import scipy.stats


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def get_data(collection_dir, log_dir: Path = Path('../logs')):
    ids = pd.read_csv(
        collection_dir / 'run_ids.txt', header=None
    ).values.flatten().tolist()
    all_metrics = {}
    for id in ids:
        metrics = load_json(str(log_dir / str(id) / 'metrics.json'))
        for key in metrics:
            values = np.array(metrics[key]['values']).reshape(1, -1)
            try:
                all_metrics[key].append(values)
            except KeyError as e:
                all_metrics[key] = [values]
    for key in all_metrics:
        try:
            all_metrics[key] = np.concatenate(all_metrics[key], axis=0) 
        except ValueError as e:
            all_metrics.pop(key)
    summary_stats = {
        key: (
            all_metrics[key].mean(axis=0),
            scipy.stats.sem(all_metrics[key], axis=0)
        )
        for key in all_metrics
    }
    return summary_stats


def save_collection_metrics(
        collection_dir: Path,
        summary_stats: Mapping[str, Tuple[np.ndarray, np.ndarray]]
):
    filename = collection_dir / 'metrics.txt'
    append_write = 'a' if os.path.exists(filename) else 'w'
    file = open(filename, append_write)
    for key in summary_stats:
        file.write(
            f'{key}: {summary_stats[key][0][-1].item():.5f} $\pm$ {summary_stats[key][1][-1].item():.5f}\n'
        )
    file.close()


def compute_and_save_metric_stats(
    collection: str, log_dir: Path = Path('../logs')
):
    collection_dir = log_dir / collection
    summary_stats = get_data(collection_dir)
    save_collection_metrics(collection_dir, summary_stats)


if __name__ == '__main__':
    log_dir = Path('../logs')
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Parse input directories')
    parser.add_argument('collection_dir')
    args = parser.parse_args()
    collections = (args.collection_dir,)
    # collections = ('collection_cifar100_mmlt_08092104', 'collection_cifar100_wsabi_08100159', 'collection_imagenet16-120_mmlt_08100650', 'collection_imagenet16-120_wsabi_08101200')

    for collection in collections:
        compute_and_save_metric_stats(collection, log_dir)
