import argparse
from pathlib import Path
import importlib
import warnings

import sys
sys.path.append('../src')

from sandbox_utils import get_dataset_from_collection
from metric_statistics import compute_and_save_metric_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse input directories')
    parser.add_argument('post_process')
    parser.add_argument('collection_dir')
    parser.add_argument('pp_args', nargs='*')
    args = parser.parse_args()
    collection = Path('../logs') / args.collection_dir
    # collection = Path('../logs/collection_imagenet16-120_bq-us_04230534/')

    pp_module = importlib.import_module(args.post_process)
    run_pp = pp_module.run_post_processing

    if collection is not None:
        dataset = get_dataset_from_collection(args.collection_dir)
        pp_collection_dir = Path(
            f'../logs/{args.collection_dir}/{args.post_process}_{"_".join(args.pp_args)}'
        )
        with open(collection / 'run_ids.txt') as f:
            run_ids = f.readlines()
        run_ids = list(map(lambda x: int(x[:-1]), run_ids))
        for run_id in run_ids:
            try:
                run_pp(
                    dataset,
                    run_id,
                    *args.pp_args,
                    collection_dir=pp_collection_dir
                )
            except Exception as e:
                warnings.warn(f'Failed for {run_id}:\n{e}')
        compute_and_save_metric_stats(str(
            pp_collection_dir.relative_to(Path('../logs'))
        ))
    else:
        DATASET = 'ImageNet16-120'
        RUN_ID = 1572
        run_pp(DATASET, RUN_ID)
