import logging
import os
from pathlib import Path
from timeit import repeat

import torch
from nats_bench import create
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver

from model.arch_ensemble import ArchEnsemble

logger = logging.getLogger(__name__)
torch.set_printoptions(linewidth=79)

# Create Sacred experiment.
ex = Experiment()
# Add observer to which experiment configuration and results will be added.
ex.observers.append(
    MongoObserver(
        url='mongodb://bqnasUser:FU4EuM0z@mongo:27017/bqnas?authMechanism=SCRAM-SHA-1',
        db_name='bqnas'
    )
)
ex.observers.append(FileStorageObserver('../logs'))

@ex.automain
def run_experiment(_config=None, _run=None):
    """Run the experiment. The arguments are automatically filled in by
    sacred.
    
    :param _config: Sacred configuration (dictionary).
    :param _run: Sacred Run object.
    """
    _config = {} if _config is None else _config
    # Set up logging.
    log_dir = Path(_config.get('experiment', {}).get('log_dir', '../logs'))
    run_id = _run._id if _run is not None else 0
    log_dir /= str(run_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    # If running repeats.
    if _config.get('collection', None) is not None:
        collection_dir = log_dir.parent / str(_config['collection'])
        collection_dir.mkdir(parents=True, exist_ok=True)
        filename = collection_dir / 'run_ids.txt'
        append_write = 'a' if os.path.exists(filename) else 'w'
        file = open(filename, append_write)
        file.write(f'{_run._id}\n')
        file.close()
        file = open(filename, 'r')
        repeat_num = len(file.readlines())
        file.close()
        seeds_filename = collection_dir / 'seeds.txt'
        seeds_file = open(seeds_filename, append_write)
        seeds_file.write(f"{_config['seed']}\n")
        seeds_file.close()
    if _config.get('load_collection', None) is not None:
        # Load the design set computed by some other method. (For the
        # baselines).
        load_collection_dir = log_dir.parent / str(_config['load_collection'])
        filename = load_collection_dir / 'run_ids.txt'
        file = open(filename, 'r')
        load_design_set = int(file.readlines()[repeat_num - 1])
        file.close()
    else:
        load_design_set = _config['numerics'].get('load_design_set', None)

    # Define dataset
    dataset = _config.get('dataset', 'cifar10')

    # Define search space
    api = create(
        None,
        'tss',
        fast_mode=_config.get('search_space', {}).get('fast_mode', True),
        verbose=False
    )
    ensemble = ArchEnsemble(
        dataset=dataset,
        api=api,
        data_dir=Path(
            _config.get('experiment', {}).get('data_dir', '../data')
        ),
        numerics=_config.get('numerics', None)
    )

    # Learn the weights of the ensemble
    learn_output = ensemble.learn(
        numerics=_config.get('numerics', None),
        load_design_set=load_design_set,
        log_dir=log_dir,
        sacred_run=_run,
        debug_options=_config.get('debug_options', {})
    )

    # Evaluate the ensemble.
    ensemble.evaluate(
        learn_output=learn_output,
        numerics=_config.get('numerics', None),
        log_dir=log_dir,
        sacred_run=_run,
        smoke_test=_config.get('debug_options', {}).get('smoke_test', False)
    )
