# Bayesian Quadrature for Neural Ensemble Search

This repository provides code to run Bayesian Quadrature for Neural Ensemble Search over the NATS-Bench benchmark.

To run, execute (from the src directory):
```
python main.py with config/cifar10/bq-us.yaml --force
```
To run repeats with the seeds listed in `src/seeds.txt` use `repeat_runs.sh`:
```
./repeat_runs.sh -c config/cifar10/bq-us.yaml -s seeds.txt
```

The results will be output under a directory named `logs` under the project root.
Each run is labelled with an ID.
Meta-data on repeats is saved in a directory named `logs/collection-config\_name-time\_stamp`


Additional tools are provided in the `sandbox` directory.
Run these using
```
python post_process.py recalculate collection-name options
```
to generate the ensemble via a different method using the same design sets, or 
```
python post_process.py eval_truncated_design_set collection-name options
```
to generate an ensemble of a different size using the same design sets.
