# Neural Reward Machines
Official repository for the paper "Neural Reward Machines. Elena Umili, Francesco Argenziano and Roberto Capobianco. accepted by the 2024 European Conference in Artificial Intelligence (ECAI 2024)".

## Requirements
- pytorch
- gym
- pygame

## How reproduce the experiments
To reproduce the experiments in the paper run the script ```experiments.py```

The script uses some flags, run ``` python experiments.py --helpfull``` to see the full list of parameters used.

```
       USAGE: experiments.py [flags]
flags:

experiments.py:
  --ENV: Environment to test, one in ['map_env', 'image_env'], default= 'map_env'
    (default: 'map_env')
  --LOG_DIR: path where to save the results, default='Results/'
    (default: 'Results/')
  --METHOD: Method to test, one in ['rnn', 'nrm', 'rm'], default= 'rnn'
    (default: 'rnn')
  --NUM_EXPERIMENTS: num of runs for each test, default= 5
    (default: '5')
    (an integer)


```
