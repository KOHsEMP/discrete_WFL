# A Unified Framework for Generalization Error Analysis of Learning with Arbitrary Discrete Weak Features

## requirements

* Python: 3.9.14

* Library: Please see `requirements.txt`


## Directory Structure

```
codes_discrete/
　├README.md
　├config/
　│　├exp_adult_full_base.yaml
　│　├exp_bank_full_base.yaml
　│　├exp_census_full_base.yaml
　│　└exp_kick_full_base.yaml
　├exp1_shell/
　│　├exp1_adult_full.sh
　│　├exp1_bank_full.sh
　│　├exp1_census_full.sh
　│　└exp1_kick_full.sh
　├data/
　│　├adult/
　│　│　├ ...
　│　├bank/
　│　│　├ ...
　│　├kick/
　│　│　├ ...
　│　└census/
　│　 　├ ...
　├libs/
　│　├learning.py
　│　├load_data.py
　│　├models.py
　│　├utils_processing.py
　│　└utils.py
　├requirements.txt
　├exp1.py
　└calc_bound.ipynb
```

* `config`: This is a directory that stores YAML files, which contain arguments that are common and fixed across the experiment scripts.
* `exp1_shell`: Shell scripts for executing experimental programs.
* `data`: This is a directory that stores the datasets used in the experiments. When executing the program, specify the path to the data directory as an argument.
* `libs`: This is a directory that stores functions and other utilities used in main.py.
* `exp1.py`: This is the experiment script.
* `calc_bound.ipynb`: The code calculates the derived error bounds. Visualization of experimental results is also provided.



## Datasets download Links

* [Adult](https://archive.ics.uci.edu/dataset/2/adult)

    * Unzip `adult.zip`, and place all the files located directly under it into `./data/adult/`.

* [Bank Marketing; Bank](https://archive.ics.uci.edu/dataset/222/bank+marketing)

    * Unzip `bank+marketing.zip`, and place all the files inside `bank.zip` into `./data/bank/`.

* [kick; Kick](https://openml.org/search?type=data&sort=qualities.NumberOfInstances&status=active&format=ARFF&id=41162)
    * Downlaod `kick.arff` and place `kick.arff` into `./data/kick/`.

* [Census-Income (KDD); Census](https://archive.ics.uci.edu/dataset/117/census+income+kdd)
    * Unzip `census+income+kdd.zip`, and place all the files inside `census+income+kdd.zip` into `./data/bank/`.


## How to Execute Experiments

```bash

# full experiments using Adult dataset 
bash ./exp1_shell/exp1_adult_full.sh

```

The explanation of the main arguments is follow:

**Experiental Settings:**

* `dataset_name`: Using dataset name. Please select from ['bank', 'adult', 'kick', 'census'].

* `data_dir`: The path of `data` directory.

* `output_dir`: Path of output directory for log data. 

* `weak_cols`: List of features to be weak features

* `sample_size`: All data size. If sample_size = -1, we use all data

* `test_rate`: Test data rate

* `use_train_size`: Size of the training data. Randomly selected from samples not assigned to test data.

* `seed`: Random Seed

* `arch`: Architecture for a label prediction model. Please choose 'mlp'.

* `hd`: The size of Hidden dimension for arch=='mlp'.

* `lr`: Learning rate

* `bs`: Batch size

* `ep`: The number of epochs

* `wd`: Weight decay

* `pred_loss`: Loss function for learning label prediction model.

* `est_error_rate`: the error rate of feature estimation models.