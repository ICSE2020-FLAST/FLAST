# Know Your Neighbor: Fast Static Prediction of Test Flakiness

This repository is a companion page for the ICSE submission "Know Your Neighbor: Fast Static Prediction of Test Flakiness".

It contains all the material required for replicating the experiments, including: the algorithm implementation, the datasets and their ground truth, the scripts for the experiments replication, and the aggregated results to answer the research questions.


Experiment Replication
---------------
In order to replicate the experiment follow these steps:

### Getting started

1. Clone the repository:
   - `git clone https://github.com/ICSE2020-FLAST/FLAST`
 
2. If you do not have python3 installed you can get the appropriate version for your OS [here](https://www.python.org/downloads/).

3. Install the additional python packages required:
   - `pip3 install -r requirements.txt`

### Dataset creation
Decompress the datasets:
   - `tar zxvf datasets.tgz`
   
### Answering the Research Questions
Execute the research questions scripts:
   - `python3 py/rq-precision_recall_time.py` (RQ1, RQ4)
   - `python3 py/rq-trainset_size.py` (RQ2)
   - `python3 py/rq-afc_category_precision.py` (RQ3)
   - `python3 py/rq-storage_overhead.py` (RQ4)

Pseudocode
---------------
The pseudocode of FLAST is available [here](pseudocode/README.md).


Experiment Results
---------------
The cleaned data of the results reported in the paper are available [here](results/README.md).


Directory Structure
---------------
This is the root directory of the repository. The directory is structured as follows:

    FLAST
     .
     |
     |--- datasets/      Datasets and dataset information (links and commit hash). The folder is automatically created after the decompression of `datasets.tgz`.
     |
     |--- py/            Scripts with FLAST implementation and scripts to run experiments.
     |
     |--- results/       Folder containing the raw results of the experiments and the aggregate results. The folder is automatically created after the execution of the scripts for the experiments.
