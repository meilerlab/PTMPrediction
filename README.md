# PTMPrediction
Supplementary scripts for the paper "Combining machine learning with structure-based protein design to predict and engineer post-translational modifications of protein therapeutics"
## Rosetta documentation
Documentation for the Rosetta SimpleMetric can be found at https://www.rosettacommons.org/docs/latest/PTMPredictionMetric.
## Setup
A conda build with the required libraries can be created with `build_conda.sh`.
## Training
To train models from scratch use either the `training.py` or the `multi_training.py` scripts, e.g. `python ./training.py -p NlinkedGlycosylation`.
For training the ptm_data.csv.gz found in `./data/` needs to be uncompressed first.
## Feature calculation
A function for calculating the features used is in `calc_features.py` (requires PyRosetta to be installed in the conda environment).
## Deamidation Prediction
PDB files can be found in `./data` and deamidation probabilities can be calculated with `./deamidation.sh` which uses the `./XML/deamidation.xml` protocol.
## Influenza Prediction and Design
For designing the aquired glycosylation sites, use the `./influenza_design.sh` script. PDB files can be found in data and glycosylation probabilities can be calculated with `./influenza.sh` which uses the `./XML/influenza.xml` protocol.
## Phosphorylation engineering
In oder to run the Monte Carlo optimization use the `run_phospho_opt.sh` script. The input pdb structure is in `./data` and the RosettaScript protocol in `./XML`.
## Models
All trained models can be found in `./models/`.
