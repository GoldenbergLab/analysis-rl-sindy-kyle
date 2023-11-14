# SINDy for RL
This repository contains all analysis and processing scripts, as well as raw and processed data and model fits in support of the paper:
LaFollette K.J., Yuval J., Schurr R., Melnikoff D., & Goldenberg A. (2023) "Identifying Learning Models from Human Behavior using Bottom-Up Equation Discovery". PsyArXiv. https://osf.io/preprints/psyarxiv/65jqh.

## Repo Structure

### Phase 1
This folder contains all simulation code, presimulated data, and analysis scripts necessary to reproduce the simulation analyses detailed in Phase 1. The main notebook, `phase1_analysis.ipynb` is a step-by-step walkthrough of these analyses from simulation, to testing recovery as a function of parameter values, to robustness to noise analyses. The `simdata` for contains presimulated jsons for recoverability analyses, as those simulations can be time prohibitive. the `src` folder contains supporting scripts for simulation, construction of the candidate feature matrix provided to SINDy (which must be modified if features are to be adjusted), and plotting functions.

### Phase 2
This folder contains all raw and processed data, analysis scripts, Stan models, and Stan fits necessary to reproduce the empirical analyses detailed in Phase 2. The main notebook `phase2_analysis.ipynb` walks through these analyses from training SINDy on the processed data from both empirical studies, to visualization, and to comparing BICs between Stan versions of traditional models and SINDy's discovered model. The main notebook relies on processed data found in the `processing` folder, which themselves can be reproduced using the `preprocessing.ipynb` notebook. Stan models are located `stan/models` subfolder, and model fits are located in the `stan/fits` subfolder.

### Phase 3
This folder contains all data and original analysis scripts used by the authors of datasets curated for our Phase 3 model comparisons. The folder structure is largely kept intact as to what was used by the original authors. Modified versions of their scripts are included where appropriate. For example, in the `kooletal_reanalysis` folder, there is a `MB_MF_rllik.m` script containing the likelihood function for the original model. We've included a modified version of the script, `MB_MF_rrlik_sindy.m` containing a modified likelihood function with Quadratic Q-weighted models in place of the previously used delta-updating rules. See SI for the extent of changes made to the models, and the original author's materials for documentation on the modeling code.

## Packages & Version Control
The repository includes a YAML file, `environment_droplet.yml`, which can be used to set up a Conda environment with all necessary packages, dependencies, and correct versions. Follow the steps below to set up the Conda environment.

### Step 1: Install Conda
First you need to install Anaconda before you can start creating Conda environments. If you haven't, download and install from the Anaconda official website: https://www.anaconda.com/products/distribution

### Step 2: Create Environment with YAML
Navigate to the location of `environment_droplet.yml` and create the environment `sindyrl` with the following command in the terminal:
```
conda env create -f environment_droplet.yml
```
### Step 3: Activate Environment
After some downloading/installing, activate the new Conda environment with:
```
conda activate sindyrl
```

