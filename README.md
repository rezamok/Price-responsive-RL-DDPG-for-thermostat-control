Reproducible Code and Data for: "Price-responsive control using deep reinforcement learning for heating systems: Simulation and living lab experiment"



\## Overview

This repository is used to train a price-responsive reinforcement learning controller to control indoor thermostat of a room. The environment is a PCNN model, embedded in gym. 

This repository contains the code and instructions to reproduce the results presented in our manuscript:



> Reza Mokhtari, Mina Montazeri, Hanmin Cai, Phillip Heer, Rongling Li, "Price-responsive control using deep reinforcement learning for heating systems: Simulation and living lab experiment" Energy, 2025.  



The repository includes:

\- Source code for all experiments and simulations

\- Instructions for setting up the environment

\- Scripts to reproduce all figures and tables



---



\## Repository Structure

── checkpoints/ # Contains the checkpoint for training progress

├── kosta/ # Contains the main files, environment and hyperparameters for the RL

├── saves/ # The environment model and historical data are placed in this folder

├── requirements.txt # Python dependencies

├── working\_directory/ # Contains the source codes to run training and testing

├──  working\_directory/run\_algorithm1.py # Source code to train the model

├──  working\_directory/run\_test\_agent\_new.py # Source code to test the trained model with test dataset

├── LICENSE # License file

└── README.md # This file





\## Installation

\### Option 1 — Using `pip`

```bash

pip install -r requirements.txt



\### Option 2 — Using Conda (recommended for reproducibility)

bash

Copy

Edit

conda env create -f environment.yml

conda activate myenv



\## Usage

After including the environment model and historical data, run run\_algorithm1.py located in the working\_directory. For hyperparameter settings, go to kosta/hyperparameters.py 





\## Contact

For questions, please contact:

Reza Mokhtari — remok@dtu.dk



