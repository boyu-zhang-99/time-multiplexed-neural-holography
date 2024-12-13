# Simultaneous Color Holography 

## Get started
To set up the required environment, you can create a Conda environment with all dependencies using the following commands:
```
conda env create -f environment.yml
conda activate tmnh
```

## Generate Reconstruction Results Using Different Loss Functions
Example target images are located in the `data/` folder.
New loss functions are implemented in the `loss_functions.py` file.

Run the following script to generate reconstruction results with different loss functions:
``` 
./run.sh
```

The reconstruction results will be saved in the `results/`. 
