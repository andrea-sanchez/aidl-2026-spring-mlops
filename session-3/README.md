# Session 3

Train a model remotely using Google Cloud Compute Engine.

## Dataset

Remember you can download the dataset [here](https://www.kaggle.com/olavomendes/cars-vs-flowers) (direct [download](https://www.kaggle.com/api/v1/datasets/download/olavomendes/cars-vs-flowers)).


## Installation
### With Conda
Create a conda environment by running

```bash
conda create --name aidl-session3 python=3.9
```

Then, activate the environment

```bash
conda activate aidl-session3
```

and install the dependencies

```bash
pip install -r requirements.txt
```

## Running the project

To run the project
```bash
python session-3/main.py
```


## Possible issues

If you observe the following error in the (cloud) computer:

```
Error: OSError: [Errno 28] No space left on device
````

even though the VM disk had plenty of space, the installation failed because pip downloads packages to the /tmp folder, which on many VM's has a small size limit. Large packages (like PyTorch) can exceed that limit.

**Quick solution** install the packages using conda instead, which internally handles temporary files differently and avoids the issue:

```
conda install numpy matplotlib Pillow pytorch torchvision cpuonly -c pytorch
```

*kudos to Montse for the feedback :) 