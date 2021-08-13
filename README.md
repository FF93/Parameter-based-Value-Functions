# Parameter-Based Value Functions

This is the official research code for the paper Faccio et al. 2021:
"Parameter-Based Value Functions".

## Installation

Install the following dependencies (in a virtualenv preferably)
```bash
pip3 install wandb gym[all] mujoco_py>=2 torch==1.6.0 numpy
```

This code uses wandb for data logging and visualization


## Training

Change the default configuration in each file and run:

PSSVF:
```bash
python3 pssvf.py
```

PSVF:
```bash
python3 psvf.py
```

PAVF:
```bash
python3 pavf.py
```

## Citing

Please, cite our paper if you use our code or reimplement our method:

```bash
@inproceedings{faccio2020parameter,
  title={Parameter-Based Value Functions},
  author={Faccio, Francesco and Kirsch, Louis and Schmidhuber, J{\"u}rgen},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```
