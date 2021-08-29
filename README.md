# PyTorch-Project-Template
## Installation
```bash
$ git clone git@github.com:tomoino/PyTorch-Project-Template.git
```

## Usage
### Setup
```bash
$ cd PyTorch-Project-Template
$ sh docker/build.sh
$ sh docker/run.sh
$ sh docker/exec.sh
```

### Start a new project
1. Add yaml file to "./configs/project"
    ```bash
    $ vi ./configs/project/new_project.yaml
    ```
1. Run train.py with -cn (or --config-name) flag to specify project
    ```bash
    $ python train.py -cn new_project
    ```

### Training
```bash
$ python train.py
```
#### Grid Search
You can run train.py with multiple different configurations.
```bash
$ python train.py -m \
    train.batch_size=16,32 \
    train.optimizer.lr=0.01,0.001
```
#### Evaluation
```bash
$ python train.py train.eval=True model.initial_ckpt=best_ckpt.pth
```

### Check the results
You can use MLflow to check the results of your experiment.
Access http://localhost:5000/ from your browser.
If necessary, you can edit env.sh to change the port.

### Jupyter Lab
You can experiment with JupyterLab.
```bash
jupyterlab
```


## How to customize
### Add dataset
1. Add module to data/dataset/ (Inherit BaseDataset module)
1. Edit data/dataset/\_\_init\_\_.py (Import module and add module to SUPPORTED_DATASET)
1. Add config yaml file to configs/project/data/dataset/

### Add sampler
1. Add module to data/sampler/ (Inherit BaseSampler module)
1. Edit data/sampler/\_\_init\_\_.py (Import module and add module to SUPPORTED_SAMPLER)
1. Add config yaml file to configs/project/data/sampler/

### Add model
1. Add module to models/networks/ (Inherit BaseModel module)
1. Edit models/\_\_init\_\_.py (Import module and add module to SUPPORTED_MODEL)
1. Add config yaml file to configs/project/model/

### Add optimizer
1. Edit trainers/optimizer/\_\_init\_\_.py (Add module to SUPPORTED_OPTIMIZER)
1. Add config yaml file to configs/project/train/optimizer

### Add criterion
1. Edit trainers/criterion/\_\_init\_\_.py (Add module to SUPPORTED_CRITERION)
1. Add config yaml file to configs/project/train/criterion

### Add metrics
1. Add module to trainers/metrics/ (Inherit BaseMetrics module)
1. Edit trainers/metrics/\_\_init\_\_.py (Import module and add module to SUPPORTED_METRICS)
1. Add config yaml file to configs/project/train/metrics

### Add trainer
1. Add module to trainers/ (Inherit BaseTrainer module)
1. Edit trainers/\_\_init\_\_.py (Import module and add module to SUPPORTED_TRAINER)
1. Add config yaml file to configs/project/train/trainer

## Structure
```bash
$ tree -I "datasets|mlruns|__pycache__|outputs|multirun"
.
├── README.md
├── configs
│   └── project
│       ├── data
│       │   ├── dataset
│       │   │   └── cifar10.yaml
│       │   └── sampler
│       │       ├── balanced_batch_sampler.yaml
│       │       └── shuffle_sampler.yaml
│       ├── default.yaml
│       ├── hydra
│       │   └── job_logging
│       │       └── custom.yaml
│       ├── model
│       │   ├── resnet18.yaml
│       │   └── simple_cnn.yaml
│       └── train
│           ├── criterion
│           │   └── cross_entropy.yaml
│           ├── metrics
│           │   ├── classification.yaml
│           │   └── default.yaml
│           ├── optimizer
│           │   └── adam.yaml
│           └── trainer
│               └── default.yaml
├── data
│   ├── __init__.py
│   ├── dataloader
│   │   └── __init__.py
│   ├── dataset
│   │   ├── __init__.py
│   │   ├── base_dataset.py
│   │   ├── cifar10.py
│   │   └── helper.py
│   └── sampler
│       ├── __init__.py
│       ├── balanced_batch_sampler.py
│       ├── base_sampler.py
│       └── shuffle_sampler.py
├── docker
│   ├── Dockerfile
│   ├── build.sh
│   ├── env.sh
│   ├── env_dev.sh
│   ├── exec.sh
│   ├── init.sh
│   ├── requirements.txt
│   └── run.sh
├── models
│   ├── __init__.py
│   ├── base_model.py
│   └── networks
│       ├── resnet18.py
│       └── simple_cnn.py
├── train.py
└── trainers
    ├── __init__.py
    ├── base_trainer.py
    ├── criterion
    │   └── __init__.py
    ├── default_trainer.py
    ├── metrics
    │   ├── __init__.py
    │   ├── base_metrics.py
    │   ├── classification_metrics.py
    │   └── default_metrics.py
    └── optimizer
        └── __init__.py
```

## TODO
- [ ] optuna
- [ ] scheduler
- [ ] flake8
- [ ] error handling
- [ ] clear cache command
- [ ] assertion
- [ ] notification
- [ ] FP16 (apex)
- [ ] classmethod, staticmethod
- [ ] value error
- [ ] usage as template
- [ ] multi-gpu
- [ ] nohup
- [ ] docker-compose
- [ ] pytorch-lightning
