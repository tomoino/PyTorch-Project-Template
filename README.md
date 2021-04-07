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

## How to customize

## Structure
```bash
$ tree -I "datasets|mlruns|__pycache__|outputs|multirun"
.
├── README.md
├── configs
│   ├── config.yaml
│   ├── hydra
│   │   └── job_logging
│   │       └── custom.yaml
│   ├── project
│   │   └── default.yaml
│   └── supported_info.py
├── data
│   ├── __init__.py
│   ├── dataloader.py
│   ├── dataset
│   │   ├── cifar10.py
│   │   └── omniglot.py
│   ├── helper.py
│   └── sampler
│       └── balanced_batch_sampler.py
├── docker
│   ├── Dockerfile
│   ├── build.sh
│   ├── env.sh
│   ├── exec.sh
│   ├── init.sh
│   ├── requirements.txt
│   └── run.sh
├── metrics
│   ├── __init__.py
│   └── classification_metric.py
├── models
│   ├── __init__.py
│   ├── base_model.py
│   ├── helper.py
│   └── networks
│       ├── resnet18.py
│       └── simple_cnn.py
├── train.py
└── trainers
    ├── __init__.py
    ├── base_trainer.py
    └── default_trainer.py
```

## TODO
- [ ] error handling
- [ ] clear cache command
- [ ] nohup
- [ ] optuna
- [ ] basic metrics
- [ ] flake8
- [ ] notification
- [ ] FP16 (apex)
- [ ] classmethod, staticmethod
- [ ] value error
- [ ] usage as template
- [ ] pytorch-lightning
- [ ] multi-gpu
- [ ] refactoring on cfg to make the modules easy to reuse.
- [ ] utils.paths
- [ ] docker-compose
- [x] trainer
- [x] evaluation mode
- [x] logger
- [x] metrics
- [x] mlflow
- [x] hydra tab completion
- [x] projects
- [x] hydra
