# PyTorch-Project-Template
## Installation
```bash
git clone git@github.com:tomoino/PyTorch-Project-Template.git
```

## Usage
### Setup
```bash
cd PyTorch-Project-Template
sh docker/build.sh
sh docker/run.sh
sh docker/exec.sh
```

### Training
```bash
python train.py
```

### Evaluation
```bash
python train.py eval=True
```

### Start a new project
1. Add yaml file to "./configs/project"
    ```bash
    vi ./configs/project/new_project.yaml
    ```
1. Run train.py with project option
    ```bash
    python train.py project=new_project
    ```

## Structure
```bash
.
├── README.md
├── configs
│   ├── default.yml
│   └── supported_info.py
├── data
│   ├── __init__.py
│   ├── dataloader.py
│   ├── dataset
│   │   ├── cifar10.py
│   │   ├── imagenet.py
│   │   └── omniglot.py
│   ├── helper.py
│   └── sampler
│       └── balanced_batch_sampler.py
├── docker
│   ├── Dockerfile
│   ├── build.sh
│   ├── env.sh
│   ├── exec.sh
│   ├── requirements.txt
│   └── run.sh
├── executor
│   └── __init__.py
├── metrics
│   ├── __init__.py
│   └── classification_metric.py
├── models
│   ├── __init__.py
│   ├── base_model.py
│   ├── helper.py
│   └── resnet18.py
├── train.py
└── utils
    ├── __init__.py
    └── load.py
```
## TODO
- [ ] logger (mlflow)
- [ ] optuna
- [ ] flake8
- [ ] FP16 (apex)
- [ ] classmethod, staticmethod
- [ ] value error
- [ ] usage as template
- [ ] pytorch-lightning
- [ ] multi-gpu
- [ ] refactoring on cfg to make the modules easy to reuse.
- [ ] utils.paths
- [ ] metric: confusion matrix
- [x] projects
- [x] hydra
