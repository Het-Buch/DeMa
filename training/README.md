# Training

Experiments used Kaggle GPU T4 x2. The demo uses included weights. Use this guide to train new models.

## Files in this folder

| File | Purpose |
|---|---|
| `train_cql.py` | Trains CQL, tests it in the simulator, and saves its results. |
| `cql.py` | Loads recorded steps and runs the simulator tests used by CQL. |
| `requirements.txt` | Installs the packages needed for CQL training. |
| `README.md` | Explains the files, settings, and training commands. |

DT and DeMa use `train_compare.py` in the repository root.

**Conservative Q-Learning (CQL)** learns action values from recorded experience and lowers estimates for unfamiliar actions. It is tested on HalfCheetah.

## Setup

Enable Internet and select GPU T4 x2. Place the repository files in the working directory and run commands from the repository root.

For DT and DeMa, install:

```python
!pip install -r requirements.txt
```

For CQL, use a separate notebook and install:

```python
!pip install -r training/requirements.txt
```

CQL uses d3rlpy with a different Gymnasium version, so it needs a separate environment.

## Train DT or DeMa

Run this cell for one HalfCheetah DT training run. `%env` sets an option and `!python` starts the script:

```python
%env DATASET_NAME=mujoco/halfcheetah/medium-v0
%env MODELS=Decision Transformer
%env SEED=42
%env LEARNING_RATE=0.0004
%env DROPOUT=0.1
%env EVAL_TARGETS=12000
%env TOTAL_STEPS=100000
%env NUM_ROLLOUTS=100
!python train_compare.py
```

To train DeMa, change these three lines before running the command:

```python
%env MODELS=DeMa
%env LEARNING_RATE=0.00005
%env DROPOUT=0.3
```

Repeat each model with `SEED=43` and `SEED=44`. A seed controls random initialization and sampling. Three seeds produce three separately trained versions of each model.

| Setting | Meaning | Value |
|---|---|---|
| Training episodes | Complete recorded runs used for learning | 150 |
| Validation episodes | Separate recorded runs used to check action error | 50 |
| History length | Consecutive steps in one input sequence | 20 |
| Batch size | Input sequences processed in one update | 64 |
| Hidden width | Size of each model input vector after projection | 128 |
| Blocks | Repeated processing stages in the model | 3 |
| Training updates | Times the model weights are adjusted | 100,000 |
| Warmup updates | Initial updates that gradually raise the learning rate | 10,000 |
| Evaluation episodes | Simulator tests per trained model and reward target | 100 |

Learning rate controls the size of a training adjustment. Dropout temporarily switches off some features during training. The two models use settings selected in separate tuning experiments on HalfCheetah.

### AntMaze

Keep the model-specific learning rate and dropout. Change the dataset settings before the training command:

```python
%env DATASET_NAME=D4RL/antmaze/umaze-v1
%env MEASURED_RANDOM_RETURN=0
%env DATASET_MEAN_RETURN=438.2091
%env EVAL_TARGETS=538,438
```

The targets are the dataset's highest episode reward, 538, and rounded average, 438. A target guides the model. The simulator determines the reward it earns.

When switching back to HalfCheetah, restore:

```python
%env DATASET_NAME=mujoco/halfcheetah/medium-v0
%env MEASURED_RANDOM_RETURN=-295.60
%env DATASET_MEAN_RETURN=12089.21
%env EVAL_TARGETS=12000
```

## Train CQL

In the separate CQL notebook, run:

```python
%env DATASET_NAME=mujoco/halfcheetah/medium-v0
%env MEASURED_RANDOM_RETURN=-295.60
%env SEED=42
%env CQL_STEPS=100000
%env CQL_ROLLOUTS=100
!python training/train_cql.py
```

Repeat with seeds 43 and 44. CQL chooses actions using learned values and does not take a requested reward target.

## Design-change experiments

Start with the model's normal settings and change one option per experiment.

| Change | What it does | Setting |
|---|---|---|
| Grouped inputs | Puts return, state, and previous action into one vector per step | `TOKEN_LAYOUT=grouped` |
| DT pre-normalization | Normalizes inputs before each DT block | `NORM_FIRST=1` |
| DeMa pre-normalization | Normalizes inputs before each DeMa block | `DEMA_NORM_FIRST=1` |
| DeMa step embeddings | Adds a learned step-number vector | `DEMA_TIMESTEP=1` |
| Smaller step initialization | Starts those step vectors with smaller values | `DEMA_TIMESTEP=1`, `TIMESTEP_EMBED_STD=0.02` |
| DeMa with SER | Blends recorded actions with predictions from an earlier model snapshot | `SER=1` |

The normal settings are `TOKEN_LAYOUT=interleaved`, `NORM_FIRST=0`, `DEMA_NORM_FIRST=0`, `DT_TIMESTEP=0`, `DEMA_TIMESTEP=0`, and `SER=0`. Reset these before changing experiments.

The [model variants notebook](../02_SER_and_MultiGrainedSSM_vs_DeMa.ipynb) contains Multi-Grained SSM and SER experiments. The [ablation notebook](../03_Ablation_Studies.ipynb) contains comparisons of the design changes.

## Outputs

Sequence-model training writes weights to `checkpoints/` and experiment records to `results/`. CQL writes its summary to `results/cql/` and training logs to `d3rlpy_logs/`. Keep each run's output before starting another run, since checkpoint names can be reused.

For W&B tracking, load `WANDB_API_KEY` from Kaggle Secrets into the notebook environment before training. Keep the key out of saved code and outputs.
