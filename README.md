# Controlling Browser Using Qwen Agent

A small reinforcement-learning project for training a Qwen language model to control a browser-like environment using GRPO.  
The goal is simple: turn browser state into text, let the model predict exactly one valid action, execute it in BrowserGym, and reward it based on task success.

This repo is built as a practical experiment in browser automation, language models, and reinforcement learning. It is also a base that I plan to extend with more training examples from MiniWoB environments.

## Want to test it out yourself

You can quickly try the trained model without running training.

1. Clone the repository


    git clone https://github.com/morty649/controlling-browser-using-qwen-agent.git
    cd controlling-browser-using-qwen-agent

2. Download the trained model

Model is available on Hugging Face:
https://huggingface.co/morty649/Qwen2-0.5B-Instruct-browsergym-20260405-121307

Run evaluation
    
    make evaluate config=qwen_evaluation_debug.yaml

This will:

load the trained model
run a few BrowserGym episodes
print prompts and model outputs
save screenshots in the media/ folder for inspection

Note: Make sure your environment is set up correctly (dependencies + BrowserGym access) before running evaluation.



## Why this project exists

Browsers are one of the most common interfaces people use every day, but they are still hard to automate reliably with traditional scripts. This project explores a different approach:

- represent the page as text through the accessibility tree
- let the model decide the next action
- train it with reinforcement learning instead of only supervised data
- improve the model using real task outcomes as feedback

The idea is to move toward agents that can understand a page, reason over the current state, and take the next best step instead of relying only on fixed rules.

## Where this is useful

This setup is useful for:

- browser automation research
- agent training on structured web tasks
- reinforcement learning with language models
- benchmarking model behavior in interactive environments
- experimenting with task-oriented browser control
- building future agents that can handle repetitive browser workflows

It is especially useful when you want a model that learns from interaction, not just from static examples.

## How it works

The training loop follows a simple structure:

1. BrowserGym provides the current environment state.
2. The accessibility tree and task goal are converted into a text prompt.
3. Qwen generates one browser action, such as `click('13')`.
4. The action is executed inside BrowserGym.
5. The task result is converted into a reward.
6. GRPO updates the model based on that reward signal.

This repo focuses on a clean action format and a small, controlled training loop so the model learns browser behavior in a focused way.

## Main components

### `fine_tuning.py`
Main training entry point.  
Handles environment setup, prompt construction, GRPO training, reward assignment, and checkpoint saving.

### `evaluate.py`
Debug and evaluation script.  
Loads a trained model, runs BrowserGym episodes, prints prompts and outputs, and saves screenshots for inspection.

### `configuration.py`
Central configuration logic for the project.

### `configuration_files/`
YAML configuration files for training and evaluation settings.

### `modal_infra.py`
Remote training setup using Modal.  
This is used to run training on an external GPU provider without depending only on local hardware.

## Why Modal is used

Training and evaluation can be expensive, especially once the model size or number of examples increases. Modal gives this project a simple way to run compute-heavy jobs remotely on GPU machines.

Using Modal here helps with:

- access to external GPUs
- running training without local setup pain
- scaling experiments more easily
- keeping the local machine free
- making the workflow easier to reproduce in the cloud

In this repo, Modal is part of the practical training pipeline, not just an optional add-on.

## Planned extension

The current setup is only the starting point.  
I also plan to finetune the model on more examples from MiniWoB environments so the agent can learn a wider range of browser interactions and improve generalization across tasks.

## Project structure

```text
.
├── configuration_files/
├── media/
├── src/browser_control_using_grpo_qwen/
│   ├── fine_tuning.py
│   ├── evaluate.py
│   ├── configuration.py
│   └── modal_infra.py
├── Makefile
├── requirements.txt
├── pyproject.toml
└── README.md



Training

The project is configured to train Qwen with LoRA, using GRPO and BrowserGym.

Typical training flow:

make fine-tune

Or run the training script directly if you prefer.

The main training config currently uses:

Qwen/Qwen2-0.5B-Instruct
LoRA adapters
vLLM
Weights & Biases logging
a BrowserGym environment hosted remotely
Evaluation

To run evaluation and inspect model behavior:

make evaluate

The evaluation script prints prompts, model responses, and task progress.
It also saves screenshots into media/ so you can inspect what happened during an episode.

External dependencies

This project relies on BrowserGym and environment code that comes from installed dependencies, not only from the files in this repo.
That means the browser logic is partly provided externally, while this repository focuses on the training loop, config, and evaluation workflow.

Notes
The README is intentionally short and practical because the code is the real documentation.
The action space is kept strict on purpose so the model learns a clean browser-control pattern.
This is a research and experimentation repo, not a polished production agent.
I will keep extending it with more MiniWoB training data and better evaluation coverage.

Status

Active experiment, still evolving.