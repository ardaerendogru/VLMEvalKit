# VLMEvalKit - Agent Guidelines

This document provides essential information for agentic coding agents working in the VLMEvalKit repository.

## Project Overview

VLMEvalKit is an open-source evaluation toolkit for Vision-Language Models (VLMs). It supports evaluating various multimodal models on multiple benchmarks.

## Build / Install Commands

```bash
# Install in editable mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Lint Commands

```bash
# Install pre-commit hooks (recommended first-time setup)
pip install pre-commit
pre-commit install

# Run linting on all files
pre-commit run --all-files

# Run specific linters
flake8 . --max-line-length=120 --ignore=F401,F403,F405,E402,E722,E741,W503,E231,E702

# Format code with YAPF
yapf --style="{column_limit=120}" -i <file>
```

### Linting Configuration

- **Formatter**: YAPF (not Black)
- **Max line length**: 120 characters
- **Flake8 ignore rules**: F401, F403, F405, E402, E722, E741, W503, E231, E702
- **Line endings**: LF (Unix style)
- **Excluded paths**: `scripts/`, `assets/`, `vlmeval/config.py`, `vlmeval/dataset/utils/`

## Test / Evaluation Commands

```bash
# Run evaluation on specific datasets with a model
python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model <model_name> --verbose

# Run with distributed inference (multi-GPU)
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN --model <model_name> --verbose

# Inference only (skip evaluation)
python run.py --data MMBench_DEV_EN MME --model <model_name> --verbose --mode infer

# Evaluation only (skip inference)
python run.py --data MMBench_DEV_EN --model <model_name> --mode eval

# Run with config file
python run.py --config path/to/config.json

# List supported models
vlmutil mlist all

# List supported datasets
vlmutil dlist all
```

### Common Test Arguments

| Argument | Description |
|----------|-------------|
| `--data` | Dataset names (space-separated) |
| `--model` | Model names (space-separated) |
| `--config` | Path to config JSON file |
| `--work-dir` | Output directory (default: `./outputs`) |
| `--mode` | `all`, `infer`, or `eval` |
| `--verbose` | Enable verbose logging |
| `--reuse` | Reuse existing prediction files |
| `--api-nproc` | Parallel API calls (default: 4) |
| `--judge` | Judge model for evaluation |

## Code Style Guidelines

### Imports

```python
# Standard library first (alphabetically sorted)
import copy as cp
import os
import os.path as osp
import warnings
from abc import abstractmethod
from pathlib import Path

# Third-party imports
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports (relative imports within package)
from ..smp import *
from ..dataset import img_root_map, DATASET_TYPE
from .base import BaseModel
```

**Notes:**
- Wildcard imports (`from ..smp import *`) are common in this codebase
- Use `# flake8: noqa: F401, F403` to suppress linting for wildcard imports if needed

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `BaseModel`, `ImageBaseDataset`, `QwenVL` |
| Functions/Methods | snake_case | `generate_inner`, `build_prompt`, `use_custom_prompt` |
| Variables | snake_case | `model_name`, `dataset_type` |
| Constants | UPPER_SNAKE_CASE | `DATASET_URL`, `DATASET_MD5`, `INTERLEAVE` |
| Private methods | Leading underscore | `_minimal_ext_cmd` |
| Class attributes (constants) | UPPER_SNAKE_CASE | `INTERLEAVE = False` |

### Type Hints

Type hints are **minimally used** in this codebase. When adding type hints, follow standard Python conventions. Docstrings are preferred for documenting function signatures:

```python
def generate(self, message, dataset=None):
    """Generate the output message.

    Args:
        message (list[dict]): The input message.
        dataset (str, optional): The name of the dataset. Defaults to None.

    Returns:
        str: The generated message.
    """
```

### Error Handling

```python
# Try-except with logging
try:
    # operation
except Exception as e:
    logging.info(f'{type(e)}: {e}')
    # fallback or re-raise

# Assertions for validation
assert model_path is not None
assert isinstance(line, pd.Series) or isinstance(line, dict)

# Warnings for non-critical issues
warnings.warn(f'Model {model_name} does not support interleaved input.')
```

### Docstring Style

Use Google-style docstrings:

```python
def build_prompt(self, line, dataset):
    """Build custom prompts for a specific dataset.

    Args:
        line (line of pd.DataFrame): The raw input line.
        dataset (str): The name of the dataset.

    Returns:
        str: The built message.
    """
```

## Project Structure

```
VLMEvalKit/
├── run.py                    # Main entry point
├── setup.py                  # Package setup
├── requirements.txt          # Dependencies
├── .pre-commit-config.yaml   # Linting config
├── vlmeval/
│   ├── __init__.py
│   ├── config.py             # Model configurations
│   ├── api/                  # API-based models (GPT4V, Claude, etc.)
│   ├── vlm/                  # Vision-language model implementations
│   │   ├── base.py          # BaseModel class
│   │   └── *.py             # Individual model files
│   ├── dataset/              # Dataset implementations
│   │   ├── image_base.py    # ImageBaseDataset class
│   │   └── *.py             # Individual datasets
│   ├── smp/                  # Utility functions (misc, file, vlm, log)
│   ├── inference.py          # Inference pipeline
│   └── tools/                # CLI tools
└── docs/
    ├── en/                   # English documentation
    └── zh-CN/                # Chinese documentation
```

## Adding New Models

1. Create a new file in `vlmeval/vlm/`
2. Inherit from `BaseModel` in `vlmeval/vlm/base.py`
3. **Required**: Implement `generate_inner(self, message, dataset=None)`
4. **Optional**: Implement `use_custom_prompt(dataset)` and `build_prompt(line, dataset)`
5. Register model in `vlmeval/config.py` under `supported_VLM`

### Minimal Model Implementation

```python
from ..smp import *
from .base import BaseModel

class MyVLM(BaseModel):
    INTERLEAVE = False  # Set True if model supports interleaved images/text
    
    def __init__(self, model_path, **kwargs):
        super().__init__()
        self.model = load_model(model_path)
    
    def generate_inner(self, message, dataset=None):
        # message is list[dict] with keys 'type' and 'value'
        # type: 'text', 'image', or 'video'
        # value: content string or path
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        return self.model.generate(prompt, image_path)
```

## Adding New Datasets

1. Create dataset class inheriting from `ImageBaseDataset` in `vlmeval/dataset/image_base.py`
2. **Required**: Implement `evaluate(self, eval_file, **judge_kwargs)`
3. Set `TYPE`, `DATASET_URL`, and `DATASET_MD5` class attributes
4. Register in `vlmeval/dataset/__init__.py`

## Multimodal Message Format

```python
# Standard format
message = [
    dict(type='image', value='/path/to/image.jpg'),
    dict(type='text', value='What is in this image?')
]

# Interleaved format (multiple images)
message = [
    dict(type='image', value='/path/to/image1.jpg'),
    dict(type='image', value='/path/to/image2.jpg'),
    dict(type='text', value='Compare these images.')
]
```

## Key Files Reference

| Purpose | File Path |
|---------|-----------|
| Main entry point | `run.py` |
| Model base class | `vlmeval/vlm/base.py` |
| Dataset base class | `vlmeval/dataset/image_base.py` |
| Model configurations | `vlmeval/config.py` |
| Utilities | `vlmeval/smp/` |
| Development guide | `docs/en/Development.md` |

## Pre-commit Checklist

Before submitting PRs:

```bash
# Run linting
pre-commit run --all-files

# Ensure code is formatted
yapf --style="{column_limit=120}" -ri vlmeval/

# Test your changes
python run.py --data MMBench_DEV_EN --model <your_model> --verbose
```
