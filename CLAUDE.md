# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLMEvalKit is an open-source evaluation toolkit for Large Vision-Language Models (LVLMs). It supports 70+ benchmarks and 200+ models with generation-based evaluation using exact matching and LLM-based answer extraction.

## Key Commands

### Installation
```bash
pip install -e .
```

### Evaluation
```bash
# Basic evaluation
python run.py --data MMBench_DEV_EN MME --model idefics_9b_instruct

# Multi-GPU evaluation (data parallel)
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN --model idefics_9b_instruct

# Inference only (skip evaluation)
python run.py --data MMBench_DEV_EN --model idefics_9b_instruct --mode infer

# Using config file for flexible settings
python run.py --config path/to/config.json
```

### Model Utilities
```bash
# Check if a model is properly configured
vlmutil check {MODEL_NAME}

# List all supported models
vlmutil mlist all

# List all supported datasets
vlmutil dlist all
```

### Linting
```bash
pre-commit run --all-files
```

## Architecture

### Core Structure
- `run.py` - Main entry point for evaluation
- `vlmeval/config.py` - Model configurations (`supported_VLM` dict)
- `vlmeval/vlm/` - VLM model implementations (must implement `generate_inner()`)
- `vlmeval/api/` - API-based model implementations
- `vlmeval/dataset/` - Dataset implementations
- `vlmeval/dataset/video_dataset_config.py` - Video dataset configurations

### Model Implementation Pattern
All models in `vlmeval/vlm/` must implement:
```python
def generate_inner(self, msgs, dataset=None):
    """
    msgs: List[dict] with keys 'type' ('image'/'text') and 'value' (path/URL or text)
    Returns: str (model prediction)
    """
```

Optional methods for custom prompts:
- `use_custom_prompt(dataset)` - Returns bool indicating if custom prompt should be used
- `build_prompt(line, dataset)` - Builds custom multimodal message for the dataset
- `chat_inner(message, dataset)` - For multi-turn chat support

### Dataset Implementation Pattern
Datasets in `vlmeval/dataset/` must implement:
```python
def build_prompt(self, line) -> List[dict]:
    """
    line: int (sample index) or pd.Series (raw record)
    Returns: Multi-modal message list [dict(type='image', value=PATH), dict(type='text', value=prompt)]
    """

def evaluate(self, eval_file, **judge_kwargs) -> dict | pd.DataFrame:
    """
    eval_file: Path to prediction file (.xlsx)
    Returns: Evaluation metrics
    """
```

### Data Format
Datasets are TSV files with mandatory fields:
- `index` - Unique integer identifier
- `image` - Base64 encoded image
- `question` - Question text
- `answer` - Ground truth answer

## Environment Variables

Create a `.env` file in the project root for API keys:
```
OPENAI_API_KEY=
GOOGLE_API_KEY=
DASHSCOPE_API_KEY=
```

Other important variables:
- `LMUData` - Custom data path (default: `$HOME/LMUData`)
- `SPLIT_THINK=True` - Enable thinking mode parsing for models with `<think/>` tags
- `PRED_FORMAT=tsv` - Use TSV format for long responses (>16k/32k tokens)
- `VLMEVALKIT_USE_MODELSCOPE` - Use ModelScope for video benchmarks

## Transformers Version Requirements

Different VLMs require specific transformers versions. Check README.md for detailed compatibility matrix. Key examples:
- `transformers==4.37.0` for LLaVA, InternVL, CogVLM series
- `transformers==4.40.0` for IDEFICS2, MiniCPM-Llama3-V2.5
- `transformers==latest` for LLaVA-Next, GLM-4v-9B, Llama-3.2 series

## Code Style

- Max line length: 120 characters
- Linter: flake8 (ignores F401, F403, F405, E402, E722, E741, W503, E231, E702)
- Formatter: YAPF with `column_limit=120`

## Important Patterns

### Multi-modal Message Format
```python
# Image + text message
msg = [
    dict(type='image', value='/path/to/image.jpg'),
    dict(type='text', value='What is in this image?')
]

# Multiple images
msg = [
    dict(type='image', value='image1.jpg'),
    dict(type='image', value='image2.jpg'),
    dict(type='text', value='Compare these images.')
]
```

### Distributed Inference
- Use `torchrun` for data parallel inference on small models
- Use `python` with `CUDA_VISIBLE_DEVICES` for large models requiring multiple GPUs
- VLLM backend not compatible with torchrun; use `python` only

### Judge Model Selection
The evaluation system automatically selects judge models based on dataset type:
- MCQ/Y/N datasets: `chatgpt-0125` (default)
- MMVet, LLaVABench: `gpt-4-turbo`
- MathVista, MathVerse: `gpt-4o-mini`
- VGRPBench, MMDU: `gpt-4o`
