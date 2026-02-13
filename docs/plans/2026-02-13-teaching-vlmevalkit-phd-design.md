# Teaching VLMEvalKit to a PhD Student - Design Document

**Date:** 2026-02-13
**Author:** Claude Code
**Status:** Approved

---

## Overview

This document outlines a 2-week curriculum for teaching VLMEvalKit to a PhD student working on crime detection and surveillance VLM development.

## Student Profile

| Attribute | Description |
|-----------|-------------|
| **Goals** | Use, extend, and deeply understand VLMEvalKit |
| **Background** | Strong ML/DL (PyTorch, transformers), new to VLMs |
| **Timeline** | 1-2 weeks |
| **Learning Style** | Theory first, then hands-on practice |
| **Domain** | Building VLMs for crime detection and surveillance |

## Learning Objectives

By the end of this curriculum, the PhD student will be able to:

1. **Explain VLM evaluation paradigms** - Generation-based vs. PPL-based, LLM judges vs. exact matching, and when to use each
2. **Navigate VLMEvalKit architecture** - Trace the complete flow from `run.py` → `config.py` → model/dataset instantiation → inference → evaluation
3. **Integrate a new VLM** - Implement `generate_inner()` and optional custom prompts for their surveillance model
4. **Create custom benchmarks** - Build a surveillance-specific dataset with proper TSV format, `build_prompt()`, and `evaluate()` methods
5. **Run distributed evaluations** - Use `torchrun`, LMDeploy, and VLLM for large-scale evaluation
6. **Debug evaluation issues** - Identify when models/datasets aren't interacting correctly and trace the problem

**Success criterion:** They can independently integrate their surveillance VLM, create a crime-detection benchmark, and produce evaluation results.

---

## Week 1: Foundations

### Day 1-2: VLM Fundamentals & Evaluation Paradigms

**Theory Sessions:**
- **VLM landscape overview**: Architecture patterns (encoder-decoder, decoder-only with vision encoder), tokenization strategies for images
- **Evaluation paradigms**:
  - Generation-based (VLMEvalKit's approach) vs. PPL-based (SEEDBench original)
  - Why generation-based generalizes better across model architectures
- **Answer extraction methods**:
  - Exact matching (for MCQ, Y/N tasks)
  - LLM-based extraction (GPT-4, local models via LMDeploy)
  - When each is appropriate
- **Key papers to read**: VLMEvalKit paper (arXiv:2407.11691), MMBench paper

**Hands-on (minimal, for intuition):**
```bash
# Run a simple evaluation to see the output format
python run.py --data MMBench_DEV_EN --model qwen_chat --verbose
# Examine the .xlsx output files
```

### Day 3-4: VLMEvalKit Architecture Deep-Dive

**Theory Sessions:**
- **Entry point**: `run.py` - argument parsing, config loading, distributed setup
- **Configuration system**: `vlmeval/config.py` - how models are registered
- **Model abstraction**:
  - Base class in `vlmeval/vlm/base.py`
  - `generate_inner(msgs, dataset)` contract
  - Multi-modal message format: `[dict(type='image', value=...), dict(type='text', value=...)]`
- **Dataset abstraction**:
  - `ImageBaseDataset` hierarchy
  - `build_prompt(line)` → multimodal messages
  - `evaluate(eval_file, **judge_kwargs)` → metrics
- **Inference pipeline**: `vlmeval/inference.py` - how models and datasets interact

**Code Reading Assignments:**
- `run.py:main()` - full flow
- `vlmeval/vlm/internvl_chat.py` - example model with custom prompts
- `vlmeval/dataset/image_mcq.py` - example dataset with evaluation

### Day 5: Hands-On Evaluation

**Practical Session:**
```bash
# 1. Run evaluation on multiple benchmarks
python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_9b_instruct

# 2. Multi-GPU evaluation
torchrun --nproc-per-node=4 run.py --data MME --model qwen_chat

# 3. Examine outputs:
#    - {model}_{dataset}.xlsx (predictions + ground truth)
#    - {model}_{dataset}_{judge}.xlsx (evaluation results)
```

**End-of-week deliverable:** Write a 1-page summary of VLMEvalKit's architecture in their own words.

---

## Week 2: Application to Surveillance Domain

### Day 1-2: Model Integration Pattern

**Theory Session:**
- **Model registration in `config.py`**: How to add a new model entry
- **The `generate_inner()` contract**:
  - Input: `msgs` (multi-modal message list), optional `dataset` for strategy switching
  - Output: string prediction
  - Handling images: local paths vs URLs vs base64
- **Optional methods**:
  - `use_custom_prompt(dataset)` - when to use custom formatting
  - `build_prompt(line, dataset)` - custom prompt construction
  - `INTERLEAVE = True/False` - multi-image support flag
- **Integration with LMDeploy/VLLM**: Using `use_lmdeploy` or `use_vllm` flags for faster inference

**Practical Session:**
```python
# Exercise: Wrap a simple model (e.g., a CLIP-based classifier)
# File: vlmeval/vlm/surveillance_vlm.py

class SurveillanceVLM(BaseVLM):
    INTERLEAVE = False  # Single image at a time

    def __init__(self, model_path, **kwargs):
        # Load their surveillance model
        pass

    def generate_inner(self, msgs, dataset=None):
        # Extract image and text from msgs
        # Run inference
        # Return prediction string
        pass
```

**Assignment:** Create a placeholder implementation for their surveillance VLM structure.

### Day 3-4: Surveillance Benchmark Creation

**Theory Session:**
- **TSV format requirements**:
  - Mandatory: `index`, `image` (base64), `question`, `answer`
  - Optional: `category`, `hint`, multi-choice options
- **Dataset class structure**:
  - Inherit from `ImageBaseDataset` or `ImageMCQDataset`
  - Implement `build_prompt()` for surveillance-specific formatting
  - Implement `evaluate()` for domain-specific metrics
- **Evaluation strategies for surveillance**:
  - Binary classification (crime detected: Y/N)
  - Multi-class (crime type classification)
  - Open-ended description (LLM judge required)

**Practical Session:**
```python
# Exercise: Create surveillance benchmark
# File: vlmeval/dataset/surveillance_bench.py

class SurveillanceBench(ImageMCQDataset):
    TYPE = 'MCQ'  # or 'Y/N' for binary

    def build_prompt(self, line):
        # Format surveillance-specific prompts
        # e.g., "Analyze this surveillance frame. Is suspicious activity present?"
        pass

    def evaluate(self, eval_file, **judge_kwargs):
        # Calculate accuracy, precision, recall for crime detection
        pass
```

**Assignment:** Design the TSV schema and dataset class for their surveillance benchmark.

### Day 5: End-to-End Integration

**Full Pipeline Session:**
```bash
# 1. Register surveillance VLM in config.py
# 2. Place surveillance benchmark TSV in $LMUData
# 3. Run evaluation
python run.py --data SurveillanceBench --model SurveillanceVLM --verbose

# 4. Analyze results
# 5. Iterate on prompts/evaluation
```

**End-of-week deliverable:** Working evaluation of their surveillance VLM on a small test set (10-20 samples) with documented results.

---

## Resources & Materials

### Reading Materials

**Essential Papers (Day 1-2):**
- VLMEvalKit Paper (arXiv:2407.11691) - Toolkit philosophy and design
- MMBench Paper - MCQ evaluation methodology
- MMMU Paper - Multi-modal multi-disciplinary evaluation

**Documentation:**
- `docs/en/Quickstart.md` - Getting started guide
- `docs/en/Development.md` - Model and benchmark implementation guide
- `CLAUDE.md` - Architecture reference

### Code Reference Files

**Model Examples:**
- `vlmeval/vlm/base.py` - Base class and contracts
- `vlmeval/vlm/internvl_chat.py` - Full-featured model with custom prompts
- `vlmeval/vlm/qwen2_vl.py` - Multi-image interleaved input handling
- `vlmeval/api/gpt4v.py` - API-based model pattern

**Dataset Examples:**
- `vlmeval/dataset/image_mcq.py` - Multi-choice QA benchmark
- `vlmeval/dataset/image_base.py` - Base dataset class
- `vlmeval/dataset/mmvet.py` - Open-ended evaluation with GPT judge

### Tools & Utilities

```bash
# Model listing
vlmutil mlist all

# Dataset listing
vlmutil dlist all

# Model health check
vlmutil check {MODEL_NAME}

# Image encoding/decoding utilities
from vlmeval.smp.vlm import encode_image_to_base64, decode_base64_to_image
```

### Environment Setup

```bash
# .env file for API keys (if using LLM judges)
OPENAI_API_KEY=...
GOOGLE_API_KEY=...

# Data directory
export LMUData=/path/to/surveillance/data
```

---

## Assessment & Milestones

### Checkpoint Questions (End of Each Day)

**Week 1:**
- Day 2: "Explain why VLMEvalKit uses generation-based evaluation instead of PPL-based."
- Day 4: "Trace the code path from `run.py` to when a model generates a prediction."
- Day 5: "What files are produced by an evaluation run and what does each contain?"

**Week 2:**
- Day 2: "What must a model implement to be compatible with VLMEvalKit?"
- Day 4: "What fields are mandatory in a benchmark TSV file and why?"
- Day 5: "How would you debug if your model returns empty predictions?"

### Milestone Deliverables

| Day | Deliverable | Validates |
|-----|-------------|-----------|
| W1-D5 | Architecture summary (1 page) | Understanding of system design |
| W2-D2 | Placeholder model implementation | `generate_inner()` contract |
| W2-D4 | Benchmark TSV schema + dataset class outline | Dataset integration pattern |
| W2-D5 | Working evaluation on test set | End-to-end capability |

### Self-Assessment Rubric (End of Week 2)

**Can they independently:**
- [ ] Add a new model to `config.py` and run it?
- [ ] Create a TSV benchmark file with correct format?
- [ ] Implement `build_prompt()` for their domain?
- [ ] Implement `evaluate()` with appropriate metrics?
- [ ] Debug a failed evaluation run?

### Support Channels

- **VLMEvalKit Discord**: https://discord.gg/evDT4GZmxN
- **GitHub Issues**: For bug reports and feature questions
- **Code examples**: Reference PRs listed in `docs/en/Development.md`

---

## Next Steps

After this curriculum is complete, the student should be able to:
1. Integrate their surveillance VLM into VLMEvalKit
2. Create domain-specific benchmarks for crime detection evaluation
3. Run and interpret evaluations
4. Contribute back to VLMEvalKit if desired
