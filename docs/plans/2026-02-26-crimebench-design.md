# CrimeBench Implementation Design

**Goal:** Add CrimeBench video QA benchmark to VLMEvalKit for evaluating VLMs on crime/violence understanding.

**Date:** 2026-02-26

---

## Data Summary

| Attribute | Value |
|-----------|-------|
| Total QA pairs | 5,625 |
| Unique videos | 481 |
| Tiers | 1, 2, 3, 5 |
| Question types | 18 |
| Crime types | 19 |
| Answer format | 4-choice MCQ (index 0-3) |
| Video source | XD-Violence |

## Schema

```
id: string (unique identifier)
video_path: string (e.g., "Burglary033_x264.mp4")
video_source: string (XD-Violence)
tier: int (1, 2, 3, 5)
question_type: string (18 types)
question: string
candidates: list[string] (4 options)
answer: int (0-3)
ground_truth: {contextual: string, crime_type: string}
certificate_length: float
crime_type: string (19 categories)
source_description: string
has_desc: bool
```

## Question Types (18)

- entity_recognition, crime_classification, trajectory_forecasting
- intent_prediction, counterfactual_adjacent, degraded_input
- counterfactual_long_chain, impossible_scenarios, anomaly_classification
- guardrail_consistency, rule_violation, spatial_grounding
- omission_causality, multi_event_causal, temporal_localization
- counting, action_sequence, negative_control

## Crime Types (19)

Normal, Explosion, Shooting, RoadAccidents, Shoplifting, Riot,
Fighting, Abuse, Burglary, Car accident, Arson, Arrest,
Robbery, Vandalism, Stealing, Assault, Violence,
Privacy violation, Physical altercation

---

## Architecture

### File Structure

```
vlmeval/dataset/crimebench.py    # New dataset class
vlmeval/dataset/__init__.py      # Add import + registration
```

### Class Design

```python
class CrimeBench(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    MODALITY = 'VIDEO'
    DATASET_PATH = '/home/ubuntu/data/crimebench/crimebench_export'
    
    def prepare_dataset(self, dataset_name):
        # Read Arrow IPC from parquet directory
        # Convert to TSV format
        # Return {root: video_dir, data_file: tsv_path}
    
    def build_prompt(self, line, video_llm=False):
        # Video LLM: return video path + question
        # Frame-based: extract frames + question
    
    def evaluate(cls, eval_file, **judge_kwargs):
        # Calculate accuracy with breakdowns:
        # - Overall
        # - Per-tier
        # - Per-crime-type
        # - Per-question-type
```

### Data Loading

1. Check `CRIMEBENCH_DATA_PATH` env var, fallback to default path
2. Read Arrow IPC stream from `crimebench_v2_final.parquet/`
3. Convert to TSV (cached, regenerate if arrow newer)
4. Map answer index (0-3) to letter (A-D)

### Evaluation Metrics

- **Overall accuracy**: (correct / valid) * 100
- **Per-tier accuracy**: breakdown by tiers 1, 2, 3, 5
- **Per-crime-type accuracy**: breakdown by 19 crime categories
- **Per-question-type accuracy**: breakdown by 18 question types

### Prompt Template

```
You will receive a video clip.
Based on the video, answer the following multiple-choice question.

Question: {question}

A. {candidates[0]}
B. {candidates[1]}
C. {candidates[2]}
D. {candidates[3]}

Answer with the option letter (A, B, C, or D) of the correct option.
```

---

## Registration

In `vlmeval/dataset/__init__.py`:

1. Add import: `from .crimebench import CrimeBench`
2. Add to `VIDEO_DATASET` list: `CrimeBench`

---

## Usage

```bash
# Run evaluation
python run.py --model Qwen2-VL-7B --benchmark CrimeBench --nframe 8

# With custom data path
CRIMEBENCH_DATA_PATH=/path/to/data python run.py --model Qwen2-VL-7B --benchmark CrimeBench
```

---

## Implementation Tasks

1. Create `crimebench.py` with `CrimeBench` class
2. Implement `prepare_dataset()` to read Arrow IPC
3. Implement `build_prompt()` for video LLM input
4. Implement `evaluate()` with comprehensive metrics
5. Register in `__init__.py`
6. Test with sample evaluation
