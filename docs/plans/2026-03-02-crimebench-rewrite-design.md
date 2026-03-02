# CrimeBench Rewrite Design

## Overview

Rewrite the CrimeBench dataset implementation to use the new data location and updated dataset format.

## Requirements

- Use new data location: `/home/ubuntu/data/CrimeBench`
- Support environment variable override: `CRIMEBENCH_DATA_PATH`
- Read from `crimebench_v2_reduced.parquet` (Arrow IPC format)
- Keep TSV caching for performance
- Maintain same evaluation metrics

## Data Statistics

- Total questions: 4,926
- Unique videos: 3,226
- Tiers: 1, 2, 3, 5
- Question types: 18
- Crime types: 30

## Configuration

```python
DATASET_PATH = "/home/ubuntu/data/CrimeBench"
PARQUET_NAME = "crimebench_v2_reduced.parquet"

# Environment variable override
data_path = os.environ.get("CRIMEBENCH_DATA_PATH", self.DATASET_PATH)
```

## Data Loading

### Directory Structure

```
/home/ubuntu/data/CrimeBench/
├── crimebench_v2_reduced.parquet/
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
└── videos/
    ├── ucf-crime/
    │   ├── events/test/*.mp4
    │   └── videos/test/*.mp4
    └── xd-violence/
        └── *.mp4
```

### TSV Columns

| Column | Source | Notes |
|--------|--------|-------|
| `index` | row index | |
| `video` | `video_path` without extension | e.g., `ucf-crime/events/test/Arrest024_x264_E0` |
| `video_path` | direct copy | Full relative path from videos/ |
| `question` | direct copy | |
| `A`, `B`, `C`, `D` | `candidates[0-3]` | |
| `answer` | `chr(65 + answer)` | Convert 0→A, 1→B, etc. |
| `tier` | direct copy | |
| `question_type` | direct copy | |
| `crime_type` | direct copy | |

### Video Path Handling

Video paths now include subdirectories (e.g., `ucf-crime/events/test/Arrest024_x264_E0.mp4`).
Store full relative path in `video_path` column for proper resolution.

## Prompt Building

### For Video LLMs

```python
message = [
    dict(type="text", value="You will receive a video clip.\nBased on the video, answer the following multiple-choice question."),
    dict(type="video", value="/full/path/to/video.mp4"),
    dict(type="text", value="Question: ...\n\nA. ...\nB. ...\nC. ...\nD. ...\n\nAnswer with the option letter (A, B, C, or D) of the correct option.")
]
```

### For Frame-based Models

- Extract frames using `save_video_frames()` from VideoBaseDataset
- Same text structure but with multiple `type="image"` entries

## Evaluation

### Metrics

1. **Overall accuracy** - total correct / total valid
2. **Per-tier accuracy** - for tiers 1, 2, 3, 5
3. **Per-crime-type accuracy** - for all crime types in data
4. **Per-question-type accuracy** - for all 18 question types

### Scoring Logic

- Extract first A-D letter from prediction
- Compare with ground truth answer letter
- Failed predictions get score -1 (excluded from valid count)

## Dataset Registration

No changes needed to `video_dataset_config.py`. Existing variants work via `partial()`:

```python
"CrimeBench_8frame": partial(CrimeBench, dataset="CrimeBench", nframe=8),
"CrimeBench_16frame": partial(CrimeBench, dataset="CrimeBench", nframe=16),
"CrimeBench_32frame": partial(CrimeBench, dataset="CrimeBench", nframe=32),
"CrimeBench_64frame": partial(CrimeBench, dataset="CrimeBench", nframe=64),
"CrimeBench_1fps": partial(CrimeBench, dataset="CrimeBench", fps=1.0),
"CrimeBench_2fps": partial(CrimeBench, dataset="CrimeBench", fps=2.0),
```

## Files to Modify

1. `vlmeval/dataset/crimebench.py` - Main implementation
   - Update `DATASET_PATH`
   - Update `PARQUET_NAME`
   - Update docstring statistics
   - Fix video path handling in `_generate_tsv_from_arrow()`
