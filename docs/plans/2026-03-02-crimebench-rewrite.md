# CrimeBench Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite CrimeBench dataset implementation to use new data location and format.

**Architecture:** Update the existing `crimebench.py` to read from the new `/home/ubuntu/data/CrimeBench` location with `crimebench_v2_reduced.parquet`. Keep TSV caching, fix video path handling for subdirectories.

**Tech Stack:** Python, PyArrow, Pandas, VLMEvalKit VideoBaseDataset

---

## Task 1: Update Configuration and Docstring

**Files:**
- Modify: `vlmeval/dataset/crimebench.py:1-110`

**Step 1: Update docstring with new statistics**

Replace lines 16-22:
```python
Data Statistics:
- Total QA pairs: 4,926
- Unique videos: 3,226
- Tiers: 1, 2, 3, 5
- Question types: 18
- Crime types: 30
```

**Step 2: Update DATASET_PATH constant**

Replace line 109:
```python
DATASET_PATH = "/home/ubuntu/data/CrimeBench"
```

**Step 3: Commit**

```bash
git add vlmeval/dataset/crimebench.py
git commit -m "feat(crimebench): update docstring and data path"
```

---

## Task 2: Update prepare_dataset Method

**Files:**
- Modify: `vlmeval/dataset/crimebench.py:136-186`

**Step 1: Update parquet directory name**

Replace line 160:
```python
parquet_dir = os.path.join(data_path, "crimebench_v2_reduced.parquet")
```

**Step 2: Update error message**

Replace lines 163-167:
```python
if not os.path.exists(arrow_file):
    raise FileNotFoundError(
        f"Arrow file not found: {arrow_file}. "
        f"Please ensure crimebench_v2_reduced.parquet directory exists."
    )
```

**Step 3: Commit**

```bash
git add vlmeval/dataset/crimebench.py
git commit -m "feat(crimebench): update parquet directory name"
```

---

## Task 3: Fix Video Path Handling in TSV Generation

**Files:**
- Modify: `vlmeval/dataset/crimebench.py:195-251`

**Step 1: Update _generate_tsv_from_arrow to store full video path**

Replace lines 222-235:
```python
tsv_row = {
    "index": idx,
    "video": os.path.splitext(video_path)[0],
    "video_path": video_path,
    "question": row["question"],
    "A": candidates[0] if len(candidates) > 0 else "",
    "B": candidates[1] if len(candidates) > 1 else "",
    "C": candidates[2] if len(candidates) > 2 else "",
    "D": candidates[3] if len(candidates) > 3 else "",
    "answer": answer_letter,
    "tier": row["tier"],
    "question_type": row["question_type"],
    "crime_type": row["crime_type"],
}
```

Note: The `video` column now contains paths like `ucf-crime/events/test/Arrest024_x264_E0` (with subdirectories).

**Step 2: Commit**

```bash
git add vlmeval/dataset/crimebench.py
git commit -m "feat(crimebench): store full video path with subdirectories"
```

---

## Task 4: Update build_prompt for Video Path Resolution

**Files:**
- Modify: `vlmeval/dataset/crimebench.py:252-289`

**Step 1: Update video path resolution in build_prompt**

Replace lines 268-272:
```python
if video_llm:
    message = [dict(type="text", value=self.FRAMES_TMPL_SYS_4VIDEO_LLM.strip())]
    video_path = os.path.normpath(
        os.path.join(self.data_root, line["video_path"])
    )
    message.append(dict(type="video", value=video_path))
```

The key change is using `line["video_path"]` instead of constructing from `line["video"]`.

**Step 2: Commit**

```bash
git add vlmeval/dataset/crimebench.py
git commit -m "feat(crimebench): use video_path column for path resolution"
```

---

## Task 5: Verify Implementation

**Step 1: Test dataset loading**

```bash
python -c "
from vlmeval.dataset import CrimeBench
ds = CrimeBench(dataset='CrimeBench', nframe=8)
print(f'Dataset loaded: {len(ds)} samples')
print(f'First sample: {ds[0]}')
"
```

Expected: Dataset loads successfully, prints sample count and first sample.

**Step 2: Test evaluation**

```bash
python -c "
from vlmeval.dataset import CrimeBench
# Create a minimal test file
import pandas as pd
data = pd.DataFrame({
    'index': [0],
    'video': ['ucf-crime/events/test/Arrest024_x264_E0'],
    'video_path': ['ucf-crime/events/test/Arrest024_x264_E0.mp4'],
    'question': ['Test?'],
    'A': ['A'], 'B': ['B'], 'C': ['C'], 'D': ['D'],
    'answer': ['A'],
    'prediction': ['A'],
    'tier': [1],
    'question_type': ['crime_classification'],
    'crime_type': ['Arrest']
})
data.to_csv('/tmp/test_crimebench.tsv', sep='\t', index=False)
results = CrimeBench.evaluate('/tmp/test_crimebench.tsv')
print(f'Evaluation results: {results}')
"
```

Expected: Evaluation runs and returns results dict.

**Step 3: Final commit if needed**

```bash
git add vlmeval/dataset/crimebench.py
git commit -m "feat(crimebench): complete rewrite for new data location"
```

---

## Summary

- Update docstring and `DATASET_PATH`
- Update parquet directory name to `crimebench_v2_reduced.parquet`
- Fix video path handling to include subdirectories
- Update `build_prompt` to use `video_path` column
- Verify loading and evaluation work
