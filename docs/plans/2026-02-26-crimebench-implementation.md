# CrimeBench Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CrimeBench video QA benchmark to VLMEvalKit for evaluating VLMs on crime/violence scene understanding.

**Architecture:** Create `CrimeBench` class extending `VideoBaseDataset` that reads Arrow IPC data from the parquet directory, converts to TSV format on first run, and provides comprehensive evaluation metrics (overall, per-tier, per-crime-type, per-question-type accuracy).

**Tech Stack:** Python, PyArrow, Pandas, VLMEvalKit dataset framework

---

## Task 1: Create CrimeBench Dataset Class Skeleton

**Files:**
- Create: `vlmeval/dataset/crimebench.py`

**Step 1: Create file with class skeleton and constants**

```python
# flake8: noqa
"""
CrimeBench Dataset Implementation for VLMEvalKit

CrimeBench is a video question answering benchmark for crime and violence scenes.
It features multiple reasoning tasks across different crime types and difficulty tiers.

*** MANUAL DOWNLOAD REQUIRED ***
Dataset must be downloaded manually.

Key Implementation Details:
- Reads Arrow IPC format from parquet directory
- Converts to TSV format on first run (cached)
- Supports video LLM input (primary) with frame extraction fallback
- Reports comprehensive metrics: overall, per-tier, per-crime-type, per-question-type

Data Statistics:
- Total QA pairs: 5,625
- Unique videos: 481
- Tiers: 1, 2, 3, 5
- Question types: 18
- Crime types: 19
"""

import os
import io
import re
import warnings
import zipfile
from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE

FAIL_MSG = 'Failed to obtain answer via API.'

# Question type names for display
QUESTION_TYPES = [
    'entity_recognition', 'crime_classification', 'trajectory_forecasting',
    'intent_prediction', 'counterfactual_adjacent', 'degraded_input',
    'counterfactual_long_chain', 'impossible_scenarios', 'anomaly_classification',
    'guardrail_consistency', 'rule_violation', 'spatial_grounding',
    'omission_causality', 'multi_event_causal', 'temporal_localization',
    'counting', 'action_sequence', 'negative_control'
]

# Crime type names for display
CRIME_TYPES = [
    'Normal', 'Explosion', 'Shooting', 'RoadAccidents', 'Shoplifting',
    'Riot', 'Fighting', 'Abuse', 'Burglary', 'Car accident',
    'Arson', 'Arrest', 'Robbery', 'Vandalism', 'Stealing',
    'Assault', 'Violence', 'Privacy violation', 'Physical altercation'
]

# Tier names
TIERS = [1, 2, 3, 5]


class CrimeBench(VideoBaseDataset):
    """
    CrimeBench Dataset Implementation

    *** MANUAL DOWNLOAD REQUIRED ***
    This dataset requires manual download.
    Set CRIMEBENCH_DATA_PATH environment variable to point to dataset directory.

    Args:
        dataset: Dataset name (default: 'CrimeBench')
        nframe: Number of frames to sample (mutually exclusive with fps)
        fps: Frames per second for sampling (mutually exclusive with nframe)

    Dataset Statistics:
    - Total: 5,625 QA pairs from 481 videos
    - Tiers: 1, 2, 3, 5
    - Question types: 18
    - Crime types: 19
    """

    TYPE = 'Video-MCQ'
    MODALITY = 'VIDEO'

    DATASET_PATH = '/home/ubuntu/data/crimebench/crimebench_export'

    FRAMES_TMPL_SYS = """
You will receive a video clip.
Based on the video, answer the following multiple-choice question.
"""

    FRAMES_TMPL_SYS_4VIDEO_LLM = """
You will receive a video clip.
Based on the video, answer the following multiple-choice question.
"""

    QUESTION_TMPL = """
Question: {}

A. {}
B. {}
C. {}
D. {}

Answer with the option letter (A, B, C, or D) of the correct option.
"""

    def __init__(self, dataset='CrimeBench', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return ['CrimeBench']
```

**Step 2: Verify file was created**

Run: `head -50 /home/ubuntu/VLMEvalKit/vlmeval/dataset/crimebench.py`
Expected: Shows the class skeleton code

**Step 3: Commit**

```bash
cd /home/ubuntu/VLMEvalKit && git add vlmeval/dataset/crimebench.py && git commit -m "feat(dataset): add CrimeBench class skeleton"
```

---

## Task 2: Implement prepare_dataset Method

**Files:**
- Modify: `vlmeval/dataset/crimebench.py`

**Step 1: Add prepare_dataset method after __init__**

Add this method to the CrimeBench class:

```python
    def prepare_dataset(self, dataset_name='CrimeBench'):
        """
        Prepare CrimeBench dataset from Arrow IPC format.

        The dataset should be located at DATASET_PATH with the following structure:
        crimebench_export/
        ├── crimebench_v2_final.parquet/
        │   ├── data-00000-of-00001.arrow
        │   ├── dataset_info.json
        │   └── state.json
        └── videos/
            └── *.mp4 files

        Returns:
            dict with 'root' (video directory) and 'data_file' (TSV file path)
        """
        data_path = os.environ.get('CRIMEBENCH_DATA_PATH', self.DATASET_PATH)

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"CrimeBench dataset not found at {data_path}. "
                f"Please set CRIMEBENCH_DATA_PATH environment variable or ensure dataset is at {self.DATASET_PATH}."
            )

        parquet_dir = os.path.join(data_path, 'crimebench_v2_final.parquet')
        arrow_file = os.path.join(parquet_dir, 'data-00000-of-00001.arrow')

        if not os.path.exists(arrow_file):
            raise FileNotFoundError(
                f"Arrow file not found: {arrow_file}. "
                f"Please ensure crimebench_v2_final.parquet directory exists."
            )

        tsv_filename = f'{dataset_name}.tsv'
        tsv_path = os.path.join(data_path, tsv_filename)

        if not os.path.exists(tsv_path) or self._needs_regeneration(arrow_file, tsv_path):
            self._generate_tsv_from_arrow(arrow_file, tsv_path, data_path)

        video_root = os.path.join(data_path, 'videos')

        if not os.path.exists(video_root):
            raise FileNotFoundError(
                f"Video directory not found: {video_root}. "
                f"Please ensure videos/ directory exists."
            )

        return dict(root=video_root, data_file=tsv_path)

    def _needs_regeneration(self, arrow_file, tsv_path):
        """Check if TSV needs to be regenerated."""
        if not os.path.exists(tsv_path):
            return True
        arrow_mtime = os.path.getmtime(arrow_file)
        tsv_mtime = os.path.getmtime(tsv_path)
        return arrow_mtime > tsv_mtime

    def _generate_tsv_from_arrow(self, arrow_file, tsv_path, data_path):
        """
        Convert CrimeBench Arrow IPC format to VLMEvalKit TSV format.
        """
        import pyarrow as pa

        with open(arrow_file, 'rb') as f:
            data = f.read()

        reader = pa.ipc.open_stream(io.BytesIO(data))
        table = reader.read_all()
        df = table.to_pandas()

        data_rows = []
        video_files = set()

        for idx, row in df.iterrows():
            video_path = row['video_path']
            video_files.add(video_path)

            candidates = row['candidates']
            if isinstance(candidates, np.ndarray):
                candidates = candidates.tolist()

            answer_idx = int(row['answer'])
            answer_letter = chr(65 + answer_idx)

            tsv_row = {
                'index': idx,
                'video': os.path.splitext(video_path)[0],
                'video_path': video_path,
                'question': row['question'],
                'A': candidates[0] if len(candidates) > 0 else '',
                'B': candidates[1] if len(candidates) > 1 else '',
                'C': candidates[2] if len(candidates) > 2 else '',
                'D': candidates[3] if len(candidates) > 3 else '',
                'answer': answer_letter,
                'tier': row['tier'],
                'question_type': row['question_type'],
                'crime_type': row['crime_type'],
            }

            data_rows.append(tsv_row)

        result_df = pd.DataFrame(data_rows)
        result_df.to_csv(tsv_path, sep='\t', index=False)

        print(f"Generated TSV file: {tsv_path}")
        print(f"Total questions: {len(result_df)}")
        print(f"Unique videos: {len(video_files)}")

        if 'tier' in result_df.columns:
            tier_counts = result_df['tier'].value_counts().sort_index()
            print(f"Tier distribution:")
            for tier, count in tier_counts.items():
                print(f"  Tier {tier}: {count} ({count/len(result_df)*100:.1f}%)")
```

**Step 2: Test the data loading**

Run: `cd /home/ubuntu/VLMEvalKit && source .venv/bin/activate && python3 -c "
from vlmeval.dataset.crimebench import CrimeBench
import os
os.environ['CRIMEBENCH_DATA_PATH'] = '/home/ubuntu/data/crimebench/crimebench_export'
cb = CrimeBench(nframe=8)
print(f'Dataset loaded: {len(cb.data)} rows')
print(f'Columns: {cb.data.columns.tolist()}')
print(cb.data.head(2))
"`
Expected: Dataset loads successfully, shows row count and columns

**Step 3: Commit**

```bash
cd /home/ubuntu/VLMEvalKit && git add vlmeval/dataset/crimebench.py && git commit -m "feat(dataset): add prepare_dataset method for CrimeBench"
```

---

## Task 3: Implement build_prompt Method

**Files:**
- Modify: `vlmeval/dataset/crimebench.py`

**Step 1: Add build_prompt method after _generate_tsv_from_arrow**

```python
    def build_prompt(self, line, video_llm=False):
        """
        Build prompt for CrimeBench question.

        Args:
            line: Data row (can be int index or dict)
            video_llm: If True, use video path instead of frames

        Returns:
            Message list with video/frames and text prompt
        """
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        if video_llm:
            message = [dict(type='text', value=self.FRAMES_TMPL_SYS_4VIDEO_LLM.strip())]
            video_path = os.path.normpath(os.path.join(self.data_root, line['video_path']))
            message.append(dict(type='video', value=video_path))
        else:
            frame_paths = self.save_video_frames(line['video'] + '.mp4')
            message = [dict(type='text', value=self.FRAMES_TMPL_SYS.strip().format(len(frame_paths)))]
            for frame_path in frame_paths:
                message.append(dict(type='image', value=frame_path))

        question_prompt = self.QUESTION_TMPL.format(
            line['question'],
            line['A'],
            line['B'],
            line['C'],
            line['D']
        ).strip()
        message.append(dict(type='text', value=question_prompt))

        return message
```

**Step 2: Test prompt building**

Run: `cd /home/ubuntu/VLMEvalKit && source .venv/bin/activate && python3 -c "
from vlmeval.dataset.crimebench import CrimeBench
import os
os.environ['CRIMEBENCH_DATA_PATH'] = '/home/ubuntu/data/crimebench/crimebench_export'
cb = CrimeBench(nframe=8)
prompt = cb.build_prompt(0, video_llm=True)
print('Video LLM prompt:')
for msg in prompt:
    print(f'  {msg[\"type\"]}: {str(msg[\"value\"])[:80]}...' if len(str(msg.get('value',''))) > 80 else f'  {msg[\"type\"]}: {msg.get(\"value\",\"\")}')
"`
Expected: Shows prompt with video path and question

**Step 3: Commit**

```bash
cd /home/ubuntu/VLMEvalKit && git add vlmeval/dataset/crimebench.py && git commit -m "feat(dataset): add build_prompt method for CrimeBench"
```

---

## Task 4: Implement evaluate Method

**Files:**
- Modify: `vlmeval/dataset/crimebench.py`

**Step 1: Add evaluate class method after build_prompt**

```python
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        Evaluate predictions on CrimeBench dataset.

        Computes:
        - Overall accuracy
        - Per-tier accuracy
        - Per-crime-type accuracy
        - Per-question-type accuracy

        Args:
            eval_file: Path to evaluation file with predictions
            **judge_kwargs: Additional arguments for judge

        Returns:
            Dictionary with evaluation results
        """
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'data file should be a supported format (xlsx/json/tsv) file'

        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')

            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None

            data = load(eval_file)

            for idx in data['index']:
                ans = str(data.loc[data['index'] == idx, 'answer'].values[0]).strip().upper()
                pred_raw = data.loc[data['index'] == idx, 'prediction'].values[0]

                if pd.isna(pred_raw) or (isinstance(pred_raw, str) and FAIL_MSG in pred_raw):
                    data.loc[idx, 'score'] = -1
                else:
                    pred = str(pred_raw)
                    pred_clean = pred.strip().upper()

                    match = re.search(r'[A-D]', pred_clean)
                    if match:
                        pred_letter = match.group(0)
                        data.loc[idx, 'score'] = int(pred_letter == ans)
                    else:
                        data.loc[idx, 'score'] = 0

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, '
                f'failed to obtain prediction for {len(data) - len(data[~pd.isna(data["prediction"])])} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        data = load(score_file)
        valid_data = data[data['score'] != -1]
        overall_acc = valid_data['score'].sum() / len(valid_data) if len(valid_data) > 0 else 0

        results = {
            'overall': {
                'acc': overall_acc,
                'total': len(data),
                'valid': len(valid_data),
                'correct': int(valid_data['score'].sum()) if len(valid_data) > 0 else 0
            }
        }

        # Per-tier accuracy
        if 'tier' in data.columns:
            for tier in TIERS:
                tier_data = data[data['tier'] == tier]
                tier_valid = tier_data[tier_data['score'] != -1]
                if len(tier_valid) > 0:
                    tier_acc = tier_valid['score'].sum() / len(tier_valid)
                    results[f'tier_{tier}'] = {
                        'acc': tier_acc,
                        'total': len(tier_data),
                        'valid': len(tier_valid),
                        'correct': int(tier_valid['score'].sum())
                    }

        # Per-crime-type accuracy
        if 'crime_type' in data.columns:
            for crime_type in data['crime_type'].unique():
                crime_data = data[data['crime_type'] == crime_type]
                crime_valid = crime_data[crime_data['score'] != -1]
                if len(crime_valid) > 0:
                    crime_acc = crime_valid['score'].sum() / len(crime_valid)
                    safe_key = crime_type.lower().replace(' ', '_').replace('-', '_')
                    results[f'crime_{safe_key}'] = {
                        'acc': crime_acc,
                        'total': len(crime_data),
                        'valid': len(crime_valid),
                        'correct': int(crime_valid['score'].sum())
                    }

        # Per-question-type accuracy
        if 'question_type' in data.columns:
            for q_type in data['question_type'].unique():
                q_type_data = data[data['question_type'] == q_type]
                q_type_valid = q_type_data[q_type_data['score'] != -1]
                if len(q_type_valid) > 0:
                    q_type_acc = q_type_valid['score'].sum() / len(q_type_valid)
                    results[f'qtype_{q_type}'] = {
                        'acc': q_type_acc,
                        'total': len(q_type_data),
                        'valid': len(q_type_valid),
                        'correct': int(q_type_valid['score'].sum())
                    }

        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(results, tgt_file)

        print(f"\nCrimeBench Evaluation Results:")
        print(f"Overall Accuracy: {overall_acc:.4f} ({int(valid_data['score'].sum()) if len(valid_data) > 0 else 0}/{len(valid_data)})")

        if 'tier' in data.columns:
            print(f"\nPer-Tier Accuracy:")
            for tier in TIERS:
                key = f'tier_{tier}'
                if key in results:
                    r = results[key]
                    print(f"  Tier {tier}: {r['acc']:.4f} ({r['correct']}/{r['valid']})")

        return results
```

**Step 2: Verify evaluate method syntax**

Run: `cd /home/ubuntu/VLMEvalKit && source .venv/bin/activate && python3 -c "from vlmeval.dataset.crimebench import CrimeBench; print('evaluate method exists:', hasattr(CrimeBench, 'evaluate'))"`
Expected: `evaluate method exists: True`

**Step 3: Commit**

```bash
cd /home/ubuntu/VLMEvalKit && git add vlmeval/dataset/crimebench.py && git commit -m "feat(dataset): add evaluate method with comprehensive metrics for CrimeBench"
```

---

## Task 5: Register CrimeBench in __init__.py

**Files:**
- Modify: `vlmeval/dataset/__init__.py`

**Step 1: Add import statement**

Find line with `from .trafficqa import TrafficQA` and add after it:

```python
from .crimebench import CrimeBench
```

**Step 2: Add to VIDEO_DATASET list**

Find the `VIDEO_DATASET` list (around line 240-250) and add `CrimeBench` to it:

```python
VIDEO_DATASET = [
    MMBenchVideo, VideoMME, MVBench, MVBench_MP4, MVTamperBench,
    LongVideoBench, WorldSense, VDC, MovieChat1k, MEGABench,
    MLVU, MLVU_MCQ, MLVU_OpenEnded,
    TempCompass, TempCompass_MCQ, TempCompass_Captioning, TempCompass_YorN,
    CGBench_MCQ_Grounding_Mini, CGBench_OpenEnded_Mini, CGBench_MCQ_Grounding, CGBench_OpenEnded,
    QBench_Video, QBench_Video_MCQ, QBench_Video_VQA,
    Video_MMLU_CAP, Video_MMLU_QA,
    Video_Holmes, VCRBench, CGAVCounting,
    EgoExoBench_MCQ, DREAM, VideoTT, VideoMMMU, VSIBench,
    TrafficQA, MotionBench, CrimeBench
]
```

**Step 3: Verify registration**

Run: `cd /home/ubuntu/VLMEvalKit && source .venv/bin/activate && python3 -c "from vlmeval.dataset import build_dataset, SUPPORTED_DATASETS; print('CrimeBench in supported:', 'CrimeBench' in SUPPORTED_DATASETS)"`
Expected: `CrimeBench in supported: True`

**Step 4: Commit**

```bash
cd /home/ubuntu/VLMEvalKit && git add vlmeval/dataset/__init__.py && git commit -m "feat(dataset): register CrimeBench in VIDEO_DATASET"
```

---

## Task 6: End-to-End Test

**Files:**
- Test: `vlmeval/dataset/crimebench.py`

**Step 1: Test full dataset loading and prompt building**

Run: `cd /home/ubuntu/VLMEvalKit && source .venv/bin/activate && python3 << 'EOF'
import os
os.environ['CRIMEBENCH_DATA_PATH'] = '/home/ubuntu/data/crimebench/crimebench_export'

from vlmeval.dataset import build_dataset

# Build dataset
ds = build_dataset('CrimeBench', nframe=8)
print(f"Dataset type: {type(ds).__name__}")
print(f"Dataset length: {len(ds)}")
print(f"Dataset name: {ds.dataset_name}")
print(f"Data columns: {ds.data.columns.tolist()}")
print(f"Sample row: {dict(ds[0])}")

# Test prompt building for video LLM
prompt = ds.build_prompt(0, video_llm=True)
print(f"\nPrompt for video LLM has {len(prompt)} parts")

# Check supported datasets
from vlmeval.dataset import SUPPORTED_DATASETS
print(f"\nCrimeBench in SUPPORTED_DATASETS: {'CrimeBench' in SUPPORTED_DATASETS}")
EOF
`
Expected: Dataset loads, shows correct length, prompt builds successfully

**Step 2: Verify TSV file was created**

Run: `ls -la /home/ubuntu/data/crimebench/crimebench_export/CrimeBench.tsv && head -3 /home/ubuntu/data/crimebench/crimebench_export/CrimeBench.tsv`
Expected: TSV file exists with proper columns

**Step 3: Final commit**

```bash
cd /home/ubuntu/VLMEvalKit && git add -A && git commit -m "feat(dataset): complete CrimeBench implementation"
```

---

## Task 7: Run Sample Evaluation (Optional)

**Files:**
- Test: Full evaluation pipeline

**Step 1: Run a quick test with a model**

Run: `cd /home/ubuntu/VLMEvalKit && source .venv/bin/activate && CRIMEBENCH_DATA_PATH=/home/ubuntu/data/crimebench/crimebench_export python run.py --model Qwen2-VL-7B --benchmark CrimeBench --nframe 8 --verbose`

Note: This requires a model to be configured. Skip if not available.

**Expected:** Evaluation runs and produces results with overall, per-tier, per-crime-type, and per-question-type accuracy.

---

## Summary

After completing all tasks:
1. CrimeBench dataset class is created in `vlmeval/dataset/crimebench.py`
2. Dataset is registered in `vlmeval/dataset/__init__.py`
3. Can be used with: `python run.py --model <MODEL> --benchmark CrimeBench --nframe 8`

**Files Modified/Created:**
- Created: `vlmeval/dataset/crimebench.py`
- Modified: `vlmeval/dataset/__init__.py`
- Generated: `{CRIMEBENCH_DATA_PATH}/CrimeBench.tsv` (cached)
