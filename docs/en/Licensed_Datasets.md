# Working with Licensed Datasets

This guide explains how VLMEvalKit handles datasets that require manual download or have licensing restrictions.

## Overview

VLMEvalKit uses a **two-tier approach** for data management:

| Tier | Description | Examples |
|------|-------------|----------|
| **Tier 1: HuggingFace** | Publicly available datasets that auto-download | MMBench, Video-MME, LongVideoBench |
| **Tier 2: Local/Manual** | Licensed or restricted datasets requiring user action | TrafficQA, EgoExo4D, proprietary data |

## Data Root Configuration

### The `LMUData` Environment Variable

VLMEvalKit uses `LMUData` as the default root for local datasets:

```python
def LMUDataRoot():
    # 1. Check environment variable first
    if 'LMUData' in os.environ and osp.exists(os.environ['LMUData']):
        return os.environ['LMUData']

    # 2. Default to ~/LMUData
    home = osp.expanduser('~')
    root = osp.join(home, 'LMUData')
    os.makedirs(root, exist_ok=True)
    return root
```

**Usage:**
```bash
# Use default location (~/LMUData)
# Datasets will be stored in ~/LMUData/DatasetName/

# Set custom location
export LMUData=/storage/disk3/datasets
# Datasets will be stored in /storage/disk3/datasets/DatasetName/
```

### Dataset-Specific Environment Variables

Individual datasets can have their own path variables:

```bash
# TrafficQA specific path
export TRAFFICQA_DATA_PATH=/custom/path/to/TrafficQA

# Other datasets follow the pattern: {DATASET_NAME}_DATA_PATH
```

## Implementation Patterns

### Pattern 1: Auto-Download (Public Datasets)

For datasets available on HuggingFace:

```python
from huggingface_hub import snapshot_download
from ..smp import get_cache_path

class PublicDataset(VideoBaseDataset):
    def prepare_dataset(self, dataset_name='PublicDataset'):
        # Try cache first
        cache_path = get_cache_path('org/dataset')
        if cache_path and check_integrity(cache_path):
            return dict(root=cache_path, data_file=...)

        # Download from HuggingFace
        dataset_path = snapshot_download(
            repo_id='org/dataset',
            repo_type='dataset'
        )
        return dict(root=dataset_path, data_file=...)
```

### Pattern 2: Manual Download (Licensed Datasets)

For datasets requiring user action:

```python
import os
from ..smp import LMUDataRoot

class LicensedDataset(VideoBaseDataset):
    # Default path
    DATASET_PATH = os.path.join(LMUDataRoot(), 'LicensedDataset')

    def prepare_dataset(self, dataset_name='LicensedDataset'):
        # Environment override
        data_path = os.environ.get(
            'LICENSED_DATA_PATH',
            self.DATASET_PATH
        )

        # Check existence
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Dataset not found at {data_path}.\n"
                f"Please download from: https://dataset-website.org/download\n"
                f"Set LICENSED_DATA_PATH to custom location if needed.\n"
                f"Expected structure:\n"
                f"  {data_path}/\n"
                f"  ├── annotations/\n"
                f"  └── videos/"
            )

        # Verify required files
        required_files = [
            'annotations/test.jsonl',
            'videos/'
        ]
        for file in required_files:
            if not os.path.exists(os.path.join(data_path, file)):
                raise FileNotFoundError(f"Missing: {file}")

        return dict(root=data_path, data_file=tsv_path)
```

### Pattern 3: Mixed Approach (Partial Auto-Download)

For datasets with both public and licensed components:

```python
class MixedDataset(VideoBaseDataset):
    def __init__(self, skip_licensed=False):
        self.skip_licensed = skip_licensed

    def prepare_dataset(self, dataset_name='MixedDataset'):
        # Auto-download public annotations
        dataset_path = snapshot_download(
            repo_id='org/mixed-dataset',
            repo_type='dataset'
        )

        # Local path for licensed videos
        video_root = os.path.join(LMUDataRoot(), 'videos', 'MixedDataset')
        os.makedirs(video_root, exist_ok=True)

        # Check if licensed component exists
        licensed_path = os.path.join(video_root, 'licensed_component/')
        if not os.path.exists(licensed_path):
            if self.skip_licensed:
                warnings.warn("Licensed component not found, skipping...")
            else:
                raise FileNotFoundError(
                    f"Licensed component required. "
                    f"Download from: https://example.org/license-request\n"
                    f"Or use skip_licensed=True to exclude this component."
                )

        return dict(root=video_root, data_file=...)
```

## Real Examples

### Example 1: EgoExoBench

**From**: `vlmeval/dataset/EgoExoBench/README.md`

```markdown
The script will automatically download the processed video data, **except Ego-Exo4D**,
due to license restrictions. You need to manually download it from the
[official website](https://ego-exo4d-data.org/) and organize it as shown below.
```

**Expected Structure**:
```
$LMUData/videos/EgoExoBench/
├── CVMHAT/
├── Ego-Exo4D/          # ← Manual download required
├── EgoExoLearn/
├── EgoMe/
├── LEMMA/
├── TF2023/
├── processed_frames/
└── processed_videos/
```

**Usage**:
```bash
# Evaluate with Ego-Exo4D (requires manual download)
python run.py --data EgoExoBench_MCQ --model <model>

# Skip Ego-Exo4D
python run.py --data EgoExoBench_64frame_skip_EgoExo4D --model <model>
```

### Example 2: TrafficQA

**Implementation**: `vlmeval/dataset/trafficqa.py`

```python
class TrafficQA(VideoBaseDataset):
    DATASET_PATH = '/storage/disk3/datasets/SUTD-TrafficQA'

    def prepare_dataset(self, dataset_name='TrafficQA'):
        # Environment override
        data_path = os.environ.get('TRAFFICQA_DATA_PATH', self.DATASET_PATH)

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"TrafficQA dataset not found at {data_path}. "
                f"Download from: https://github.com/SUTD-TaCheng/TrafficQA"
            )
```

**Usage**:
```bash
# Use default path
python run.py --data TrafficQA_test_8frame --model <model>

# Use custom path
export TRAFFICQA_DATA_PATH=/custom/path/TrafficQA
python run.py --data TrafficQA_test_8frame --model <model>
```

## Best Practices

### 1. Clear Error Messages

When data is missing, provide:
- What's missing
- Where to download
- How to configure paths
- Expected directory structure

```python
raise FileNotFoundError(
    f"Dataset 'XXX' not found at {data_path}\n"
    f"\n"
    f"Please follow these steps:\n"
    f"1. Request access from: https://example.org/request\n"
    f"2. Download the dataset\n"
    f"3. Organize files as:\n"
    f"   {data_path}/\n"
    f"   ├── annotations/\n"
    f"   └── videos/\n"
    f"4. Or set custom path: export XXX_DATA_PATH=/custom/path\n"
)
```

### 2. Documentation

Create a README in your dataset folder with:

```markdown
# DatasetName

## License

This dataset requires acceptance of a license agreement.

## Download

1. Request access: https://dataset.org/request
2. Download after approval
3. Extract to: $LMUData/DatasetName/

## Structure

```
$LMUData/DatasetName/
├── annotations/
│   ├── test.jsonl
│   └── train.jsonl
└── videos/
    └── *.mp4
```

## Configuration

```bash
# Optional: Set custom path
export DATASET_DATA_PATH=/custom/path
```
```

### 3. Environment Variable Naming

Follow the pattern: `{DATASET_NAME}_DATA_PATH`

- `TRAFFICQA_DATA_PATH`
- `EGOEXOBENCH_DATA_PATH`
- `CUSTOM_DATASET_DATA_PATH`

Use uppercase and replace hyphens with underscores.

### 4. Verification

Always verify required files exist:

```python
required_files = [
    'annotations/test.jsonl',
    'videos/',
    'metadata.json'
]

for file in required_files:
    path = os.path.join(data_path, file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            f"Please ensure the dataset is completely downloaded."
        )
```

### 5. Graceful Degradation

For datasets with optional components:

```python
# Check if optional licensed component exists
if not os.path.exists(licensed_path):
    if self.skip_optional:
        warnings.warn("Optional component not found, using reduced dataset")
        return prepare_reduced_dataset()
    else:
        raise FileNotFoundError("Licensed component required")
```

## Template: Creating a New Licensed Dataset

```python
# vlmeval/dataset/your_dataset.py

from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
import os

class YourLicensedDataset(VideoBaseDataset):
    """
    YourLicensedDataset - Description

    License: Requires acceptance at https://example.org/license
    Download: Request access from https://example.org/request
    """

    TYPE = 'Video-MCQ'  # or 'Video-VQA', 'Y/N', etc.

    # Default path (can be overridden via environment)
    DATASET_PATH = os.path.join(LMUDataRoot(), 'YourLicensedDataset')

    def __init__(self, dataset='YourLicensedDataset', nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['YourLicensedDataset']

    def prepare_dataset(self, dataset_name='YourLicensedDataset'):
        # 1. Get data path (with environment override)
        data_path = os.environ.get(
            'YOUR_LICENSED_DATASET_DATA_PATH',
            self.DATASET_PATH
        )

        # 2. Check if dataset exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"YourLicensedDataset not found at {data_path}\n"
                f"\n"
                f"To use this dataset:\n"
                f"1. Request access from: https://example.org/request\n"
                f"2. Download the dataset\n"
                f"3. Extract to: {self.DATASET_PATH}\n"
                f"4. Or set custom path: export YOUR_LICENSED_DATASET_DATA_PATH=/custom/path\n"
                f"\n"
                f"Expected structure:\n"
                f"  {data_path}/\n"
                f"  ├── annotations/\n"
                f"  │   ├── test.jsonl\n"
                f"  │   └── train.jsonl\n"
                f"  └── videos/\n"
                f"      └── *.mp4\n"
            )

        # 3. Verify required files
        tsv_path = os.path.join(data_path, f'{dataset_name}.tsv')
        jsonl_path = os.path.join(data_path, 'annotations', 'test.jsonl')

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Required file not found: {jsonl_path}")

        video_root = os.path.join(data_path, 'videos')
        if not os.path.exists(video_root):
            raise FileNotFoundError(f"Video directory not found: {video_root}")

        # 4. Generate TSV if needed
        if not os.path.exists(tsv_path) or self._needs_regeneration(jsonl_path, tsv_path):
            self._generate_tsv_from_jsonl(jsonl_path, tsv_path)

        return dict(root=video_root, data_file=tsv_path)

    def _needs_regeneration(self, jsonl_path, tsv_path):
        """Check if TSV needs regeneration based on file modification times."""
        return (os.path.exists(jsonl_path) and
                (not os.path.exists(tsv_path) or
                 os.path.getmtime(jsonl_path) > os.path.getmtime(tsv_path)))

    def _generate_tsv_from_jsonl(self, jsonl_path, tsv_path):
        """Convert JSONL format to VLMEvalKit TSV format."""
        import json
        import pandas as pd

        data_rows = []
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                record = json.loads(line)

                # Convert to TSV row
                tsv_row = {
                    'index': line_num,
                    'video': record['video_id'],
                    'video_path': f"{record['video_id']}.mp4",
                    'question': record['question'],
                    'options': record['options'],
                    'answer': record['answer'],
                }
                data_rows.append(tsv_row)

        df = pd.DataFrame(data_rows)
        df.to_csv(tsv_path, sep='\t', index=False)

    def build_prompt(self, line, video_llm=False):
        """Build prompt for the model."""
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        if video_llm:
            message = [dict(type='video', value=os.path.join(self.data_root, line['video_path']))]
        else:
            frame_paths = self.save_video_frames(line)
            message = [dict(type='image', value=fp) for fp in frame_paths]

        message.append(dict(type='text', value=line['question']))
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """Evaluate predictions."""
        data = load(eval_file)

        # Calculate accuracy
        for idx in data['index']:
            ans = str(data.loc[data['index'] == idx, 'answer'].values[0]).strip().upper()
            pred = str(data.loc[data['index'] == idx, 'prediction'].values[0]).strip().upper()
            data.loc[idx, 'score'] = int(ans == pred)

        valid_data = data[data['score'] != -1]
        accuracy = valid_data['score'].sum() / len(valid_data)

        return {'acc': accuracy, 'total': len(data), 'valid': len(valid_data)}
```

## Summary

| Aspect | Auto-Download | Manual/Licensed |
|--------|---------------|-----------------|
| **Source** | HuggingFace | User-provided |
| **Path Config** | Cache (auto) | Environment variable |
| **Error Handling** | Auto-download | Clear error + download link |
| **Structure** | Managed by toolkit | Documented in README |
| **Example** | Video-MME, MMBench | TrafficQA, Ego-Exo4D |

## See Also

- [TrafficQA Documentation](TrafficQA.md)
- [Adding a New Dataset](Development.md)
- [Configuration System](ConfigSystem.md)
