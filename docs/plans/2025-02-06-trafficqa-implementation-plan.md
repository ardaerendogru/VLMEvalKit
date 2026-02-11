# TrafficQA Dataset Implementation Plan

## Overview

This plan details the implementation of **SUTD-TrafficQA** (Traffic Question Answering) dataset into VLMEvalKit. TrafficQA is a video question answering benchmark for traffic scenes published in CVPR 2021.

### Dataset Summary
- **Name**: SUTD-TrafficQA
- **Type**: Video Question Answering (Multiple Choice)
- **Source**: https://github.com/sutdcv/SUTD-TrafficQA
- **Paper**: CVPR 2021 - "SUTD-TrafficQA: A Question Answering Benchmark and an Efficient Network for Video Reasoning over Traffic Events"
- **Scale**: 10,080 traffic videos (12GB), 62,533 QA pairs
- **Format**: JSONL annotations (array format) with variable options
- **Splits**: Train (56,459), Test (6,074)
- **Local Path**: `/storage/disk3/datasets/SUTD-TrafficQA`

### Local Dataset Structure

```
/storage/disk3/datasets/SUTD-TrafficQA/
├── annotations/
│   ├── R2_all.jsonl      (62,534 lines: header + 62,533 QA pairs)
│   ├── R2_train.jsonl    (56,460 lines: header + 56,459 QA pairs)
│   ├── R2_test.jsonl     (6,075 lines: header + 6,074 QA pairs)
│   ├── vid_filename_to_id.json (10,080 mappings)
│   ├── vid_id_to_filename.json (10,080 mappings)
│   └── ICCV 2021 Workshop - MMVRAC/
│       ├── workshop_train.jsonl (56,460 QAs, no q_type field)
│       ├── workshop_test_no_ans.jsonl (3,000 QAs, competition set)
│       └── workshop_test_answer.json (answer keys for test set)
└── compressed_videos/
    ├── b_114411w739_clip_006.mp4
    ├── b_114411w739_clip_007.mp4
    └── ... (10,080 video files total, 12GB)
```

**Dataset Statistics**:
- Total videos: 10,080 (12GB)
- Total QA pairs: 62,533 (R2_all)
- R2_train: 56,459 QAs from 10,051 videos
- R2_test: 6,075 QAs from 4,111 videos
- ICCV Workshop test: 3,000 QAs from 2,325 videos (q_type hidden)
- Avg QA per video: 6.2 (range: 1-15)
- Video filename patterns: 77.8% `b_*`, 7.7% `c_*`, 14.5% other

### The Six Reasoning Tasks (from CVPR 2021 Paper)

1. **Basic Understanding** - Perceiving and understanding traffic scenarios at the basic level, including:
   - Feature-query (vehicle type, road situation, environment description)
   - Event-query (accident existence, pedestrian action, temporal relations)
   - Event classification (accident type)
   - Counting (road-agent number)

2. **Event Forecasting** - Infer future events based on observed videos; queries about the outcome of the current situation

3. **Reverse Reasoning** - Ask about events that happened before the start of a video segment

4. **Counterfactual Inference** - Query consequent outcomes of certain hypotheses (e.g., "what if the blue sedan had not accelerated?"); requires reasoning about imagined events under designated conditions

5. **Introspection** - Test if models can provide preventive advice (e.g., "what could the pedestrian have done to avoid the collision?")

6. **Attribution** - Seek explanations about the causes of traffic events (e.g., "what are the reasons for the rear-end crash?")

## Critical Dataset Issues

### ⚠️ Train/Test Video Overlap (By Design)

**Discovery**: The train/test split has significant video overlap:
- **99.3%** of test videos (4,082 out of 4,111) are also in training set
- **99.5%** of test QAs (6,042 out of 6,075) come from overlapping videos
- Only **29 test-only videos** (0.7%) with **33 QAs** (0.5%) are truly unseen

**Understanding**: This is **by design** - TrafficQA follows the "same video, different questions" paradigm similar to:
- VQA v2: Same images, multiple questions
- GQA: Same images, different question structures
- MovieChat: Same video clips, different questions

**Original Paper Evaluation**:
- CVPR 2021: Used full dataset (62,535 QAs) for training/evaluation
- ICCV 2021 Workshop: Created 3000 QA competition set (q_type hidden)
- Human performance: 95.43% (Setting-1/4)

**Recommendation for VLMEvalKit**:
1. Use **R2_test.jsonl** (6,075 QAs) as primary evaluation set - community standard
2. Document: "99.5% of test QAs from videos also in training set (by design)"
3. Report per-reasoning-task metrics (U, A, F, R, C, I)
4. Optionally report "strict" metrics on 33 test-only QAs as sanity check
5. Do NOT use R2_train.jsonl for evaluation purposes

### Perspective Distribution

**Discovery**: Perspective 2 is completely unused:
- Perspective 1: 46.4% (likely: dashboard view)
- Perspective 2: 0% (NOT USED)
- Perspective 3: 53.6% (likely: external view)

**Implication**: Only 2 camera perspectives are used, not 3 as the field name suggests.

## Architecture Analysis

### VLMEvalKit Video Dataset Pattern

Video datasets in VLMEvalKit follow this pattern:

1. **Inherit from `VideoBaseDataset`** (`vlmeval/dataset/video_base.py`)
2. **Implement three required abstract methods**:
   - `prepare_dataset()` - Returns dict with `root` (video dir) and `data_file` (annotation path)
   - `build_prompt()` - Builds the prompt message for model input
   - `evaluate()` - Evaluates predictions and returns metrics

3. **Register in `__init__.py`**:
   - Add import statement
   - Add to `VIDEO_DATASET` list
   - Add configuration variants to `video_dataset_config.py`

### TrafficQA Specifics

From local dataset analysis:
- **Annotation Format**: JSONL with array format (each line is a JSON array, not object)
- **Fields** (in order):
  - `record_id`: Unique identifier
  - `vid_id`: Video ID
  - `vid_filename`: Video filename (e.g., `b_1ZV411k741_clip_023.mp4`)
  - `perspective`: Camera perspective (1, 2, or 3 for different angles)
  - `q_body`: Question text
  - `q_type`: Question type code (maps to reasoning tasks)
  - `option0`, `option1`, `option2`, `option3`: Multiple choice options (some may be empty)
  - `answer`: Correct option index (0-3)

### q_type to Reasoning Task Mapping

From local dataset analysis and paper verification:

| q_type | Count | Percentage | Reasoning Task | Description |
|--------|-------|------------|----------------|-------------|
| **U** | 38,772 | 62.0% | Basic Understanding | Perceiving traffic scenarios, counting, classification |
| **A** | 12,719 | 20.3% | Attribution | Explaining causes of traffic events/accidents |
| **C** | 3,009 | 4.8% | Counterfactual Inference | "What if" hypothetical scenarios |
| **F** | 2,736 | 4.4% | Event Forecasting | Predicting future events |
| **R** | 2,769 | 4.4% | Reverse Reasoning | Inferring past events before video starts |
| **I** | 2,528 | 4.0% | Introspection | Providing preventive advice |

- **Video Diversity**:
  - Different weather conditions (sunny/rainy/windy/snowy)
  - Different times (daytime/night)
  - Various road situations (congested/sparse, urban/rural)
  - Various traffic events (accidents, vehicle turning, pedestrian behaviors, traffic lights)
  - Different video perspectives (surveillance camera, car-mounted, hand-held)
  - Various clip lengths (1-70 seconds)

- **Question Statistics**:
  - Average question length: 8.6 words
  - Number of candidate answers varies from 2 to 12 (sampled to 4 for balanced evaluation)
  - Balanced distribution across reasoning tasks to minimize language biases

## Implementation Plan

### File Structure

```
vlmeval/dataset/
├── trafficqa.py              # NEW: TrafficQA dataset class
├── video_base.py             # EXISTING: Base class
├── __init__.py               # MODIFY: Import TrafficQA
└── video_dataset_config.py   # MODIFY: Add TrafficQA config
```

### Step 1: Create `trafficqa.py`

**Location**: `/storage/disk0/arda/VLMEvalKit/vlmeval/dataset/trafficqa.py`

**Implementation Outline**:

```python
from .video_base import VideoBaseDataset
from ..smp import *
import pandas as pd
import json

class TrafficQA(VideoBaseDataset):
    TYPE = 'Video-VQA'
    MODALITY = 'VIDEO'

    # Dataset URL and MD5 (to be determined)
    DATASET_URL = {}
    DATASET_MD5 = {}

    def __init__(self, dataset='TrafficQA', pack=False, nframe=8, fps=-1, subset='all'):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)
        # Filter by subset if needed (e.g., by q_type)

    @classmethod
    def supported_datasets(cls):
        return ['TrafficQA']

    def prepare_dataset(self, dataset):
        # 1. Check if local dataset exists at /storage/disk3/datasets/SUTD-TrafficQA
        # 2. Convert JSONL (array format) to TSV format expected by VLMEvalKit
        # 3. Handle empty options by filtering and reindexing
        # 4. Return dict: {'data_file': tsv_path, 'root': video_dir}

        # Local path
        data_root = '/storage/disk3/datasets/SUTD-TrafficQA'
        jsonl_file = osp.join(data_root, 'annotations', 'R2_all.jsonl')
        video_dir = osp.join(data_root, 'compressed_videos')

    def build_prompt(self, line, video_llm=False):
        # 1. Extract video path
        # 2. Build question with options
        # 3. Return message format:
        #    [video_dict, text_dict] if video_llm
        #    [image_dict, ..., text_dict] otherwise

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        # Calculate accuracy metrics
        # Return results dict/DataFrame
```

**Key Design Decisions**:

1. **Data Format Conversion**: VLMEvalKit expects TSV format with specific columns. Need to:
   - Parse JSONL format
   - Map fields: `q_body` → `question`, options → `A, B, C, D`, `answer` → index
   - Add required columns: `index`, `video`

2. **Video Path Handling**:
   - Remove `.mp4` extension for internal video ID
   - Map `vid_filename` to actual video file location

3. **Prompt Building**:
   - Format: "Question: {q_body}\nA. {option0}\nB. {option1}\nC. {option2}\nD. {option3}\nAnswer:"
   - Support both video LLM (native video) and frame-based (image sequence)

### Step 2: Data Preparation Details

**JSONL to TSV Conversion**:

Input JSONL structure (array format):
```json
["record_id", "vid_id", "vid_filename", "perspective", "q_body", "q_type", "option0", "option1", "option2", "option3", "answer"]
[9875, 20485, "b_1ZV411k741_clip_023.mp4", 1, "What type of accident happened?", "A", "Head-on collision", "No accident happened.", "The vehicle collided with road infrastructure.", "Others", 1]
```

**Important Notes**:
- JSONL uses **array format**, not JSON objects
- Some options may be **empty strings** (need to handle carefully)
- Header is on first line
- **Critical**: Options can be in ANY positions (not just starting from option0)
- Answer index (0-3) ALWAYS points to a valid (non-empty) option in its ORIGINAL position
- **Do NOT reindex options** - the answer index refers to original option positions

### Variable Option Positioning (Critical!)

Analysis of local dataset shows:
- 42.3% of entries have non-contiguous options or don't start from option0
- 11 different option position combinations exist
- Answer index is ALWAYS valid (points to non-empty option)

**Examples**:
```
Filled: option1, option3 (B, D positions) → Answer 3 means option3 (D)
Filled: option2, option3 (C, D positions) → Answer 2 means option2 (C)
Filled: option0, option2 (A, C positions) → Answer 0 means option0 (A)
```

**Implication for Implementation**:
- Keep original option positions (option0=A, option1=B, option2=C, option3=D)
- Map empty options to empty string in TSV
- Answer index stays the same - it's already valid
- When building prompts, filter out empty options but maintain position mapping

Output TSV structure (VLMEvalKit format):
```
index	video	question	A	B	C	D	answer		q_type
0	b_1ZV411k741_clip_023	What type of accident happened?	Head-on collision	No accident happened.	The vehicle collided with road infrastructure.	Others	1	A
```

### Step 3: Dataset Download Strategy

**Options**:
1. **Direct from GitHub**: Use raw GitHub content URL
2. **HuggingFace Dataset**: Create/upload dataset to HF (preferred for caching)

**Recommended**: Create HuggingFace dataset for better caching and versioning:
- Upload converted TSV files
- Upload videos or provide download instructions
- Use `snapshot_download()` for automatic caching

### Step 4: Prompt Building Logic

**Critical: Handling Variable Option Positions**

Since options can be in any positions (option0, option1+option3, option2+option3, etc.), we need to:

1. Filter out empty options when building the prompt
2. Create a mapping from displayed letters to original indices
3. Parse model answers to match the original indices

**Frame-based (non-video-llm)**:
```python
def build_prompt_nopack(self, line, video_llm=False):
    question = line['question']

    # Get non-empty options and create mapping
    options = []
    option_letters = []
    letter_to_original_idx = {}

    for opt_idx, opt_letter in enumerate(['A', 'B', 'C', 'D']):
        opt_key = f'option{opt_idx}'
        if opt_key in line and line[opt_key] and line[opt_key].strip():
            options.append(line[opt_key])
            option_letters.append(opt_letter)
            letter_to_original_idx[opt_letter] = opt_idx

    # Build prompt with only non-empty options
    prompt = f"Question: {question}\n"
    for letter, text in zip(option_letters, options):
        prompt += f"{letter}. {text}\n"
    prompt += "Answer with the option letter."

    video_id = line['video']  # Without extension
    frames = self.save_video_frames(video_id)
    message = [dict(type='image', value=fp) for fp in frames]
    message.append(dict(type='text', value=prompt))

    # Store mapping for answer parsing
    message.append(dict(type='meta', value={'letter_to_idx': letter_to_original_idx}))

    return message
```

**Answer Parsing**:
```python
def parse_answer(self, prediction, letter_to_original_idx):
    # Model returns letter (A, B, C, D)
    match = re.search(r'\b([ABCD])\b', str(prediction).upper())
    if match:
        letter = match.group(1)
        # Convert back to original index
        return letter_to_original_idx.get(letter, -1)
    return -1
```

**Video LLM (native video)**:
```python
def build_prompt_video_llm(self, line):
    # Similar prompt but with video file instead of frames
    video_path = osp.join(self.data_root, line['video'] + '.mp4')
    return [
        dict(type='video', value=video_path),
        dict(type='text', value=prompt)
    ]
```

### Step 5: Evaluation Logic

**Metrics**:
- Overall accuracy
- Per-reasoning-task accuracy for all 6 tasks:
  1. Basic Understanding
  2. Event Forecasting
  3. Reverse Reasoning
  4. Counterfactual Inference
  5. Introspection
  6. Attribution
- By perspective (if relevant)

**Implementation**:
```python
@classmethod
def evaluate(cls, eval_file, **judge_kwargs):
    data = load(eval_file)
    data = data[~pd.isna(data['prediction'])]

    # Extract answer letter from prediction
    def extract_answer(pred):
        # Parse prediction to get A, B, C, or D
        match = re.search(r'\b([ABCD])\b', str(pred).upper())
        return match.group(1) if match else None

    data['parsed'] = data['prediction'].apply(extract_answer)

    # Map answer index to letter
    def idx_to_letter(idx):
        return ['A', 'B', 'C', 'D'][int(idx)]

    data['correct'] = data['parsed'] == data['answer'].apply(idx_to_letter)

    # Calculate metrics
    results = {
        'overall_accuracy': data['correct'].mean() * 100,
        'total': len(data),
        'correct': data['correct'].sum()
    }

    # Per-reasoning-task metrics
    # Map q_type codes (from actual dataset) to full reasoning task names
    reasoning_task_map = {
        'U': 'Basic_Understanding',
        'A': 'Attribution',
        'C': 'Counterfactual_Inference',
        'F': 'Event_Forecasting',
        'R': 'Reverse_Reasoning',
        'I': 'Introspection'
    }

    if 'q_type' in data:
        for task_code, task_name in reasoning_task_map.items():
            subset = data[data['q_type'] == task_code]
            if len(subset) > 0:
                results[f'{task_name}_accuracy'] = subset['correct'].mean() * 100
                results[f'{task_name}_total'] = len(subset)

    return results
```

**Note**: The `q_type` field in the dataset uses single-letter codes (U, A, C, F, R, I) that directly map to the 6 reasoning tasks. Verified against local dataset:
- U (62%) → Basic Understanding
- A (20%) → Attribution (accident cause analysis)
- C (5%) → Counterfactual Inference
- F (4%) → Event Forecasting
- R (4%) → Reverse Reasoning
- I (4%) → Introspection

### Step 6: Registration in `__init__.py`

**Add import** (around line 67):
```python
from .trafficqa import TrafficQA
```

**Add to VIDEO_DATASET list** (around line 247):
```python
VIDEO_DATASET = [
    ...
    TrafficQA,
    ...
]
```

### Step 7: Configuration in `video_dataset_config.py`

**Add configuration variants** (around line 207):
```python
trafficqa_dataset = {
    'TrafficQA_8frame': partial(TrafficQA, dataset='TrafficQA', nframe=8),
    'TrafficQA_16frame': partial(TrafficQA, dataset='TrafficQA', nframe=16),
    'TrafficQA_32frame': partial(TrafficQA, dataset='TrafficQA', nframe=32),
    'TrafficQA_64frame': partial(TrafficQA, dataset='TrafficQA', nframe=64),
    'TrafficQA_1fps': partial(TrafficQA, dataset='TrafficQA', fps=1.0),
    'TrafficQA_0.5fps': partial(TrafficQA, dataset='TrafficQA', fps=0.5),
}
```

**Add to dataset_groups** (around line 216):
```python
dataset_groups = [
    ...
    trafficqa_dataset,
]
```

## Testing Plan

### Unit Tests
1. Test `prepare_dataset()` - Verify data file and video root are set correctly
2. Test `build_prompt()` - Verify prompt format for both video_llm and non-video_llm
3. Test JSONL to TSV conversion - Verify all fields map correctly

### Integration Tests
1. Load dataset with `build_dataset('TrafficQA')`
2. Run inference on a small subset
3. Verify evaluation metrics are calculated correctly

### Usage Examples

```python
# Basic usage
python run.py --dataset TrafficQA_8frame --model GPT4o

# With video LLM
python run.py --dataset TrafficQA_8frame --model VideoLLM --video_llm

# Custom frame count
python run.py --dataset TrafficQA --model InternVL2 --nframe 16
```

## Dependencies

- `decord` - Video frame extraction (already required)
- `pandas` - Data manipulation (already required)
- `json` - JSONL parsing (standard library)

## Potential Challenges

1. **Array Format JSONL**: Dataset uses JSON arrays instead of JSON objects - requires custom parsing
2. **Empty Options**: Some options are empty strings - need filtering and reindexing logic
3. **Variable Options**: While sampled to 4, some questions have fewer valid options
4. **Answer Parsing**: Models may return various formats (letters, numbers, full text)
5. **Local Path Dependency**: Current implementation relies on `/storage/disk3/datasets/SUTD-TrafficQA`

## Implementation Considerations

### Variable Option Positioning (Critical!)

**Key Discovery**: 42.3% of entries have options NOT starting from option0 or are non-contiguous.

**Examples**:
- option1, option3 filled → displayed as "A. option1", "B. option3" → model answers A/B → need to map back to indices 1/3
- option2, option3 filled → displayed as "A. option2", "B. option3" → model answers A/B → need to map back to indices 2/3

**Solution**: Store bidirectional mapping during prompt building:
- Display letters (A, B, C, D) → Original indices (0, 1, 2, 3)
- When model answers "B", look up original index from mapping

**Do NOT reindex options** - this would invalidate all existing answer indices!

### TSV Storage Approach

For TSV format, we have two options:

**Option 1: Preserve original structure** (recommended)
- Keep all 4 columns (A, B, C, D) even if empty
- Answer index stays as-is (0-3)
- Pros: No data loss, simple conversion
- Cons: Empty cells in TSV

**Option 2: Compact storage**
- Filter empty options, store only valid ones
- Store mapping separately
- Pros: More compact
- Cons: More complex, need additional metadata

### Local vs. Remote Dataset

For production use, consider:
1. Check local path first (`/storage/disk3/datasets/SUTD-TrafficQA`)
2. Fallback to downloading from GitHub/HuggingFace
3. Use environment variable for dataset root path

## Summary of Key Findings

### Dataset Characteristics
| Aspect | Finding | Implication |
|--------|---------|-------------|
| **Format** | JSONL arrays (not objects) | Custom parsing required |
| **Options** | Variable positions, 42.3% non-standard | Bidirectional mapping needed |
| **Answer Index** | Always valid, refers to original position | Do NOT reindex |
| **Perspectives** | Only 1 and 3 used (perspective 2 unused) | Document as 2-view dataset |
| **Splits** | 99.5% train/test video overlap | Use test split only for eval |
| **Balance** | ~25% per answer option, well-balanced | Good for evaluation |

### Option Position Distribution (11 patterns)
```
option0,option1,option2,option3: 30,092 (48.1%)
option0,option1:                   4,420 ( 7.1%)
option1,option2:                   4,393 ( 7.0%)
option1,option3:                   4,347 ( 7.0%)
option2,option3:                   4,319 ( 6.9%)
option0,option2:                   4,318 ( 6.9%)
option0,option3:                   4,280 ( 6.8%)
[3-option combinations]:           6,364 (10.2%)
```

### Implementation Checklist
- [ ] Parse JSONL array format correctly
- [ ] Handle variable option positioning with bidirectional mapping
- [ ] Use test split only (document overlap issue)
- [ ] Support perspective filtering (1 and 3 only)
- [ ] Implement per-reasoning-task metrics
- [ ] Add local dataset path check with fallback
- [ ] Create frame extraction logic for traffic videos
- [ ] Handle empty options in TSV storage
- [ ] Test with video LLM and frame-based models

## References

- **Paper**: Xu, L., Huang, H., & Liu, J. (2021). "SUTD-TrafficQA: A Question Answering Benchmark and an Efficient Network for Video Reasoning over Traffic Events." CVPR 2021.
  - [arXiv](https://arxiv.org/abs/2103.15538)
  - [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_SUTD-TrafficQA_A_Question_Answering_Benchmark_and_an_Efficient_Network_for_CVPR_2021_paper.pdf)
- **GitHub**: https://github.com/sutdcv/SUTD-TrafficQA
- **Project Page**: https://sutdcv.github.io/SUTD-TrafficQA/
- VLMEvalKit Video Base: `/storage/disk0/arda/VLMEvalKit/vlmeval/dataset/video_base.py`
- Example Implementation: `/storage/disk0/arda/VLMEvalKit/vlmeval/dataset/video_mmlu.py`
