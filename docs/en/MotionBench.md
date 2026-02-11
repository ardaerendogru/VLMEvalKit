# MotionBench Dataset

MotionBench is a CVPR 2025 benchmark for evaluating fine-grained video motion understanding in Vision-Language Models (VLMs). It features 6 core capabilities with 8,052 questions across diverse video sources including industrial, sports, gaming, and medical videos.

## Dataset Download

MotionBench is available on HuggingFace and will be automatically downloaded when you first run an evaluation:

**HuggingFace Repository**: [THUDM/MotionBench](https://huggingface.co/datasets/THUDM/MotionBench)

```bash
# The dataset will be auto-downloaded to HuggingFace cache
# No manual download required
python run.py --data MotionBench_8frame --model <model_name>
```

## Dataset Overview

| Property | Value |
|----------|-------|
| **Paper** | [MotionBench: Benchmarking and Improving Fine-grained Video Motion Understanding](https://arxiv.org/abs/2501.02955) (CVPR 2025) |
| **GitHub** | [zai-org/MotionBench](https://github.com/zai-org/MotionBench) |
| **Task** | Video Question Answering (Multiple Choice) |
| **Format** | Video + Text Question → Option Letter (A/B/C/D) |
| **Videos** | ~5,385 videos from diverse sources |
| **Questions** | 8,052 QA pairs total |
| **DEV Set** | ~4,018 QA pairs (with ground truth) |
| **TEST Set** | ~4,034 QA pairs (for leaderboard submission) |
| **License** | CC-BY-NC-SA-4.0 |

## Key Dataset Characteristics

### 1. Six Core Capabilities

MotionBench evaluates models on six distinct aspects of motion understanding:

| Question Type | # Questions | Description |
|---------------|-------------|-------------|
| **Motion Recognition** | 2,944 | Identify the motion or action being performed in the video |
| **Motion-related Objects** | 1,415 | Identify objects involved in or affected by motion |
| **Location-related Motion** | 1,143 | Track and describe spatial movement patterns |
| **Action Order** | 1,001 | Determine the temporal sequence of multiple actions |
| **Camera Motion** | 775 | Analyze camera movements and focus changes |
| **Repetition Count** | 774 | Count the number of times a motion is repeated |

### 2. Diverse Video Sources

Videos come from multiple domains to ensure comprehensive evaluation:

- **Self-collected videos** (~5,351 videos): Videos collected and annotated by the MotionBench team
- **Public datasets**: Clips from existing benchmarks including:
  - MedVid: Medical procedure videos
  - SportsSloMo: Sports slow-motion videos
  - HA-ViD: Human-Activity videos

### 3. Fine-grained Motion Focus

Unlike general VideoQA benchmarks, MotionBench specifically targets:
- **Fine-grained motion comprehension** at the frame level
- **Dynamic information processing** with annotation density of 12.63 words per second
- **Temporal understanding** across various time scales (sub-second to multi-second)

## Usage

### Basic Evaluation

```bash
# Evaluate with 8 frames (default)
python run.py --data MotionBench_8frame --model Qwen2.5-VL-7B-Instruct

# Evaluate with fps-based sampling
python run.py --data MotionBench_1fps --model InternVL2-Llama3-76B

# Evaluate with more frames for better temporal understanding
python run.py --data MotionBench_64frame --model <model_name>
```

### Frame Sampling Options

**Fixed Frame Count:**
- `MotionBench_8frame` - 8 frames uniformly sampled
- `MotionBench_16frame` - 16 frames uniformly sampled
- `MotionBench_32frame` - 32 frames uniformly sampled
- `MotionBench_64frame` - 64 frames uniformly sampled

**FPS-based Sampling:**
- `MotionBench_2fps` - 2 frames per second
- `MotionBench_1fps` - 1 frame per second
- `MotionBench_0.5fps` - 0.5 frames per second
- `MotionBench_0.25fps` - 0.25 frames per second (for long videos)

### Distributed Evaluation

```bash
# Run with 8 GPUs
torchrun --nproc-per-node=8 run.py --data MotionBench_16frame --model Qwen2-VL-72B-Instruct
```

### Inference Only (No Evaluation)

```bash
# Generate predictions without evaluation
python run.py --data MotionBench_8frame --model <model_name> --mode infer
```

## Output Format

The evaluation produces results with:

1. **Overall Accuracy** - Performance across all questions with ground truth

2. **Per-Question-Type Accuracy** - Performance for each of the 6 motion understanding categories:
   - `Motion Recognition`
   - `Motion-related Objects`
   - `Location-related Motion`
   - `Action Order`
   - `Camera Motion`
   - `Repetition Count`

Example output:
```
MotionBench Evaluation Results:

Question Type              | Success | Overall | Accuracy
---------------------------|---------|---------|--------
Motion Recognition         | 1200    | 2944    | 40.75%
Motion-related Objects     | 580     | 1415    | 41.00%
Location-related Motion    | 450     | 1143    | 39.37%
Action Order               | 400     | 1001    | 39.96%
Camera Motion              | 300     | 775     | 38.71%
Repetition Count           | 310     | 774     | 40.05%
---------------------------|---------|---------|--------
Overall                    | 3240    | 8052    | 40.24%
```

**Note**: Samples with `answer: "NA"` (TEST set) are automatically excluded from evaluation.

## Implementation Details

### File Location

- Dataset implementation: `vlmeval/dataset/motionbench.py`
- Configuration: `vlmeval/dataset/video_dataset_config.py`

### Key Classes

```python
class MotionBench(VideoBaseDataset):
    TYPE = 'Video-MCQ'

    def __init__(self, dataset='MotionBench', nframe=8, fps=-1):
        # nframe: fixed number of frames (mutually exclusive with fps)
        # fps: frames per second for sampling (mutually exclusive with nframe)
```

### Data Format

The dataset is stored in JSONL format with the following structure:

```json
{
  "question_type": "Motion Recognition",
  "video_type": "industrial",
  "key": "unique_video_id",
  "qa": [{
    "uid": "unique_question_id",
    "start": null,
    "end": null,
    "answer": "C",
    "question": "What is the boy holding in his hand?\nA. Screw\nB. Fountain pen\nC. Screwdriver\nD. Wrench"
  }],
  "video_path": "video_filename.mp4",
  "video_info": {
    "duration": 8.38,
    "fps": 60.0,
    "resolution": {"width": 1280, "height": 1280}
  }
}
```

The implementation automatically:
1. Downloads the dataset from HuggingFace
2. Converts JSONL to TSV format for VLMEvalKit
3. Handles video paths for both self-collected and public dataset videos
4. Filters out NA answers during evaluation

### Prompt Format

The system prompt emphasizes fine-grained motion observation:

```
These are frames from a video showing various motions and actions.
Carefully observe the fine-grained motion details in the frames.
Select the best answer to the following multiple-choice question.
Respond with only the letter (A, B, C, or D) of the correct option.
```

## Leaderboard Submission

For samples without ground truth (TEST set), you can submit results to the official MotionBench leaderboard:

1. **Official Leaderboard**: [motion-bench.github.io](https://motion-bench.github.io/#leaderboard)
2. **HuggingFace Leaderboard**: [THUDM/MotionBench Leaderboard](https://huggingface.co/spaces/THUDM/MotionBench)

Generate predictions:
```bash
# Run inference on TEST set (NA answers are included in full dataset)
python run.py --data MotionBench_8frame --model <model_name> --mode infer
```

Format your predictions according to the leaderboard requirements and submit.

## Citation

If you use MotionBench in your research, please cite:

```bibtex
@misc{hong2024motionbench,
      title={MotionBench: Benchmarking and Improving Fine-grained Video Motion Understanding for Vision Language Models},
      author={Wenyi Hong and Yean Cheng and Zhuoyi Yang and Weihan Wang and Lefan Wang and Xiaotao Gu and Shiyu Huang and Yuxiao Dong and Jie Tang},
      year={2024},
      eprint={2501.02955},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Notes

- The dataset will be automatically downloaded from HuggingFace on first use
- Samples with `answer: "NA"` are from the TEST set and excluded from evaluation
- Per-question-type metrics help identify model strengths/weaknesses
- MotionBench requires careful frame-by-frame analysis for best results
- Consider using higher frame counts (32-64) for challenging motion types
