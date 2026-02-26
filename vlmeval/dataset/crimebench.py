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

FAIL_MSG = "Failed to obtain answer via API."

# Question type names for display
QUESTION_TYPES = [
    "entity_recognition",
    "crime_classification",
    "trajectory_forecasting",
    "intent_prediction",
    "counterfactual_adjacent",
    "degraded_input",
    "counterfactual_long_chain",
    "impossible_scenarios",
    "anomaly_classification",
    "guardrail_consistency",
    "rule_violation",
    "spatial_grounding",
    "omission_causality",
    "multi_event_causal",
    "temporal_localization",
    "counting",
    "action_sequence",
    "negative_control",
]

# Crime type names for display
CRIME_TYPES = [
    "Normal",
    "Explosion",
    "Shooting",
    "RoadAccidents",
    "Shoplifting",
    "Riot",
    "Fighting",
    "Abuse",
    "Burglary",
    "Car accident",
    "Arson",
    "Arrest",
    "Robbery",
    "Vandalism",
    "Stealing",
    "Assault",
    "Violence",
    "Privacy violation",
    "Physical altercation",
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

    TYPE = "Video-MCQ"
    MODALITY = "VIDEO"

    DATASET_PATH = "/home/ubuntu/data/crimebench/crimebench_export"

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

    def __init__(self, dataset="CrimeBench", nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset

    def prepare_dataset(self, dataset_name="CrimeBench"):
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
        data_path = os.environ.get("CRIMEBENCH_DATA_PATH", self.DATASET_PATH)

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"CrimeBench dataset not found at {data_path}. "
                f"Please set CRIMEBENCH_DATA_PATH environment variable or ensure dataset is at {self.DATASET_PATH}."
            )

        parquet_dir = os.path.join(data_path, "crimebench_v2_final.parquet")
        arrow_file = os.path.join(parquet_dir, "data-00000-of-00001.arrow")

        if not os.path.exists(arrow_file):
            raise FileNotFoundError(
                f"Arrow file not found: {arrow_file}. "
                f"Please ensure crimebench_v2_final.parquet directory exists."
            )

        tsv_filename = f"{dataset_name}.tsv"
        tsv_path = os.path.join(data_path, tsv_filename)

        if not os.path.exists(tsv_path) or self._needs_regeneration(
            arrow_file, tsv_path
        ):
            self._generate_tsv_from_arrow(arrow_file, tsv_path, data_path)

        video_root = os.path.join(data_path, "videos")

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

        with open(arrow_file, "rb") as f:
            data = f.read()

        reader = pa.ipc.open_stream(io.BytesIO(data))
        table = reader.read_all()
        df = table.to_pandas()

        data_rows = []
        video_files = set()

        for idx, row in df.iterrows():
            video_path = row["video_path"]
            video_files.add(video_path)

            candidates = row["candidates"]
            if isinstance(candidates, np.ndarray):
                candidates = candidates.tolist()

            answer_idx = int(row["answer"])
            answer_letter = chr(65 + answer_idx)

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

            data_rows.append(tsv_row)

        result_df = pd.DataFrame(data_rows)
        result_df.to_csv(tsv_path, sep="\t", index=False)

        print(f"Generated TSV file: {tsv_path}")
        print(f"Total questions: {len(result_df)}")
        print(f"Unique videos: {len(video_files)}")

        if "tier" in result_df.columns:
            tier_counts = result_df["tier"].value_counts().sort_index()
            print(f"Tier distribution:")
            for tier, count in tier_counts.items():
                print(f"  Tier {tier}: {count} ({count / len(result_df) * 100:.1f}%)")

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
            message = [dict(type="text", value=self.FRAMES_TMPL_SYS_4VIDEO_LLM.strip())]
            video_path = os.path.normpath(
                os.path.join(self.data_root, line["video_path"])
            )
            message.append(dict(type="video", value=video_path))
        else:
            frame_paths = self.save_video_frames(line["video"] + ".mp4")
            message = [
                dict(
                    type="text",
                    value=self.FRAMES_TMPL_SYS.strip(),
                )
            ]
            for frame_path in frame_paths:
                message.append(dict(type="image", value=frame_path))

        question_prompt = self.QUESTION_TMPL.format(
            line["question"], line["A"], line["B"], line["C"], line["D"]
        ).strip()
        message.append(dict(type="text", value=question_prompt))

        return message

    @classmethod
    def supported_datasets(cls):
        return ["CrimeBench"]
