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

    @classmethod
    def supported_datasets(cls):
        return ["CrimeBench"]
