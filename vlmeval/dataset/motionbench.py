# flake8: noqa
"""
MotionBench Dataset Implementation for VLMEvalKit

MotionBench is a CVPR 2025 benchmark for fine-grained video motion understanding.
Features 6 core capabilities with questions from both public datasets and self-collected data.

Paper: https://arxiv.org/abs/2501.02955
GitHub: https://github.com/zai-org/MotionBench
HuggingFace: https://huggingface.co/datasets/THUDM/MotionBench

*** DATASET DOWNLOAD ***
Dataset is available on HuggingFace: https://huggingface.co/datasets/THUDM/MotionBench
The dataset will be automatically downloaded via huggingface_hub.

Question Types:
1. Motion Recognition - Identify the motion/action in video
2. Motion-related Objects - Identify objects involved in motion
3. Location-related Motion - Track spatial movement patterns
4. Action Order - Determine temporal sequence of actions
5. Camera Motion - Analyze camera movement
6. Repetition Count - Count repeated motions

Dataset Statistics:
- Public dataset: Videos from MedVid, SportsSloMo, HA-ViD
- Self-collected: Videos collected specifically for MotionBench
"""

import ast
import os
import json
import re
import sys
from huggingface_hub import snapshot_download
from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE

FAIL_MSG = 'Failed to obtain answer via API.'


def verbose_print(msg, flush=True):
    """Print verbose message to stderr for immediate output."""
    print(f'[MotionBench] {msg}', file=sys.stderr, flush=flush)


# Question type descriptions for reference
QUESTION_TYPE_DESCRIPTIONS = {
    'Motion Recognition': 'Identify the motion or action being performed in the video',
    'Motion-related Objects': 'Identify objects that are involved in or affected by motion',
    'Location-related Motion': 'Track and describe spatial movement patterns',
    'Action Order': 'Determine the temporal sequence of multiple actions',
    'Camera Motion': 'Analyze camera movements and focus changes',
    'Repetition Count': 'Count the number of times a motion is repeated'
}


class MotionBench(VideoBaseDataset):
    """
    MotionBench Dataset Implementation

    Args:
        dataset: Dataset name (default: 'MotionBench')
        nframe: Number of frames to sample (mutually exclusive with fps)
        fps: Frames per second for sampling (mutually exclusive with nframe)

    Dataset Statistics:
        - Question types: 6 categories
        - Video sources: Public datasets (MedVid, SportsSloMo, HA-ViD) + Self-collected

    The dataset uses MCQ format with 4 options (A, B, C, D).
    Samples with 'NA' as answer are from the TEST set (no ground truth available).
    """

    MD5 = ''  # Will be computed after TSV generation
    TYPE = 'Video-MCQ'

    FRAMES_TMPL = """
These are frames from a video showing various motions and actions.
Carefully observe the fine-grained motion details in the frames.
Select the best answer to the following multiple-choice question.
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    def __init__(self, dataset='MotionBench', nframe=8, fps=-1, preload_videos=True):
        verbose_print(f'Initializing MotionBench: nframe={nframe}, fps={fps}')
        # When fps is specified, nframe should be 0 to avoid conflict
        if fps > 0:
            nframe = 0
        self._preload_videos = preload_videos
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset

        # Pre-download all videos to avoid downloading during evaluation
        if self._preload_videos:
            self.preload_all_videos()

        verbose_print(f'MotionBench initialized with {len(self.data)} samples')

    @classmethod
    def supported_datasets(cls):
        return ['MotionBench']

    def prepare_dataset(self, dataset_name='MotionBench', repo_id='THUDM/MotionBench'):
        """
        Prepare MotionBench dataset from HuggingFace repository.

        Downloads the dataset and converts JSONL format to TSV format.

        Returns:
            dict with 'root' (video directory) and 'data_file' (TSV file path)
        """

        def check_integrity(pth):
            """Check if the TSV file already exists and contains videos from both sources."""
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not osp.exists(data_file):
                verbose_print('TSV file does not exist')
                return False

            # Load TSV and check it has videos from both self-collected and public-dataset
            data = load(data_file)

            # Verify both video sources are present
            video_prefixes = set(data['video_prefix'].unique())
            required_prefixes = {'./self-collected/', './public-dataset/'}

            if not required_prefixes.issubset(video_prefixes):
                verbose_print(f'TSV missing video sources. Has: {video_prefixes}, needs: {required_prefixes}')
                return False

            # Count videos from each source
            self_collected_count = len(data[data['video_prefix'] == './self-collected/'])
            public_count = len(data[data['video_prefix'] == './public-dataset/'])

            verbose_print(f'TSV integrity check: self-collected={self_collected_count}, public-dataset={public_count}')

            # Both sources should have videos
            if self_collected_count == 0 or public_count == 0:
                verbose_print('TSV is missing one of the video sources')
                return False

            return True

        def check_video_integrity(pth):
            """Check if video directories are present."""
            has_public_dataset = osp.isdir(osp.join(pth, 'public-dataset'))
            has_self_collected = osp.isdir(osp.join(pth, 'self-collected'))
            return has_public_dataset or has_self_collected

        def generate_tsv(pth):
            """
            Convert MotionBench JSONL to TSV format.

            Original JSONL format:
            {
                "question_type": "Motion Recognition",
                "video_type": "industrial" | null,
                "key": "unique_video_id",
                "qa": [{
                    "uid": "unique_question_id",
                    "start": start_second | null,
                    "end": end_second | null,
                    "answer": "A"|"B"|"C"|"D"|"NA",
                    "question": "Question text with options..."
                }],
                "video_path": "video_filename.mp4",
                "video_info": {"duration": ..., "fps": ..., "resolution": {...}}
            }

            TSV format:
            index | question_type | video_type | video | video_prefix | video_suffix |
            question | options | answer | uid | ...
            """
            # The MotionBench dataset has files in a MotionBench/ subdirectory
            jsonl_file = osp.join(pth, 'MotionBench', 'video_info.meta.jsonl')
            tsv_file = osp.join(pth, f'{dataset_name}.tsv')

            # TSV generation is controlled by check_integrity() which verifies both video sources
            verbose_print(f'Generating TSV from: {jsonl_file}')
            logging.info(f'Generating MotionBench TSV from {jsonl_file}')

            verbose_print(f'Reading JSONL from: {jsonl_file}')
            logging.info(f'Generating MotionBench TSV from {jsonl_file}')

            data_list = []
            public_count = 0
            self_collected_count = 0

            with open(jsonl_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                verbose_print(f'Total lines in JSONL: {len(lines)}')

                for line_num, line in enumerate(lines, 1):
                    if line_num % 500 == 0:
                        verbose_print(f'Processing line {line_num}/{len(lines)}')

                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        logging.warning(f'Failed to parse line {line_num}: {e}')
                        continue

                    # Extract basic info
                    question_type = data.get('question_type', '')
                    video_type = data.get('video_type') or 'None'
                    key = data.get('key', '')
                    video_path = data.get('video_path', '')

                    # Determine video prefix and suffix
                    # Videos are in either 'self-collected/' or 'public-dataset/'
                    # public-dataset videos have pattern: {uuid}_{start}_{end}.mp4
                    # self-collected videos have pattern: {hex_hash}.mp4
                    # Check if video_path matches {uuid}_{number}_{number}.mp4 pattern
                    if re.search(r'_[\d]+_[\d]+\.mp4$', video_path):
                        # These are pre-cut clips from public datasets
                        video_prefix = './public-dataset/'
                        public_count += 1
                    else:
                        # Self-collected videos
                        video_prefix = './self-collected/'
                        self_collected_count += 1

                    # Extract filename without extension as video ID
                    video_id = video_path.replace('.mp4', '')
                    video_suffix = '.mp4'

                    # Process QA pairs (usually one per video)
                    for qa_item in data.get('qa', []):
                        uid = qa_item.get('uid', '')
                        answer = qa_item.get('answer', 'NA')
                        # Normalize missing/empty answer to NA (test set)
                        if answer is None or (isinstance(answer, str) and not answer.strip()):
                            answer = 'NA'
                        
                        # Skip TEST set samples - don't waste compute
                        if answer == 'NA':
                            continue

                        question_text = qa_item.get('question', '')

                        # Parse question to separate main question from options
                        # The question format is: "Main question?\nA. Option 1\nB. Option 2\n..."
                        lines = question_text.strip().split('\n')
                        main_question = lines[0].strip()

                        # Skip shitty data: empty question
                        if not main_question:
                            verbose_print(f'Skipping sample {uid}: empty question')
                            continue

                        # Extract options (lines starting with A., B., C., D.)
                        options = []
                        for line in lines[1:]:
                            line = line.strip()
                            if line and len(line) > 2 and line[1] == '.':
                                options.append(line)

                        # If no options found, try to parse differently
                        if not options:
                            # Fallback: split by known patterns
                            for opt_letter in ['A.', 'B.', 'C.', 'D.']:
                                idx = question_text.find(opt_letter)
                                if idx != -1:
                                    next_idx = question_text.find('\n', idx)
                                    if next_idx == -1:
                                        opt_text = question_text[idx:].strip()
                                    else:
                                        opt_text = question_text[idx:next_idx].strip()
                                    if opt_text:
                                        options.append(opt_text)

                        # Build list of valid option letters (only options with non-empty content after "X. ")
                        # Excludes weird options like "D. " (empty) so GT never points to them
                        valid_options = [
                            opt[0] for opt in options
                            if len(opt) > 2 and opt[1] == '.' and opt[2:].strip()
                        ]

                        # For labeled (non-NA) answers, require GT to point to an option with real content
                        if answer != 'NA' and answer not in valid_options:
                            verbose_print(f'Skipping sample {uid}: GT={answer} but only {valid_options} exist')
                            continue

                        # Skip if no valid options (malformed QA)
                        if not valid_options:
                            verbose_print(f'Skipping sample {uid}: no valid options')
                            continue

                        # Keep variable option count (2, 3, or 4 options - all valid)

                        data_list.append({
                            'question_type': question_type,
                            'video_type': video_type,
                            'video': video_id,
                            'video_prefix': video_prefix,
                            'video_suffix': video_suffix,
                            'question': main_question,
                            'options': str(options),
                            'answer': answer,
                            'uid': uid,
                            'key': key
                        })

            # Create DataFrame and save as TSV
            data_df = pd.DataFrame(data_list)
            data_df = data_df.assign(index=range(len(data_df)))

            # Reorder columns for better readability
            columns = ['index', 'question_type', 'video_type', 'video', 'video_prefix',
                      'video_suffix', 'question', 'options', 'answer', 'uid', 'key']
            data_df = data_df[[col for col in columns if col in data_df.columns]]

            verbose_print(f'Saving TSV with {len(data_df)} samples (public: {public_count}, self-collected: {self_collected_count})')
            data_df.to_csv(tsv_file, sep='\t', index=False)

            # Update MD5
            self.MD5 = md5(tsv_file)
            logging.info(f'Generated MotionBench TSV with {len(data_df)} samples: {tsv_file}')
            logging.info(f'MD5: {self.MD5}')
            verbose_print(f'TSV saved successfully: {tsv_file}')

        # Try cache first
        verbose_print(f'Looking for cached dataset: {repo_id}')
        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            verbose_print(f'Using cached dataset at: {cache_path}')
            dataset_path = cache_path
        else:
            # Download from HuggingFace - only metadata, videos downloaded on-demand
            verbose_print(f'Downloading MotionBench metadata from {repo_id}')
            logging.info(f'Downloading MotionBench metadata from {repo_id}')
            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                # Only download the metadata file and directory structure
                # Videos will be downloaded on-demand to avoid timeout
                from huggingface_hub import snapshot_download
                verbose_print(f'Calling snapshot_download with allow_patterns...')
                dataset_path = snapshot_download(
                    repo_id=repo_id,
                    repo_type='dataset',
                    allow_patterns=['MotionBench/video_info.meta.jsonl', '*.md', '.gitattributes']
                )
                verbose_print(f'Download complete, path: {dataset_path}')

            # Generate TSV if needed
            verbose_print('Generating TSV...')
            generate_tsv(dataset_path)

        # The actual data is in a MotionBench/ subdirectory
        # Update dataset_path to point to that subdirectory
        motionbench_path = osp.join(dataset_path, 'MotionBench')
        if osp.isdir(motionbench_path):
            dataset_path = motionbench_path
        # TSV is at parent level, not in MotionBench/ subdirectory
        data_file = osp.join(osp.dirname(dataset_path), f'{dataset_name}.tsv')
        verbose_print(f'Dataset prepared: root={dataset_path}, data_file={data_file}')
        return dict(root=dataset_path, data_file=data_file)

    def preload_all_videos(self):
        """
        Pre-download all videos from HuggingFace before evaluation starts.

        This prevents the slow on-demand downloading during evaluation.
        Downloads videos from both self-collected and public-dataset directories.
        """
        from huggingface_hub import hf_hub_download
        from concurrent.futures import ThreadPoolExecutor, as_completed

        verbose_print('Starting video pre-download...')

        # Get unique videos from the dataset
        unique_videos = self.data[['video', 'video_prefix', 'video_suffix']].drop_duplicates()
        total_videos = len(unique_videos)
        verbose_print(f'Found {total_videos} unique videos to download')

        downloaded = 0
        skipped = 0
        failed = 0

        def download_video(row):
            video = row['video']
            video_prefix = row['video_prefix'].lstrip('./')
            video_suffix = row['video_suffix']
            vid_path = osp.join(self.data_root, video_prefix, video + video_suffix)

            if osp.exists(vid_path):
                return ('skipped', video)

            prefix_clean = video_prefix.strip('./')
            hf_path = f'MotionBench/{prefix_clean}/{video}.mp4'

            try:
                downloaded_path = hf_hub_download('THUDM/MotionBench', hf_path, repo_type='dataset')
                if osp.abspath(downloaded_path) != osp.abspath(vid_path):
                    os.makedirs(osp.dirname(vid_path), exist_ok=True)
                    if osp.exists(vid_path) or osp.islink(vid_path):
                        os.remove(vid_path)
                    os.symlink(downloaded_path, vid_path)
                return ('downloaded', video)
            except Exception as e:
                return ('failed', video, str(e))

        # Use thread pool for parallel downloads
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(download_video, row): row['video'] for _, row in unique_videos.iterrows()}

            for future in as_completed(futures):
                result = future.result()
                if result[0] == 'downloaded':
                    downloaded += 1
                elif result[0] == 'skipped':
                    skipped += 1
                else:
                    failed += 1
                    verbose_print(f'Failed to download {result[1]}: {result[2]}')

                # Progress update every 50 videos
                total_done = downloaded + skipped + failed
                if total_done % 50 == 0 or total_done == total_videos:
                    verbose_print(f'Progress: {total_done}/{total_videos} (downloaded={downloaded}, skipped={skipped}, failed={failed})')

        verbose_print(f'Video pre-download complete: downloaded={downloaded}, skipped={skipped}, failed={failed}')

    def save_video_frames(self, video, video_llm=False):
        """
        Extract and save frames from video.

        Downloads videos on-demand from HuggingFace if not present locally.

        Args:
            video: Video identifier
            video_llm: If True, return video path for video LLMs

        Returns:
            frame_paths (list): Paths to extracted frames, or video path if video_llm=True
        """
        verbose_print(f'save_video_frames: video={video}, video_llm={video_llm}')

        # Get video info from data
        video_info = self.data[self.data['video'] == video].iloc[0]
        video_prefix = video_info['video_prefix'].lstrip('./')

        vid_path = osp.join(self.data_root, video_prefix, video + video_info['video_suffix'])
        verbose_print(f'Video path: {vid_path}')

        # Download video on-demand if not present
        if not osp.exists(vid_path):
            verbose_print(f'Video not found locally, downloading from HuggingFace...')
            logging.info(f'Video not found locally, downloading: {video}')
            from huggingface_hub import hf_hub_download
            # Construct the HuggingFace path (strip slashes to avoid double-slash)
            prefix_clean = video_prefix.strip('./')
            hf_path = f'MotionBench/{prefix_clean}/{video}.mp4'
            try:
                verbose_print(f'Downloading: {hf_path}')
                downloaded_path = hf_hub_download('THUDM/MotionBench', hf_path, repo_type='dataset')
                verbose_print(f'Downloaded to: {downloaded_path}')
                # Only create symlink if downloaded path differs from expected path
                # hf_hub_download may already place the file at vid_path
                if osp.abspath(downloaded_path) != osp.abspath(vid_path):
                    # Ensure parent directory exists for creating symlink
                    os.makedirs(osp.dirname(vid_path), exist_ok=True)
                    # Create a symlink to the expected location
                    if osp.exists(vid_path) or osp.islink(vid_path):
                        os.remove(vid_path)
                    os.symlink(downloaded_path, vid_path)
                    verbose_print(f'Created symlink: {vid_path} -> {downloaded_path}')
                else:
                    verbose_print(f'File already at expected location: {vid_path}')
            except Exception as e:
                logging.error(f'Failed to download video {video}: {e}')
                verbose_print(f'ERROR downloading video: {e}')
                raise
        else:
            verbose_print(f'Video exists locally: {vid_path}')

        if video_llm:
            # For video LLMs, return the video path directly
            verbose_print(f'Returning video path for video_llm mode')
            return vid_path

        verbose_print(f'Opening video with decord...')
        import decord
        vid = decord.VideoReader(vid_path)
        video_info_dict = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        verbose_print(f'Video info: fps={video_info_dict["fps"]:.2f}, n_frames={video_info_dict["n_frames"]}')

        # Determine frame sampling strategy
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
            verbose_print(f'Using nframe mode: nframe={self.nframe}, indices={indices[:3]}...')
        elif self.fps > 0:
            total_duration = video_info_dict['n_frames'] / video_info_dict['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info_dict['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))
            verbose_print(f'Using fps mode: fps={self.fps}, frames={len(indices)}')
        else:
            raise ValueError('Either nframe or fps must be specified')

        # Check if frames already exist
        flag = np.all([osp.exists(p) for p in frame_paths])
        if flag:
            verbose_print(f'All {len(frame_paths)} frames already exist, returning cached paths')
            return frame_paths

        verbose_print(f'Extracting {len(indices)} frames from video...')
        # Extract and save frames
        lock_path = osp.splitext(vid_path)[0] + '.lock'
        verbose_print(f'Acquiring lock: {lock_path}')
        with portalocker.Lock(lock_path, 'w', timeout=30):
            if np.all([osp.exists(p) for p in frame_paths]):
                verbose_print(f'Frames created by another process, returning')
                return frame_paths
            verbose_print(f'Reading frames at indices: {indices[:3]}...')
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            verbose_print(f'Saving {len(images)} frames to disk...')
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)
            verbose_print(f'Frames saved successfully')

        return frame_paths

    def build_prompt(self, line, video_llm):
        """
        Build prompt message for the model.

        Args:
            line: Data row (can be int index or dict)
            video_llm: If True, use video input; otherwise use frames

        Returns:
            list of message dicts with interleaved video/frames and text
        """
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
            verbose_print(f'build_prompt: index={line.name}, video={line["video"]}')
        else:
            verbose_print(f'build_prompt: video={line["video"]}')

        # Parse question and options
        question = line['question']
        options = eval(line['options'])  # Convert string back to list

        # Format question with options
        question_text = question
        if not any(opt in question_text for opt in ['A.', 'B.', 'C.', 'D.']):
            # Append options if not already in question
            question_text = question + '\n' + '\n'.join(options)

        message = [dict(type='text', value=self.FRAMES_TMPL)]

        # Add video or frames
        if video_llm:
            verbose_print(f'build_prompt: using video_llm mode')
            # Ensure the video exists locally (download on-demand if needed)
            vid_path = self.save_video_frames(line['video'], video_llm=True)
            message.append(dict(type='video', value=vid_path))
        else:
            verbose_print(f'build_prompt: extracting frames...')
            frame_paths = self.save_video_frames(line['video'], video_llm=False)
            verbose_print(f'build_prompt: got {len(frame_paths)} frames')
            for fp in frame_paths:
                message.append(dict(type='image', value=fp))

        # Add question prompt
        prompt = f'Question: {question_text}\nAnswer with the option letter (A, B, C, or D) of the correct option.'
        message.append(dict(type='text', value=prompt))

        verbose_print(f'build_prompt: built message with {len(message)} parts')
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate MotionBench predictions.

        Calculates accuracy for each question type and overall accuracy.
        Samples with 'NA' as answer (TEST set) are excluded from evaluation.

        Args:
            eval_file: Path to prediction file (xlsx/tsv/json)
            **judge_kwargs: Optional judge model settings

        Returns:
            DataFrame with accuracy results per question_type and overall
        """
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'data file should be a supported format (xlsx/json/tsv)'

        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')

            if model == 'exact_matching':
                # Use exact string matching for evaluation
                data = load(eval_file)
                data = data[data['answer'].notna() & (data['answer'] != 'NA')]  # Exclude TEST set (NA) from accuracy

                if 'prediction' in data.columns:
                    predictions = data['prediction']
                else:
                    raise ValueError(f'Prediction column not found in {eval_file}')

                # Extract answer letter from prediction (robust to "Answer: C" or "C" etc.)
                def extract_answer(pred):
                    if pd.isna(pred):
                        return 'INVALID'
                    s = str(pred).strip().upper()
                    if not s:
                        return 'INVALID'
                    # Single letter A/B/C/D
                    if s[0] in 'ABCD' and (len(s) == 1 or not s[1].isalnum()):
                        return s[0]
                    # Starts with A., B., C., D.
                    for letter in ['A', 'B', 'C', 'D']:
                        if s.startswith(letter + '.') or s.startswith(letter + ')'):
                            return letter
                    # First occurrence of A, B, C, or D (e.g. "The answer is B")
                    match = re.search(r'\b([ABCD])\b', s)
                    if match:
                        return match.group(1)
                    return 'INVALID'

                data['extracted_answer'] = predictions.apply(extract_answer)
                # Normalize GT to uppercase for comparison (TSV may have mixed case)
                gt_normalized = data['answer'].astype(str).str.strip().str.upper()
                data['correct'] = (data['extracted_answer'] == gt_normalized) & (data['extracted_answer'] != 'INVALID')

                # Calculate metrics per question_type
                result_dict = {}

                for qt in data['question_type'].unique():
                    if pd.isna(qt):
                        continue
                    qt_data = data[data['question_type'] == qt]
                    success = qt_data['correct'].sum()
                    overall = len(qt_data)
                    result_dict[f'{qt}'] = {'success': success, 'overall': overall}

                # Overall metrics
                total_success = data['correct'].sum()
                total_overall = len(data)
                result_dict['overall'] = {'success': total_success, 'overall': total_overall}

                # Create result DataFrame (guard division by zero for empty question types)
                result = pd.DataFrame.from_dict(result_dict, orient='index')
                result = result.reset_index().rename(columns={'index': 'question_type'})
                result['acc'] = result.apply(
                    lambda r: round((r['success'] / r['overall'] * 100), 2) if r['overall'] > 0 else 0.0,
                    axis=1
                )

                # Save results
                dump(result, score_file)
            else:
                # Use LLM judge for more flexible evaluation
                model = build_judge(**judge_kwargs)
                if not model.working():
                    logging.warning('Judge model not working, falling back to exact matching')
                    return self.evaluate(eval_file, model='exact_matching')

                data = load(eval_file)
                data = data[data['answer'].notna() & (data['answer'] != 'NA')]  # Exclude TEST set (NA) from accuracy

                # Use judge to evaluate each prediction
                results = []
                for idx, row in data.iterrows():
                    question = row['question']
                    try:
                        options = ast.literal_eval(row['options']) if isinstance(row['options'], str) else row['options']
                    except (ValueError, SyntaxError):
                        options = eval(row['options']) if isinstance(row['options'], str) else row['options']
                    if not isinstance(options, list):
                        options = []
                    prediction = row.get('prediction', '')
                    gt_answer = row['answer']

                    prompt = f"""
Question: {question}
Options:
{chr(10).join(str(o) for o in options)}

Ground Truth: {gt_answer}
Model Prediction: {prediction}

Is the model prediction correct? Respond with only 'yes' or 'no'.
"""
                    response = model.generate(prompt)
                    correct = 'yes' in response.lower()
                    results.append({
                        'question_type': row['question_type'],
                        'correct': correct
                    })

                results_df = pd.DataFrame(results)

                # Calculate metrics (guard division by zero)
                result_dict = {}
                for qt in results_df['question_type'].unique():
                    qt_results = results_df[results_df['question_type'] == qt]
                    success = qt_results['correct'].sum()
                    overall = len(qt_results)
                    result_dict[qt] = {'success': success, 'overall': overall}

                total_success = results_df['correct'].sum()
                total_overall = len(results_df)
                result_dict['overall'] = {'success': total_success, 'overall': total_overall}

                result = pd.DataFrame.from_dict(result_dict, orient='index')
                result = result.reset_index().rename(columns={'index': 'question_type'})
                result['acc'] = result.apply(
                    lambda r: round((r['success'] / r['overall'] * 100), 2) if r['overall'] > 0 else 0.0,
                    axis=1
                )

                dump(result, score_file)
        else:
            result = load(score_file)

        return result
