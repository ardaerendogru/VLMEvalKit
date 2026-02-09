# flake8: noqa
"""
MotionBench Dataset Implementation for VLMEvalKit

MotionBench is a CVPR 2025 benchmark for fine-grained video motion understanding.
Features 6 core capabilities with 8,052 questions across diverse video sources.

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
- Total: 8,052 questions
- With ground truth: ~4,018 (DEV set for validation)
- Without ground truth: ~4,034 (TEST set for leaderboard submission)
"""

import os
import json
from huggingface_hub import snapshot_download
from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE

FAIL_MSG = 'Failed to obtain answer via API.'


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
        - Total questions: 8,052
        - Question types: 6 categories
        - Video sources: Self-collected + Public datasets (MedVid, SportsSloMo, HA-ViD)

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

    def __init__(self, dataset='MotionBench', nframe=8, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset

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
            """Check if the TSV file already exists and is valid."""
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not osp.exists(data_file):
                return False

            # Check if we have video files
            data = load(data_file)
            for idx, row in data.iterrows():
                video_path = osp.join(pth, row.get('video_prefix', ''), row['video'] + row.get('video_suffix', '.mp4'))
                if not osp.exists(video_path):
                    # Videos might not be downloaded yet, that's ok
                    # We'll check again during frame extraction
                    pass
            return True

        def check_video_integrity(pth):
            """Check if all videos are present."""
            has_self_collected = osp.isdir(osp.join(pth, 'self-collected'))
            has_public_dataset = osp.isdir(osp.join(pth, 'public-dataset'))
            return has_self_collected or has_public_dataset

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
            jsonl_file = osp.join(pth, 'video_info.meta.jsonl')
            tsv_file = osp.join(pth, f'{dataset_name}.tsv')

            if osp.exists(tsv_file):
                existing_md5 = md5(tsv_file)
                if existing_md5 == self.MD5:
                    logging.info(f'MotionBench TSV already exists and matches MD5: {tsv_file}')
                    return

            logging.info(f'Generating MotionBench TSV from {jsonl_file}')

            data_list = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
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
                    if '_0_' in video_path or '_1_' in video_path or '_2_' in video_path:
                        # These are pre-cut clips from public datasets
                        video_prefix = './public-dataset/'
                    else:
                        video_prefix = './self-collected/'

                    # Extract filename without extension as video ID
                    video_id = video_path.replace('.mp4', '')
                    video_suffix = '.mp4'

                    # Process QA pairs (usually one per video)
                    for qa_item in data.get('qa', []):
                        uid = qa_item.get('uid', '')
                        answer = qa_item.get('answer', 'NA')
                        question_text = qa_item.get('question', '')

                        # Parse question to separate main question from options
                        # The question format is: "Main question?\nA. Option 1\nB. Option 2\n..."
                        lines = question_text.strip().split('\n')
                        main_question = lines[0].strip()

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

                        # Ensure we have exactly 4 options
                        while len(options) < 4:
                            options.append(f'{chr(65 + len(options))}. ')
                        options = options[:4]

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

            data_df.to_csv(tsv_file, sep='\t', index=False)

            # Update MD5
            self.MD5 = md5(tsv_file)
            logging.info(f'Generated MotionBench TSV with {len(data_df)} samples: {tsv_file}')
            logging.info(f'MD5: {self.MD5}')

        # Try cache first
        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            # Download from HuggingFace
            logging.info(f'Downloading MotionBench from {repo_id}')
            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            # Generate TSV if needed
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def save_video_frames(self, video, video_llm=False):
        """
        Extract and save frames from video.

        Handles videos in both 'self-collected/' and 'public-dataset/' directories.

        Args:
            video: Video identifier
            video_llm: If True, return video path for video LLMs

        Returns:
            frame_paths (list): Paths to extracted frames, or video path if video_llm=True
        """
        if video_llm:
            # For video LLMs, return the video path directly
            video_info = self.data[self.data['video'] == video].iloc[0]
            vid_path = osp.join(self.data_root, video_info['video_prefix'].lstrip('./'),
                              video + video_info['video_suffix'])
            return vid_path

        # Get video path from data
        video_info = self.data[self.data['video'] == video].iloc[0]
        vid_path = osp.join(self.data_root, video_info['video_prefix'].lstrip('./'),
                          video + video_info['video_suffix'])

        import decord
        vid = decord.VideoReader(vid_path)
        video_info_dict = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }

        # Determine frame sampling strategy
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            total_duration = video_info_dict['n_frames'] / video_info_dict['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info_dict['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))
        else:
            raise ValueError('Either nframe or fps must be specified')

        # Check if frames already exist
        flag = np.all([osp.exists(p) for p in frame_paths])
        if flag:
            return frame_paths

        # Extract and save frames
        lock_path = osp.splitext(vid_path)[0] + '.lock'
        with portalocker.Lock(lock_path, 'w', timeout=30):
            if np.all([osp.exists(p) for p in frame_paths]):
                return frame_paths
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

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
            vid_path = osp.join(self.data_root, line['video_prefix'].lstrip('./'),
                              line['video'] + line['video_suffix'])
            message.append(dict(type='video', value=vid_path))
        else:
            frame_paths = self.save_video_frames(line['video'], video_llm=False)
            for fp in frame_paths:
                message.append(dict(type='image', value=fp))

        # Add question prompt
        prompt = f'Question: {question_text}\nAnswer with the option letter (A, B, C, or D) of the correct option.'
        message.append(dict(type='text', value=prompt))

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
                data = data[data['answer'] != 'NA']  # Filter out TEST set samples

                if 'prediction' in data.columns:
                    predictions = data['prediction']
                else:
                    raise ValueError(f'Prediction column not found in {eval_file}')

                # Extract answer letter from prediction
                def extract_answer(pred):
                    if pd.isna(pred):
                        return 'INVALID'
                    pred = str(pred).strip().upper()
                    # Extract first letter A, B, C, or D
                    for letter in ['A', 'B', 'C', 'D']:
                        if pred.startswith(letter):
                            return letter
                    return 'INVALID'

                data['extracted_answer'] = predictions.apply(extract_answer)
                data['correct'] = (data['extracted_answer'] == data['answer']) & (data['extracted_answer'] != 'INVALID')

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

                # Create result DataFrame
                result = pd.DataFrame.from_dict(result_dict, orient='index')
                result = result.reset_index().rename(columns={'index': 'question_type'})
                result['acc'] = (result['success'] / result['overall'] * 100).round(2)

                # Save results
                dump(result, score_file)
            else:
                # Use LLM judge for more flexible evaluation
                model = build_judge(**judge_kwargs)
                if not model.working():
                    logging.warning('Judge model not working, falling back to exact matching')
                    return self.evaluate(eval_file, model='exact_matching')

                data = load(eval_file)
                data = data[data['answer'] != 'NA']

                # Use judge to evaluate each prediction
                results = []
                for idx, row in data.iterrows():
                    question = row['question']
                    options = eval(row['options'])
                    prediction = row.get('prediction', '')
                    gt_answer = row['answer']

                    prompt = f"""
Question: {question}
Options:
{chr(10).join(options)}

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

                # Calculate metrics
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
                result['acc'] = (result['success'] / result['overall'] * 100).round(2)

                dump(result, score_file)
        else:
            result = load(score_file)

        return result
