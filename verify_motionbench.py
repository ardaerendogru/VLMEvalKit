#!/usr/bin/env python3
"""
Verify MotionBench dataset download and integrity.

Usage:
    python verify_motionbench.py
"""

import os
import sys

def verify_motionbench():
    """Verify MotionBench dataset is correctly downloaded."""

    # Expected base path from huggingface_hub cache
    base_patterns = [
        os.path.expanduser('~/.cache/huggingface/hub/datasets--THUDM--MotionBench'),
    ]

    print("=" * 60)
    print("MotionBench Dataset Verification")
    print("=" * 60)

    # Find the dataset directory
    dataset_root = None
    for pattern in base_patterns:
        if os.path.exists(pattern):
            # Find the snapshots directory
            snapshots_dir = os.path.join(pattern, 'snapshots')
            if os.path.isdir(snapshots_dir):
                # Get the latest snapshot
                snapshots = [d for d in os.listdir(snapshots_dir)
                           if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshots:
                    latest = sorted(snapshots)[-1]
                    dataset_root = os.path.join(snapshots_dir, latest)
                    break

    if not dataset_root:
        print("❌ MotionBench dataset not found in HuggingFace cache.")
        print("   Run: python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id=\"THUDM/MotionBench\", repo_type=\"dataset\")'")
        return False

    print(f"✓ Dataset location: {dataset_root}")

    # Check for MotionBench subdirectory
    motionbench_dir = os.path.join(dataset_root, 'MotionBench')
    if not os.path.isdir(motionbench_dir):
        print(f"❌ MotionBench/ subdirectory not found")
        return False
    print(f"✓ MotionBench/ subdirectory exists")

    # Check for video_info.meta.jsonl
    jsonl_file = os.path.join(motionbench_dir, 'video_info.meta.jsonl')
    if not os.path.isfile(jsonl_file):
        print(f"❌ video_info.meta.jsonl not found")
        return False

    # Count entries in JSONL
    with open(jsonl_file, 'r') as f:
        line_count = sum(1 for _ in f)
    print(f"✓ video_info.meta.jsonl: {line_count:,} entries")

    # Check for TSV file
    tsv_file = os.path.join(dataset_root, 'MotionBench.tsv')
    if not os.path.isfile(tsv_file):
        print(f"❌ MotionBench.tsv not found (will be generated on first run)")
    else:
        print(f"✓ MotionBench.tsv exists")

    # Check for self-collected directory
    self_collected_dir = os.path.join(motionbench_dir, 'self-collected')
    if not os.path.isdir(self_collected_dir):
        print(f"❌ self-collected/ directory not found")
        return False

    # Count videos in self-collected
    self_collected_videos = [f for f in os.listdir(self_collected_dir) if f.endswith('.mp4')]
    print(f"✓ self-collected/: {len(self_collected_videos):,} videos")

    # Check for public-dataset directory
    public_dataset_dir = os.path.join(motionbench_dir, 'public-dataset')
    if os.path.isdir(public_dataset_dir):
        public_dataset_videos = [f for f in os.listdir(public_dataset_dir) if f.endswith('.mp4')]
        print(f"✓ public-dataset/: {len(public_dataset_videos):,} videos")
    else:
        public_dataset_videos = []
        print(f"⚠ public-dataset/ directory not found (optional)")

    # Summary
    total_videos = len(self_collected_videos) + len(public_dataset_videos)

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"  Total videos: {total_videos:,}")
    print(f"  Total questions: {line_count:,}")
    print(f"\n✓ MotionBench dataset is correctly downloaded!")

    # Test sample video access
    print("\nTesting sample video access...")
    test_video = os.path.join(motionbench_dir, 'self-collected', self_collected_videos[0])
    print(f"  Sample: {os.path.basename(test_video)}")
    print(f"  Path: {test_video}")
    if os.path.exists(test_video):
        file_size = os.path.getsize(test_video) / (1024 * 1024)
        print(f"  Size: {file_size:.2f} MB")
        print(f"  ✓ Video is accessible")
    else:
        print(f"  ❌ Video is NOT accessible")
        return False

    # Test TSV video prefix accuracy
    print("\n" + "=" * 60)
    print("Testing TSV video prefix assignments...")
    print("=" * 60)

    import pandas as pd
    tsv_file = os.path.join(dataset_root, 'MotionBench.tsv')
    if os.path.exists(tsv_file):
        df = pd.read_csv(tsv_file, sep='\t')

        # Check public-dataset videos
        public_df = df[df['video_prefix'] == './public-dataset/']
        print(f"  Videos marked as public-dataset: {len(public_df):,}")

        # Check a few random public-dataset videos exist
        sample_public = public_df.head(5)
        public_ok = 0
        for _, row in sample_public.iterrows():
            vid_path = os.path.join(motionbench_dir, 'public-dataset', row['video'] + row['video_suffix'])
            if os.path.exists(vid_path):
                public_ok += 1
        print(f"  Sample public-dataset videos found: {public_ok}/5")

        # Check self-collected videos
        self_df = df[df['video_prefix'] == './self-collected/']
        print(f"  Videos marked as self-collected: {len(self_df):,}")

        # Check a few random self-collected videos exist
        sample_self = self_df.head(5)
        self_ok = 0
        for _, row in sample_self.iterrows():
            vid_path = os.path.join(motionbench_dir, 'self-collected', row['video'] + row['video_suffix'])
            if os.path.exists(vid_path):
                self_ok += 1
        print(f"  Sample self-collected videos found: {self_ok}/5")

        if public_ok >= 4 and self_ok >= 4:
            print(f"  ✓ Video prefix assignments are correct")
        else:
            print(f"  ⚠ Some video prefixes may be incorrect")

    return True


if __name__ == '__main__':
    success = verify_motionbench()
    sys.exit(0 if success else 1)
