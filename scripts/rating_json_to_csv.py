#!/usr/bin/env python3
"""
Convert VLMEvalKit rating JSON files to CSV format.

Supports TrafficQA and MotionBench rating formats.

Usage:
    python scripts/rating_json_to_csv.py <rating_json_file> [output_csv_file]
    python scripts/rating_json_to_csv.py outputs/*/TrafficQA_test_2fps_rating.json
    python scripts/rating_json_to_csv.py outputs/*/MotionBench_2fps_rating.json results.csv

If output_csv_file is not provided, it will be created in the same directory as the input file.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def convert_trafficqa(data: dict, output_csv: str):
    """Convert TrafficQA rating format to CSV."""
    rows = []
    for category, metrics in data.items():
        if isinstance(metrics, dict) and 'acc' in metrics:
            rows.append({
                'category': category,
                'accuracy': round(metrics['acc'] * 100, 2),
                'correct': metrics.get('correct', 0),
                'total': metrics.get('total', 0),
                'valid': metrics.get('valid', 0)
            })

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['category', 'accuracy', 'correct', 'total', 'valid'])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def convert_motionbench(data: dict, output_csv: str):
    """Convert MotionBench rating format to CSV."""
    rows = []

    # Handle list format
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'question_type' in item:
                rows.append({
                    'question_type': item.get('question_type', ''),
                    'accuracy': item.get('acc', 0),
                    'success': item.get('success', 0),
                    'overall': item.get('overall', 0)
                })
    # Handle dict format where keys are question types
    else:
        for key, value in data.items():
            if isinstance(value, dict):
                rows.append({
                    'question_type': key,
                    'accuracy': value.get('acc', 0),
                    'success': value.get('success', 0),
                    'overall': value.get('overall', 0)
                })

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['question_type', 'accuracy', 'success', 'overall'])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def detect_format(data: dict) -> str:
    """Detect if data is TrafficQA or MotionBench format."""
    if 'basic_understanding' in data or 'attribution' in data or 'event_forecasting' in data:
        return 'trafficqa'
    return 'motionbench'


def main():
    parser = argparse.ArgumentParser(description='Convert VLMEvalKit rating JSON to CSV')
    parser.add_argument('input_json', help='Path to rating JSON file')
    parser.add_argument('output_csv', nargs='?', help='Path to output CSV file (optional)')
    parser.add_argument('--format', choices=['auto', 'trafficqa', 'motionbench'],
                        default='auto', help='Dataset format (default: auto-detect)')
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        output_path = input_path.with_suffix('.csv')

    # Load JSON
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Detect or use specified format
    if args.format == 'auto':
        fmt = detect_format(data)
        print(f"Detected format: {fmt}")
    else:
        fmt = args.format

    # Convert
    if fmt == 'trafficqa':
        rows = convert_trafficqa(data, str(output_path))
    else:
        rows = convert_motionbench(data, str(output_path))

    print(f"Converted {len(rows)} rows")
    print(f"Output saved to: {output_path}")

    # Print preview
    print("\nPreview:")
    if rows:
        headers = list(rows[0].keys())
        print("  " + " | ".join(headers))
        print("  " + "-" * 50)
        for row in rows[:5]:
            print("  " + " | ".join(str(row.get(h, '')) for h in headers))


if __name__ == '__main__':
    main()
