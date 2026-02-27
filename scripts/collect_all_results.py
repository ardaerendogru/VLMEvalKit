#!/usr/bin/env python3
"""
Collect all evaluation results from outputs folder and create consolidated tables.

This script:
1. Finds all rating.json files in the outputs folder
2. Extracts dataset name, model name, and fps from file paths
3. Reads the JSON files and extracts metrics
4. Creates tables grouped by dataset with fps column

Usage:
    python scripts/collect_all_results.py [--output-dir OUTPUT_DIR] [--output-format FORMAT]
    
    python scripts/collect_all_results.py --output-dir results --output-format csv
    python scripts/collect_all_results.py --output-format xlsx
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("WARNING: pandas not available. Excel output will be disabled. Install with: pip install pandas openpyxl")


def extract_info_from_path(file_path):
    """Extract dataset, model, and fps from file path."""
    path_str = str(file_path)
    filename = Path(file_path).name
    
    # Pattern: ModelName_DatasetName_fps_rating.json
    # Examples:
    #   Qwen3-VL-2B-Instruct_TrafficQA_test_1fps_rating.json
    #   Qwen3-VL-2B-Instruct-AWQ-W8A8_MVBench_MP4_1fps_rating.json
    
    # Extract fps first
    fps_match = re.search(r'(\d+)fps', filename)
    if not fps_match:
        fps_match = re.search(r'(\d+)fps', path_str)
    fps = fps_match.group(1) if fps_match else 'unknown'
    
    # Remove _rating.json suffix
    base_name = filename.replace('_rating.json', '')
    
    # Try to find dataset patterns
    # Pattern 1: Model_Dataset_fps
    pattern1 = re.match(r'(.+?)_(TrafficQA_test|MotionBench|MVBench_MP4)_(\d+)fps$', base_name)
    if pattern1:
        model = pattern1.group(1)
        dataset = pattern1.group(2) + '_' + pattern1.group(3) + 'fps'
        return dataset, model, fps
    
    # Pattern 2: Model_Dataset_fps (more general)
    # Find the last occurrence of common dataset patterns
    dataset_patterns = [
        r'TrafficQA_test',
        r'MotionBench',
        r'MVBench_MP4',
        r'TrafficQA',
        r'MVBench',
    ]
    
    for pattern in dataset_patterns:
        match = re.search(pattern, base_name)
        if match:
            # Extract model (everything before the dataset pattern)
            model_end = match.start()
            model = base_name[:model_end].rstrip('_')
            # Extract dataset (from pattern to fps)
            dataset_part = base_name[model_end:]
            # Remove fps suffix if present
            dataset = re.sub(r'_\d+fps$', '', dataset_part)
            if fps != 'unknown':
                dataset = dataset + '_' + fps + 'fps'
            return dataset, model, fps
    
    # Fallback: try to extract from filename pattern Model_Dataset_fps
    parts = base_name.split('_')
    if len(parts) >= 3:
        # Try to find fps position
        fps_idx = None
        for i, part in enumerate(parts):
            if part == fps + 'fps' or (fps != 'unknown' and part.endswith('fps')):
                fps_idx = i
                break
        
        if fps_idx and fps_idx > 0:
            model = '_'.join(parts[:fps_idx-1])
            dataset = '_'.join(parts[fps_idx-1:fps_idx+1])
            return dataset, model, fps
    
    return None, None, None


def parse_trafficqa_rating(data):
    """Parse TrafficQA rating format."""
    results = {}
    if 'overall' in data:
        results['overall_acc'] = data['overall'].get('acc', 0) * 100
        results['overall_correct'] = data['overall'].get('correct', 0)
        results['overall_total'] = data['overall'].get('total', 0)
        results['overall_valid'] = data['overall'].get('valid', 0)
    
    # Add category-specific accuracies
    for key, value in data.items():
        if key != 'overall' and isinstance(value, dict) and 'acc' in value:
            results[f'{key}_acc'] = value.get('acc', 0) * 100
            results[f'{key}_correct'] = value.get('correct', 0)
            results[f'{key}_total'] = value.get('total', 0)
    
    return results


def parse_mvbench_rating(data):
    """Parse MVBench rating format."""
    results = {}
    total_correct = 0
    total_questions = 0
    
    for category, values in data.items():
        if isinstance(values, list) and len(values) >= 2:
            correct = values[0]
            total = values[1]
            acc = (correct / total * 100) if total > 0 else 0
            
            results[f'{category}_acc'] = acc
            results[f'{category}_correct'] = correct
            results[f'{category}_total'] = total
            
            total_correct += correct
            total_questions += total
    
    if total_questions > 0:
        results['overall_acc'] = (total_correct / total_questions) * 100
        results['overall_correct'] = total_correct
        results['overall_total'] = total_questions
    
    return results


def parse_motionbench_rating(data):
    """Parse MotionBench rating format."""
    results = {}
    
    # Handle list format
    if isinstance(data, list):
        total_correct = 0
        total_questions = 0
        for item in data:
            if isinstance(item, dict):
                qtype = item.get('question_type', 'unknown')
                acc = item.get('acc', 0)
                success = item.get('success', 0)
                overall = item.get('overall', 0)
                
                results[f'{qtype}_acc'] = acc
                results[f'{qtype}_success'] = success
                results[f'{qtype}_overall'] = overall
                
                total_questions += overall if overall > 0 else 0
                total_correct += success if success > 0 else 0
    # Handle dict format
    else:
        total_correct = 0
        total_questions = 0
        for key, value in data.items():
            if isinstance(value, dict):
                acc = value.get('acc', 0)
                success = value.get('success', 0)
                overall = value.get('overall', 0)
                
                results[f'{key}_acc'] = acc
                results[f'{key}_success'] = success
                results[f'{key}_overall'] = overall
                
                total_questions += overall if overall > 0 else 0
                total_correct += success if success > 0 else 0
    
    if total_questions > 0:
        results['overall_acc'] = (total_correct / total_questions) * 100
        results['overall_correct'] = total_correct
        results['overall_total'] = total_questions
    
    return results


def detect_and_parse_rating(data):
    """Detect rating format and parse accordingly."""
    # TrafficQA format
    if 'overall' in data and isinstance(data['overall'], dict) and 'acc' in data['overall']:
        return 'TrafficQA', parse_trafficqa_rating(data)
    
    # MVBench format (has categories with [correct, total, percentage] arrays)
    if any(isinstance(v, list) and len(v) >= 2 for v in data.values()):
        return 'MVBench', parse_mvbench_rating(data)
    
    # MotionBench format
    if isinstance(data, list) or (isinstance(data, dict) and 
                                   any(isinstance(v, dict) and 'acc' in v for v in data.values())):
        return 'MotionBench', parse_motionbench_rating(data)
    
    # Default: try to extract overall accuracy
    results = {}
    if 'overall' in data:
        if isinstance(data['overall'], dict):
            if 'acc' in data['overall']:
                results['overall_acc'] = data['overall']['acc'] * 100
        elif isinstance(data['overall'], (int, float)):
            results['overall_acc'] = data['overall']
    
    return 'Unknown', results


def collect_all_results(outputs_dir='outputs'):
    """Collect all results from outputs directory."""
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        print(f"ERROR: Outputs directory not found: {outputs_dir}")
        return {}
    
    # Find all rating.json files
    rating_files = list(outputs_path.rglob('*_rating.json'))
    print(f"Found {len(rating_files)} rating files")
    
    # Group by dataset, and deduplicate by (model, fps)
    dataset_results = defaultdict(dict)  # Use dict to deduplicate
    
    for rating_file in rating_files:
        dataset, model, fps = extract_info_from_path(rating_file)
        
        if not dataset or not model:
            print(f"WARNING: Could not extract info from {rating_file}")
            continue
        
        try:
            with open(rating_file, 'r') as f:
                data = json.load(f)
            
            dataset_type, metrics = detect_and_parse_rating(data)
            
            # Create result row
            row = {
                'model': model,
                'fps': fps,
                'dataset': dataset,
                'dataset_type': dataset_type,
                **metrics
            }
            
            # Use (model, fps) as key to deduplicate
            key = (model, fps)
            # Keep the result with more metrics (more complete)
            if key not in dataset_results[dataset] or len(metrics) > len(dataset_results[dataset][key].get('metrics', {})):
                dataset_results[dataset][key] = row
            
        except Exception as e:
            print(f"ERROR: Failed to process {rating_file}: {e}")
            continue
    
    # Convert dict of dicts to dict of lists
    return {dataset: list(results.values()) for dataset, results in dataset_results.items()}


def create_tables(dataset_results, output_format='csv', output_dir='results'):
    """Create tables for each dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_tables = {}
    
    for dataset, results in dataset_results.items():
        if not results:
            continue
        
        # Sort results by model, then fps
        def sort_key(r):
            fps_val = int(r.get('fps', '999')) if str(r.get('fps', '999')).isdigit() else 999
            return (r.get('model', ''), fps_val)
        
        results_sorted = sorted(results, key=sort_key)
        
        # Filter columns: dataset, fps, model, overall_acc, and per-task accuracies
        # User wants: dataset, fps, model, accuracy, per-task accuracy
        filtered_results = []
        for r in results_sorted:
            filtered_row = {
                'dataset': r.get('dataset', dataset),
                'fps': r.get('fps', 'unknown'),
                'model': r.get('model', 'unknown'),
            }
            
            # Add overall accuracy
            if 'overall_acc' in r:
                filtered_row['overall_acc'] = r['overall_acc']
            
            # Add all per-task accuracies (keys ending with '_acc' but not 'overall_acc')
            for key, value in r.items():
                if key.endswith('_acc') and key != 'overall_acc':
                    filtered_row[key] = value
            
            filtered_results.append(filtered_row)
        
        # Get all unique keys and order them: dataset, fps, model, overall_acc, then per-task accs
        all_keys = set()
        for r in filtered_results:
            all_keys.update(r.keys())
        
        # Check if this is MVBench or TrafficQA dataset - needs special column ordering
        is_mvbench = 'MVBench' in dataset or 'mvbench' in dataset.lower()
        is_trafficqa = 'TrafficQA' in dataset or 'trafficqa' in dataset.lower()
        
        if is_mvbench:
            # MVBench specific column order: model, overall, paper_value, then MotionBench-style categories
            # Map MVBench categories to MotionBench-style categories
            # Since MVBench has different categories, we'll aggregate/calculate averages where possible
            mvbench_to_motionbench_mapping = {
                # Action Order: temporal sequence tasks
                'action_sequence_acc': 'Action Order',
                'character_order_acc': 'Action Order',
                'episodic_reasoning_acc': 'Action Order',
                
                # Motion-related Objects: object-related tasks
                'object_interaction_acc': 'Motion-related Objects',
                'object_existence_acc': 'Motion-related Objects',
                'object_shuffle_acc': 'Motion-related Objects',
                
                # Repetition Count: counting tasks
                'action_count_acc': 'Repetition Count',
                'moving_count_acc': 'Repetition Count',
                
                # Location-related Motion: spatial movement tasks
                'moving_direction_acc': 'Location-related Motion',
                'egocentric_navigation_acc': 'Location-related Motion',
                'action_localization_acc': 'Location-related Motion',
                
                # Camera Motion: camera-related tasks
                'scene_transition_acc': 'Camera Motion',
                
                # Motion Recognition: action/recognition tasks
                'action_prediction_acc': 'Motion Recognition',
                'fine_grained_action_acc': 'Motion Recognition',
                'fine_grained_pose_acc': 'Motion Recognition',
                'moving_attribute_acc': 'Motion Recognition',
                'action_antonym_acc': 'Motion Recognition',
                'unexpected_action_acc': 'Motion Recognition',
                'counterfactual_inference_acc': 'Motion Recognition',
                'state_change_acc': 'Motion Recognition',
            }
            
            # MotionBench-style column order
            motionbench_task_order = [
                'Action Order',
                'Motion-related Objects',
                'Repetition Count',
                'Location-related Motion',
                'Camera Motion',
                'Motion Recognition',
            ]
            
            # Process results: aggregate MVBench categories into MotionBench categories
            for r in filtered_results:
                # Rename overall_acc to overall
                if 'overall_acc' in r:
                    r['overall'] = r.pop('overall_acc')
                
                # Aggregate MVBench categories into MotionBench categories
                aggregated = {}
                for mvbench_key, motionbench_category in mvbench_to_motionbench_mapping.items():
                    if mvbench_key in r:
                        if motionbench_category not in aggregated:
                            aggregated[motionbench_category] = []
                        aggregated[motionbench_category].append(r[mvbench_key])
                
                # Calculate averages for each MotionBench category
                for category, values in aggregated.items():
                    if values:
                        r[category] = round(sum(values) / len(values), 2)
                
                # Remove old MVBench category columns
                for key in list(r.keys()):
                    if key.endswith('_acc') and key != 'overall_acc':
                        r.pop(key, None)
                
                # Add empty paper_value column
                r['paper_value'] = ''
            
            # Build column order: dataset, fps, model, overall, paper_value, then MotionBench tasks in order
            priority_cols = ['dataset', 'fps', 'model', 'overall', 'paper_value']
            # Add MotionBench tasks in the specified order (only if they exist in the data)
            task_cols = [task for task in motionbench_task_order if any(task in r for r in filtered_results)]
            cols = priority_cols + task_cols
            
        elif is_trafficqa:
            # TrafficQA format: dataset, fps, model, overall, paper_value, then task categories
            # Define TrafficQA task order (alphabetical by display name)
            trafficqa_task_order = [
                'Attribution',
                'Basic Understanding',
                'Counterfactual Inference',
                'Event Forecasting',
                'Introspection',
                'Reverse Reasoning',
            ]
            
            # Mapping from column names to display names
            trafficqa_task_mapping = {
                'attribution_acc': 'Attribution',
                'basic_understanding_acc': 'Basic Understanding',
                'counterfactual_inference_acc': 'Counterfactual Inference',
                'event_forecasting_acc': 'Event Forecasting',
                'introspection_acc': 'Introspection',
                'reverse_reasoning_acc': 'Reverse Reasoning',
            }
            
            # Process results: rename columns and add paper_value
            for r in filtered_results:
                # Rename overall_acc to overall
                if 'overall_acc' in r:
                    r['overall'] = r.pop('overall_acc')
                
                # Rename task columns
                renamed = {}
                for key, value in r.items():
                    if key in trafficqa_task_mapping:
                        renamed[trafficqa_task_mapping[key]] = value
                    else:
                        renamed[key] = value
                r.clear()
                r.update(renamed)
                
                # Add empty paper_value column
                r['paper_value'] = ''
            
            # Build column order: dataset, fps, model, overall, paper_value, then TrafficQA tasks in order
            priority_cols = ['dataset', 'fps', 'model', 'overall', 'paper_value']
            # Add TrafficQA tasks in the specified order (only if they exist in the data)
            task_cols = [task for task in trafficqa_task_order if any(task in r for r in filtered_results)]
            cols = priority_cols + task_cols
            
        else:
            # For other datasets: dataset, fps, model, overall_acc, then sorted per-task accuracies
            priority_cols = ['dataset', 'fps', 'model']
            if 'overall_acc' in all_keys:
                priority_cols.append('overall_acc')
            
            # Get all per-task accuracies and sort them alphabetically
            per_task_accs = sorted([k for k in all_keys if k.endswith('_acc') and k != 'overall_acc'])
            cols = priority_cols + per_task_accs
        
        # Round numeric values
        for r in filtered_results:
            for k, v in r.items():
                if isinstance(v, float):
                    r[k] = round(v, 2)
        
        # Save table
        safe_dataset_name = dataset.replace('/', '_').replace('\\', '_')
        
        if output_format == 'csv' or (output_format == 'both' and not HAS_PANDAS):
            output_file = output_path / f'{safe_dataset_name}_results.csv'
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                writer.writerows(filtered_results)
            print(f"Saved: {output_file} ({len(filtered_results)} rows)")
        
        if HAS_PANDAS:
            # Create DataFrame for Excel output or display
            df = pd.DataFrame(filtered_results)
            df = df[[c for c in cols if c in df.columns]]
            
            if output_format == 'xlsx':
                all_tables[safe_dataset_name] = df
            elif output_format == 'both':
                all_tables[safe_dataset_name] = df
            
            # Print summary
            print(f"\n{dataset} ({len(filtered_results)} results):")
            print(df.to_string(index=False))
        else:
            # Print summary without pandas
            print(f"\n{dataset} ({len(filtered_results)} results):")
            if filtered_results:
                # Print header
                print(" | ".join(cols))
                print("-" * 100)
                # Print rows
                for r in filtered_results:
                    print(" | ".join(str(r.get(c, '')) for c in cols))
        print()
    
    # Save Excel file if requested and pandas is available
    if HAS_PANDAS and output_format in ['xlsx', 'both'] and all_tables:
        excel_file = output_path / 'all_results.xlsx'
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            for sheet_name, df in all_tables.items():
                # Excel sheet names have 31 char limit
                sheet_name_short = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name_short, index=False)
        print(f"Saved Excel file: {excel_file} with {len(all_tables)} sheets")
    elif output_format in ['xlsx', 'both'] and not HAS_PANDAS:
        print("WARNING: Excel output requires pandas. Install with: pip install pandas openpyxl")
    
    return all_tables


def main():
    parser = argparse.ArgumentParser(
        description='Collect all evaluation results from outputs folder'
    )
    parser.add_argument(
        '--outputs-dir',
        type=str,
        default='outputs',
        help='Path to outputs directory (default: outputs)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Path to output directory for results (default: results)'
    )
    parser.add_argument(
        '--output-format',
        choices=['csv', 'xlsx', 'both'],
        default='both',
        help='Output format: csv, xlsx, or both (default: both)'
    )
    
    args = parser.parse_args()
    
    print("Collecting results...")
    dataset_results = collect_all_results(args.outputs_dir)
    
    if not dataset_results:
        print("No results found!")
        return
    
    print(f"\nFound results for {len(dataset_results)} datasets")
    print("\nCreating tables...")
    create_tables(dataset_results, args.output_format, args.output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
