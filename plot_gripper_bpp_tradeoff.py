#!/usr/bin/env python3
"""
Plot trade-off curve between gripper BPP and success rate.
For each quality level, selects the result with the highest success rate.
Separates static_only and compress_gripper results.
"""

import json
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def quality_key(quality_str):
    """Convert quality string to sortable key."""
    # vlo1, vlo2 -> -2, -1 (smaller than numbers)
    # numbers -> as-is
    if quality_str.startswith('vlo'):
        num = int(quality_str[3:])
        return -10 + num  # vlo1=-9, vlo2=-8
    else:
        return int(quality_str)

def find_quality_dirs(output_dir):
    """Find all directories matching msillm_quality_{number} or msillm_quality_vlo{number} pattern."""
    output_path = Path(output_dir)
    quality_dirs = {}
    
    for item in output_path.iterdir():
        if not item.is_dir():
            continue
            
        # Match patterns like:
        # - msillm_quality_1
        # - msillm_quality_vlo1
        # - msillm-NeuralCompression_main-msillm_quality_1_epoch=82
        # - msillm-NeuralCompression_main-msillm_quality_vlo1_epoch=70
        # - msillm-NeuralCompression_main-msillm_quality_1_static_only_epoch=55
        match = re.search(r'msillm_quality_(\d+|vlo\d+)', item.name)
        if match:
            quality_str = match.group(1)
            if quality_str not in quality_dirs:
                quality_dirs[quality_str] = []
            quality_dirs[quality_str].append(item)
    
    return quality_dirs

def is_static_only(dir_name):
    """Check if directory is static_only (gripper not compressed)."""
    return 'static_only' in dir_name

def load_results(results_file):
    """Load results.json and extract success rate and BPP."""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        success_rate = data.get('average_success_rate', None)
        bpp_dict = data.get('bpp', {})
        static_bpp = bpp_dict.get('rgb_static', 0.0)
        gripper_bpp = bpp_dict.get('rgb_gripper', 0.0)  # Default to 0 if not present (static_only case)
        total_bpp = static_bpp + gripper_bpp
        
        # If gripper_bpp is None explicitly, set to 0
        if gripper_bpp is None:
            gripper_bpp = 0.0
        if static_bpp is None:
            static_bpp = 0.0
        
        return success_rate, gripper_bpp, total_bpp, static_bpp
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load {results_file}: {e}")
        return None, None, None, None

def find_best_result_for_quality(quality_dirs, filter_static_only=None):
    """Find the result with highest success rate for a given quality.
    
    Args:
        quality_dirs: List of directories for a quality level
        filter_static_only: If True, only static_only dirs. If False, only non-static_only. If None, all.
    """
    best_success_rate = -1
    best_gripper_bpp = None
    best_total_bpp = None
    best_static_bpp = None
    best_dir = None
    
    for dir_path in quality_dirs:
        # Filter by static_only if specified
        if filter_static_only is not None:
            if is_static_only(dir_path.name) != filter_static_only:
                continue
        
        results_file = dir_path / 'results.json'
        if not results_file.exists():
            continue
            
        success_rate, gripper_bpp, total_bpp, static_bpp = load_results(results_file)
        
        if success_rate is not None and gripper_bpp is not None:
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_gripper_bpp = gripper_bpp
                best_total_bpp = total_bpp
                best_static_bpp = static_bpp
                best_dir = dir_path
    
    return best_success_rate, best_gripper_bpp, best_total_bpp, best_static_bpp, best_dir

def plot_results(results_both, results_direct, output_file, output_dir):
    """Plot results and save to file."""
    plt.figure(figsize=(8, 6))
    
    # Plot both images results (from checkpoint dirs)
    if results_both:
        results_both.sort(key=lambda x: quality_key(str(x['quality'])))
        both_bpps = [r['gripper_bpp'] for r in results_both]
        both_success = [r['success_rate'] for r in results_both]
        plt.scatter(both_bpps, both_success, label="MS-ILLM (Decoder)", color='purple', marker='s', s=80)
        plt.plot(both_bpps, both_success, linestyle='-', linewidth=1.0, color='purple', alpha=0.5)
    
    # Plot direct msillm_quality_{quality} results
    direct_bpps = []
    if results_direct:
        results_direct.sort(key=lambda x: quality_key(str(x['quality'])))
        direct_bpps = [r['gripper_bpp'] for r in results_direct]
        direct_success = [r['success_rate'] for r in results_direct]
        plt.scatter(direct_bpps, direct_success, label="MS-ILLM", color='green', marker='^', s=80)
        plt.plot(direct_bpps, direct_success, linestyle='-', linewidth=1.0, color='green', alpha=0.5)
    
    # Plot additional data: MS-ILLM(Decoder + Vision Encoder)
    # Use same BPP values as MS-ILLM (Direct), but with different success rates
    success_C = [0.16, 0.324, 0.796, 0.892, 0.908, 0.918, 0.918, 0.930]
    if direct_bpps and len(direct_bpps) == len(success_C):
        plt.scatter(direct_bpps, success_C, label="MS-ILLM(Decoder + Vision Encoder)", color='red', marker='o', s=80)
        plt.plot(direct_bpps, success_C, linestyle='-', linewidth=1.0, color='red', alpha=0.5)
    elif direct_bpps:
        # If lengths don't match, use the provided BPP values as fallback
        bpp_C = [0.0281, 0.0336, 0.0778, 0.117, 0.186, 0.331, 0.460, 0.710]
        plt.scatter(bpp_C, success_C, label="MS-ILLM(Decoder + Vision Encoder)", color='red', marker='o', s=80)
        plt.plot(bpp_C, success_C, linestyle='-', linewidth=1.0, color='red', alpha=0.5)
    
    plt.xlabel("BPP (bit per pixel)")
    plt.ylabel("Success Rate (MoDE Diffusion Policy)")
    plt.title("BPP vs Success Rate (Gripper Images only)")
    
    # Add baseline
    baseline = 0.922
    plt.axhline(y=baseline, color='red', linestyle='--', label="Baseline")
    plt.text(x=0.55, y=0.86, s=f"Baseline={baseline}",
             color='red', fontsize=10, ha='right')
    
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Auto-adjust xlim based on data
    all_bpps = []
    if results_both:
        all_bpps.extend([r['gripper_bpp'] for r in results_both])
    if results_direct:
        all_bpps.extend([r['gripper_bpp'] for r in results_direct])
    
    if all_bpps:
        plt.xlim(0.0, max(all_bpps) * 1.1)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()

def find_direct_quality_dirs(output_dir):
    """Find direct msillm_quality_{quality} directories."""
    output_path = Path(output_dir)
    direct_dirs = {}
    
    for item in output_path.iterdir():
        if not item.is_dir():
            continue
        
        # Match exact pattern: msillm_quality_{number} or msillm_quality_vlo{number}
        match = re.match(r'^msillm_quality_(\d+|vlo\d+)$', item.name)
        if match:
            quality_str = match.group(1)
            direct_dirs[quality_str] = item
    
    return direct_dirs

def main():
    output_dir = Path('outputs/eval/outputs')
    
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist")
        return
    
    # Find all quality directories (from checkpoints)
    quality_dirs_dict = find_quality_dirs(output_dir)
    
    # Find direct msillm_quality_{quality} directories
    direct_quality_dirs = find_direct_quality_dirs(output_dir)
    
    if not quality_dirs_dict and not direct_quality_dirs:
        print("No quality directories found")
        return
    
    # Extract quality strings and sort
    all_qualities = set()
    if quality_dirs_dict:
        all_qualities.update(quality_dirs_dict.keys())
    if direct_quality_dirs:
        all_qualities.update(direct_quality_dirs.keys())
    qualities = sorted(all_qualities, key=quality_key)
    
    print(f"Found quality levels: {qualities}")
    
    # Process both images results (from checkpoints)
    print(f"\n{'='*110}")
    print(f"Analyzing both images results (from checkpoints)...")
    print(f"{'Quality':<12} {'Best SR':<12} {'BPP':<15} {'Gripper BPP':<15} {'Best Dir':<50}")
    print("=" * 110)
    
    results_both = []
    
    for quality in qualities:
        if quality not in quality_dirs_dict:
            continue
        dirs = quality_dirs_dict[quality]
        success_rate, gripper_bpp, total_bpp, static_bpp, best_dir = find_best_result_for_quality(dirs, filter_static_only=False)
        
        if success_rate is not None and total_bpp is not None:
            results_both.append({
                'quality': quality,
                'success_rate': success_rate,
                'gripper_bpp': gripper_bpp,
                'total_bpp': total_bpp,
                'static_bpp': static_bpp,
                'dir': best_dir.name if best_dir else None
            })
            print(f"{quality:<12} {success_rate:<12.4f} {total_bpp:<15.6f} {gripper_bpp:<15.6f} {best_dir.name if best_dir else 'N/A':<50}")
        else:
            print(f"{quality:<12} {'N/A':<12} {'N/A':<15} {'N/A':<15} {'N/A':<50}")
    
    # Process direct msillm_quality_{quality} results
    print(f"\n{'='*110}")
    print(f"Analyzing direct msillm_quality_* results...")
    print(f"{'Quality':<12} {'Success Rate':<15} {'Total BPP':<15} {'Static BPP':<15} {'Gripper BPP':<15}")
    print("=" * 110)
    
    results_direct = []
    
    for quality in qualities:
        if quality not in direct_quality_dirs:
            continue
        dir_path = direct_quality_dirs[quality]
        results_file = dir_path / 'results.json'
        
        if results_file.exists():
            success_rate, gripper_bpp, total_bpp, static_bpp = load_results(results_file)
            
            if success_rate is not None and total_bpp is not None:
                results_direct.append({
                    'quality': quality,
                    'success_rate': success_rate,
                    'gripper_bpp': gripper_bpp,
                    'total_bpp': total_bpp,
                    'static_bpp': static_bpp,
                    'dir': dir_path.name
                })
                print(f"{quality:<12} {success_rate:<15.4f} {total_bpp:<15.6f} {static_bpp:<15.6f} {gripper_bpp:<15.6f}")
            else:
                print(f"{quality:<12} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        else:
            print(f"{quality:<12} {'No results.json':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    # Plot all results together
    if results_both or results_direct:
        plot_results(results_both, results_direct, 'gripper_bpp_success_rate_tradeoff_all.png', output_dir)
        
        # Save data as CSV
        csv_file = output_dir / 'gripper_bpp_success_rate_data_all.csv'
        with open(csv_file, 'w') as f:
            f.write('type,quality,success_rate,total_bpp,static_bpp,gripper_bpp,directory\n')
            for r in results_both:
                f.write(f"both_checkpoints,{r['quality']},{r['success_rate']},{r['total_bpp']},{r['static_bpp']},{r['gripper_bpp']},{r['dir']}\n")
            for r in results_direct:
                f.write(f"both_direct,{r['quality']},{r['success_rate']},{r['total_bpp']},{r['static_bpp']},{r['gripper_bpp']},{r['dir']}\n")
        print(f"\nData saved to: {csv_file}")
    else:
        print(f"\nNo valid results found")

if __name__ == '__main__':
    main()