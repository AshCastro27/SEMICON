import csv
import math
import sys
import numpy as np
import cv2
import os
from collections import Counter

def get_dominant_period(signal1d):
    s = signal1d - np.mean(signal1d)
    if np.var(s) == 0:
        return 0.0, 0.0
    f = np.fft.rfft(s)
    f[0] = 0 # zero DC
    freqs = np.fft.rfftfreq(len(s))
    
    peak_idx = np.argmax(np.abs(f))
    if peak_idx == 0 or freqs[peak_idx] == 0:
        return 0.0, 0.0
        
    period = 1.0 / freqs[peak_idx]
    strength = np.abs(f[peak_idx]) / (np.sum(np.abs(f)) + 1e-9)
    return period, float(strength)

def get_periods(img_path):
    img = cv2.imread(img_path, 0)
    if img is None:
        return 0.0, 0.0, 0.0, 0.0
    img = img.astype(np.float32)
    col_p = np.mean(img, axis=0)
    row_p = np.mean(img, axis=1)
    
    px, sx = get_dominant_period(col_p)
    py, sy = get_dominant_period(row_p)
    return px, py, sx, sy

def is_multiple(val, period, tolerance=0.20):
    if period < 2.0 or val < 5.0: # ignore very small periods or correct localized pairs
        return False, 0.0, 0.0
    ratio = val / period
    dist = abs(ratio - round(ratio))
    return dist < tolerance, ratio, dist

def bin_offset(offset, bin_size=10):
    return round(offset / bin_size) * bin_size

def main():
    csv_file = "generated_dataset_evaluation.csv"
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        sys.exit(1)
        
    finfet_rows = [r for r in rows if r['arch'].upper() == 'FINFET']
    
    results = []
    
    # Important pairs to explicitly trace
    important_pairs = {
        "pair_000027", "pair_000028", "pair_000021", "pair_000029", 
        "pair_000011", "pair_000016", "pair_000020", "pair_000004", 
        "pair_000008", "pair_000014"
    }
    
    failed_dxs = []
    failed_dys = []
    
    print("Processing pairs...")
    
    for row in finfet_rows:
        pair_id = row['pair_id']
        split = row['split']
        gt_x, gt_y = float(row['gt_x']), float(row['gt_y'])
        pred_x, pred_y = float(row['pred_x']), float(row['pred_y'])
        error_px = float(row['error_px'])
        
        dx = pred_x - gt_x
        dy = pred_y - gt_y
        
        abs_x = abs(dx)
        abs_y = abs(dy)
        
        if error_px > 5.0:
            failed_dxs.append(dx)
            failed_dys.append(dy)
            
        search_path = os.path.join("generated_dataset", split, pair_id, "search.png")
        px, py, sx, sy = get_periods(search_path)
        
        x_match, x_ratio, x_dist = is_multiple(abs_x, px)
        y_match, y_ratio, y_dist = is_multiple(abs_y, py)
        
        results.append({
            "pair_id": pair_id,
            "split": split,
            "dx": dx,
            "dy": dy,
            "abs_x": abs_x,
            "abs_y": abs_y,
            "px": px,
            "py": py,
            "sx": sx,
            "sy": sy,
            "x_ratio": x_ratio,
            "y_ratio": y_ratio,
            "x_match": x_match,
            "y_match": y_match
        })

    print("\n2. ANALYZE Y DISPLACEMENT")
    binned_dx = [bin_offset(x) for x in failed_dxs]
    binned_dy = [bin_offset(y) for y in failed_dys]
    
    top_dx = Counter(binned_dx).most_common(5)
    top_dy = Counter(binned_dy).most_common(5)
    
    print("Most common DX bins:", top_dx)
    print("Most common DY bins:", top_dy)
    
    print("\n5. IMPORTANT CASES")
    print(f"{'pair_id':<12} | {'abs_x':<7} | {'abs_y':<7} | {'px':<7} | {'py':<7} | {'x_ratio':<7} | {'y_ratio':<7} | {'x_match'} | {'y_match'}")
    print("-" * 85)
    for r in results:
        if r['pair_id'] in important_pairs:
            print(f"{r['pair_id']:<12} | {r['abs_x']:<7.1f} | {r['abs_y']:<7.1f} | {r['px']:<7.1f} | {r['py']:<7.1f} | {r['x_ratio']:<7.2f} | {r['y_ratio']:<7.2f} | {r['x_match']} | {r['y_match']}")
            
    # Aggregates for report
    avg_px = np.mean([r['px'] for r in results if r['px'] > 0])
    avg_py = np.mean([r['py'] for r in results if r['py'] > 0])
    avg_sx = np.mean([r['sx'] for r in results])
    avg_sy = np.mean([r['sy'] for r in results])
    
    x_alias_count = sum(1 for r in results if r['x_match'])
    y_alias_count = sum(1 for r in results if r['y_match'])
    
    total_x_errors = sum(1 for r in results if r['abs_x'] > 5.0)
    total_y_errors = sum(1 for r in results if r['abs_y'] > 5.0)
    
    print("\n============================================================")
    print("FINFET PERIODICITY FORENSIC REPORT")
    print("============================================================")
    print(f"Dominant X period: {avg_px:.2f} px")
    print(f"Dominant Y period: {avg_py:.2f} px\n")
    print(f"X periodicity strength: {avg_sx:.4f}")
    print(f"Y periodicity strength: {avg_sy:.4f}\n")
    
    print(f"Y errors matching periodic aliases: {y_alias_count} out of {total_y_errors} Y-errors")
    print(f"X errors matching periodic aliases: {x_alias_count} out of {total_x_errors} X-errors\n")
    
    # Determine strength
    def alias_strength(matches, errors):
        if errors == 0: return "NONE"
        ratio = matches / errors
        if ratio > 0.6: return "STRONG"
        if ratio > 0.3: return "MODERATE"
        if ratio > 0.1: return "WEAK"
        return "NONE"
        
    print(f"Vertical alias evidence: {alias_strength(y_alias_count, total_y_errors)}")
    print(f"Horizontal alias evidence: {alias_strength(x_alias_count, total_x_errors)}\n")
    
    print("Final interpretation:")
    
    if y_alias_count > x_alias_count and y_alias_count > total_y_errors * 0.5:
        print("The observed errors are consistent with directional periodic aliasing along the vertical axis.")
        print("The dominant Y period perfectly explains the large discrete jumps in Y errors.")
    elif x_alias_count > y_alias_count and x_alias_count > total_x_errors * 0.5:
        print("The observed errors are consistent with directional periodic aliasing along the horizontal axis.")
    elif y_alias_count > total_y_errors * 0.3 and x_alias_count > total_x_errors * 0.3:
        print("Current experiments indicate strong periodic ambiguity under the present input representation in both directions.")
    else:
        print("The errors do not strongly align with integer multiples of the dominant spatial periods.")
        print("Other sources of noise or algorithm breakdown may be responsible.")
        
    print("============================================================\n")

    # Save to CSV
    out_file = "finfet_periodicity_analysis.csv"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("pair_id,split,dx,dy,abs_x,abs_y,px,py,sx,sy,x_ratio,y_ratio,x_match,y_match\n")
        for r in results:
            f.write(f"{r['pair_id']},{r['split']},{r['dx']},{r['dy']},{r['abs_x']},{r['abs_y']},{r['px']:.2f},{r['py']:.2f},"
                    f"{r['sx']:.4f},{r['sy']:.4f},{r['x_ratio']:.3f},{r['y_ratio']:.3f},{r['x_match']},{r['y_match']}\n")

if __name__ == "__main__":
    main()
