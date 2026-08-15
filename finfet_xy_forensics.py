import csv
import math
import sys
import numpy as np
from collections import Counter

def format_gt(gt_x, gt_y):
    return f"({float(gt_x):.0f},{float(gt_y):.0f})"

def bin_offset(offset, bin_size=10):
    return round(offset / bin_size) * bin_size

def main():
    csv_file = "generated_dataset_evaluation.csv"
    
    # STEP 1
    required_cols = [
        "pair_id", "split", "arch", "gt_x", "gt_y", 
        "pred_x", "pred_y", "error_px", "acc5", "acc50", 
        "confidence", "elapsed_ms", "candidates_evaluated"
    ]
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            for col in required_cols:
                if col not in headers:
                    print(f"Error: Missing required column '{col}'. Stop.")
                    sys.exit(1)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        sys.exit(1)

    finfet_rows = [r for r in rows if r['arch'].upper() == 'FINFET']
    if not finfet_rows:
        print("No FINFET pairs found.")
        sys.exit(1)
        
    # STEP 2
    processed = []
    inconsistency_found = False
    for row in finfet_rows:
        pred_x, pred_y = float(row['pred_x']), float(row['pred_y'])
        gt_x, gt_y = float(row['gt_x']), float(row['gt_y'])
        orig_err = float(row['error_px'])
        
        dx = pred_x - gt_x
        dy = pred_y - gt_y
        
        abs_x = abs(dx)
        abs_y = abs(dy)
        
        calc_err = math.sqrt(abs_x**2 + abs_y**2)
        if abs(calc_err - orig_err) > 1.0: # allow floating point slop
            inconsistency_found = True
            print(f"Inconsistency in {row['pair_id']}: original {orig_err} != calculated {calc_err}")
            
        cat = ""
        if abs_x <= 5 and abs_y <= 5:
            cat = "BOTH_CORRECT"
        elif abs_x <= 5 and abs_y > 5:
            cat = "X_ONLY_CORRECT"
        elif abs_y <= 5 and abs_x > 5:
            cat = "Y_ONLY_CORRECT"
        else:
            cat = "BOTH_WRONG"
            
        row['dx'] = dx
        row['dy'] = dy
        row['abs_error_x'] = abs_x
        row['abs_error_y'] = abs_y
        row['error_category'] = cat
        processed.append(row)
        
    if not inconsistency_found:
        pass # All good

    # STEP 3 & 4
    def get_split_stats(split_name, data):
        if not data:
            return None
        abs_xs = [r['abs_error_x'] for r in data]
        abs_ys = [r['abs_error_y'] for r in data]
        mae_x = np.mean(abs_xs)
        mae_y = np.mean(abs_ys)
        med_x = np.median(abs_xs)
        med_y = np.median(abs_ys)
        acc5_x = np.mean([1 if x <= 5 else 0 for x in abs_xs]) * 100
        acc50_x = np.mean([1 if x <= 50 else 0 for x in abs_xs]) * 100
        acc5_y = np.mean([1 if y <= 5 else 0 for y in abs_ys]) * 100
        acc50_y = np.mean([1 if y <= 50 else 0 for y in abs_ys]) * 100
        return (split_name, len(data), mae_x, mae_y, med_x, med_y, acc5_x, acc5_y, acc50_x, acc50_y)

    print("\nSplit       Pairs   MAE-X   MAE-Y   Med-X   Med-Y   Acc5-X   Acc5-Y   Acc50-X   Acc50-Y")
    print("-" * 91)
    
    splits = ["train", "validation", "test"]
    stats_map = {}
    for sp in splits:
        sp_data = [r for r in processed if r['split'] == sp]
        res = get_split_stats(sp, sp_data)
        if res:
            stats_map[sp] = res
            s, p, mx, my, mdx, mdy, a5x, a5y, a50x, a50y = res
            print(f"{s:<11} {p:<7} {mx:<7.1f} {my:<7.1f} {mdx:<7.1f} {mdy:<7.1f} {a5x:<8.1f} {a5y:<8.1f} {a50x:<9.1f} {a50y:<9.1f}")
            
    all_res = get_split_stats("ALL", processed)
    stats_map["ALL"] = all_res
    s, p, mx, my, mdx, mdy, a5x, a5y, a50x, a50y = all_res
    print(f"{s:<11} {p:<7} {mx:<7.1f} {my:<7.1f} {mdx:<7.1f} {mdy:<7.1f} {a5x:<8.1f} {a5y:<8.1f} {a50x:<9.1f} {a50y:<9.1f}")
    
    # STEP 5
    cat_counts = Counter([r['error_category'] for r in processed])
    print("\nCategory             Count      Percentage")
    print("-" * 42)
    for c in ["BOTH_CORRECT", "X_ONLY_CORRECT", "Y_ONLY_CORRECT", "BOTH_WRONG"]:
        cnt = cat_counts[c]
        pct = (cnt / len(processed)) * 100
        print(f"{c:<20} {cnt:<10} {pct:.1f}%")

    # STEP 6
    abs_xs = [r['abs_error_x'] for r in processed]
    abs_ys = [r['abs_error_y'] for r in processed]
    if np.std(abs_xs) == 0 or np.std(abs_ys) == 0:
        correlation = "Cannot calculate (zero variance)"
    else:
        correlation = f"{np.corrcoef(abs_xs, abs_ys)[0,1]:.4f}"

    # STEP 7
    dxs = [r['dx'] for r in processed]
    dys = [r['dy'] for r in processed]
    mean_bias_x = np.mean(dxs)
    mean_bias_y = np.mean(dys)
    med_bias_x = np.median(dxs)
    med_bias_y = np.median(dys)
    
    def print_top(items):
        for r in items:
            gt_str = format_gt(r['gt_x'], r['gt_y'])
            pr_str = format_gt(r['pred_x'], r['pred_y'])
            print(f"{r['pair_id']:<11} | {r['split']:<10} | {gt_str:<11} | {pr_str:<11} | {r['abs_error_x']:<7.1f} | {r['abs_error_y']:<7.1f} | {float(r['error_px']):.1f}")
            
    # STEP 8
    print("\n10 LARGEST X ERRORS:")
    print("pair_id     | split      | GT          | prediction  | X-error | Y-error | total-error")
    worst_x = sorted(processed, key=lambda r: r['abs_error_x'], reverse=True)[:10]
    print_top(worst_x)

    # STEP 9
    print("\n10 LARGEST Y ERRORS:")
    print("pair_id     | split      | GT          | prediction  | X-error | Y-error | total-error")
    worst_y = sorted(processed, key=lambda r: r['abs_error_y'], reverse=True)[:10]
    print_top(worst_y)

    # STEP 10
    print("\nX-CORRECT / Y-WRONG CASES:")
    print("pair_id     | split      | GT          | prediction  | X-error | Y-error | total-error")
    xc_yw = [r for r in processed if r['abs_error_x'] <= 5 and r['abs_error_y'] > 50]
    print_top(xc_yw[:10])

    # STEP 11
    print("\nY-CORRECT / X-WRONG CASES:")
    print("pair_id     | split      | GT          | prediction  | X-error | Y-error | total-error")
    yc_xw = [r for r in processed if r['abs_error_y'] <= 5 and r['abs_error_x'] > 50]
    print_top(yc_xw[:10])

    # STEP 12
    failed_pairs = [r for r in processed if float(r['error_px']) > 5]
    binned_dx = [bin_offset(r['dx']) for r in failed_pairs]
    binned_dy = [bin_offset(r['dy']) for r in failed_pairs]
    
    top_dx = Counter(binned_dx).most_common(3)
    top_dy = Counter(binned_dy).most_common(3)
    
    most_common_dx = ", ".join([f"{v}px (x{c})" for v, c in top_dx])
    most_common_dy = ", ".join([f"{v}px (x{c})" for v, c in top_dy])

    # STEP 13
    out_csv = "finfet_xy_forensics.csv"
    with open(out_csv, 'w', encoding='utf-8') as f:
        out_cols = required_cols[:]
        # insert new columns before confidence
        idx = out_cols.index("confidence")
        out_cols.insert(idx, "error_category")
        out_cols.insert(idx, "abs_error_y")
        out_cols.insert(idx, "abs_error_x")
        
        f.write(",".join(out_cols) + "\n")
        for r in processed:
            row_vals = [str(r[c]) for c in out_cols]
            f.write(",".join(row_vals) + "\n")

    # STEP 14
    print("\n============================================================")
    print("FINFET X/Y FORENSIC ANALYSIS")
    print("============================================================")
    
    print(f"ALL FINFET PAIRS:\nPairs: {all_res[1]}\n")
    print(f"MAE-X: {all_res[2]:.2f}")
    print(f"MAE-Y: {all_res[3]:.2f}\n")
    print(f"Median-X: {all_res[4]:.2f}")
    print(f"Median-Y: {all_res[5]:.2f}\n")
    print(f"Acc@5-X: {all_res[6]:.1f}%")
    print(f"Acc@50-X: {all_res[8]:.1f}%\n")
    print(f"Acc@5-Y: {all_res[7]:.1f}%")
    print(f"Acc@50-Y: {all_res[9]:.1f}%\n")
    
    print(f"BOTH_CORRECT: {cat_counts['BOTH_CORRECT']}")
    print(f"X_ONLY_CORRECT: {cat_counts['X_ONLY_CORRECT']}")
    print(f"Y_ONLY_CORRECT: {cat_counts['Y_ONLY_CORRECT']}")
    print(f"BOTH_WRONG: {cat_counts['BOTH_WRONG']}\n")
    
    print(f"Error correlation: {correlation}\n")
    
    print(f"Mean X bias: {mean_bias_x:.2f}")
    print(f"Mean Y bias: {mean_bias_y:.2f}\n")
    print(f"Median X bias: {med_bias_x:.2f}")
    print(f"Median Y bias: {med_bias_y:.2f}\n")
    print(f"Most common X displacement: {most_common_dx}")
    print(f"Most common Y displacement: {most_common_dy}")
    print("============================================================")
    
    # Interpretation
    # Let's decide based on data.
    # If BOTH_WRONG is dominant -> 2D
    # If X_ONLY_CORRECT is high -> VERTICAL problem (X is fine, Y is wrong)
    # If Y_ONLY_CORRECT is high -> HORIZONTAL problem
    
    print("\nCONCLUSION:")
    if cat_counts['X_ONLY_CORRECT'] > max(cat_counts['Y_ONLY_CORRECT'], cat_counts['BOTH_WRONG']):
        print("VERTICAL")
        print("The data shows that X localization is frequently correct while Y localization fails significantly. The observed errors are consistent with directional periodic aliasing along the vertical axis.")
    elif cat_counts['Y_ONLY_CORRECT'] > max(cat_counts['X_ONLY_CORRECT'], cat_counts['BOTH_WRONG']):
        print("HORIZONTAL")
        print("The data shows that Y localization is frequently correct while X localization fails significantly. The observed errors are consistent with directional periodic aliasing along the horizontal axis.")
    elif cat_counts['BOTH_WRONG'] > max(cat_counts['X_ONLY_CORRECT'], cat_counts['Y_ONLY_CORRECT']):
        print("2D")
        print("The data shows significant failures in both X and Y simultaneously. The observed errors are consistent with 2D symmetric ambiguity or a general loss of signal.")
    else:
        print("INCONCLUSIVE")
        print("The error distribution does not heavily favor a single directional axis exclusively.")

    # STEP 15
    print("\nFILES MODIFIED:")
    print("None")
    print("\nFILES CREATED:")
    print("finfet_xy_forensics.py")
    print("finfet_xy_forensics.csv")
    print("\nALGORITHM MODIFIED:")
    print("NO")
    print("\nMODEL RETRAINED:")
    print("NO")

if __name__ == "__main__":
    main()
