import os
import json
import math
import subprocess
import sys
import glob

def parse_inference_output(output):
    try:
        start_idx = output.find('{')
        end_idx = output.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            import re
            fixed_str = output[start_idx:end_idx+1].replace('\n', '').replace('\r', '')
            fixed_str = re.sub(r'([a-zA-Z])\s{2,}([a-zA-Z])', r'\1\2', fixed_str)
            return json.loads(fixed_str)
    except:
        pass
    return None

def evaluate_script(script_name, gt_files):
    results = []
    print(f"Evaluating {script_name}...")
    for gt_path in gt_files:
        pair_dir = os.path.dirname(gt_path)
        ref_path = os.path.join(pair_dir, "reference.png")
        search_path = os.path.join(pair_dir, "search.png")
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt = json.load(f)
            
        pair_id = gt.get("pair_id")
        arch = gt.get("arch")
        if arch != "FINFET": continue
        
        center_x = gt.get("center_x", 0)
        center_y = gt.get("center_y", 0)
        
        cmd = [sys.executable, script_name, ref_path, search_path, "--arch", "FINFET", "--json"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
        pred = parse_inference_output(proc.stdout)
        
        if pred:
            px, py = pred.get("x", 0), pred.get("y", 0)
            err = math.sqrt((px - center_x)**2 + (py - center_y)**2)
            results.append({
                "pair_id": pair_id,
                "gt_x": center_x,
                "gt_y": center_y,
                "pred_x": px,
                "pred_y": py,
                "error_px": err,
                "abs_x": abs(px - center_x),
                "abs_y": abs(py - center_y),
            })
    return results

def get_stats(data):
    if not data: return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    errs = [r["error_px"] for r in data]
    abs_xs = [r["abs_x"] for r in data]
    abs_ys = [r["abs_y"] for r in data]
    
    a5 = sum(1 for e in errs if e <= 5) / len(errs) * 100
    a50 = sum(1 for e in errs if e <= 50) / len(errs) * 100
    me = sum(errs) / len(errs)
    md = sorted(errs)[len(errs)//2]
    
    maex = sum(abs_xs) / len(abs_xs)
    maey = sum(abs_ys) / len(abs_ys)
    a5x = sum(1 for x in abs_xs if x <= 5) / len(abs_xs) * 100
    a5y = sum(1 for y in abs_ys if y <= 5) / len(abs_ys) * 100
    a50x = sum(1 for x in abs_xs if x <= 50) / len(abs_xs) * 100
    a50y = sum(1 for y in abs_ys if y <= 50) / len(abs_ys) * 100
    
    return a5, a50, me, md, maex, maey, a5x, a5y, a50x, a50y

def main():
    gt_files = glob.glob(os.path.join("generated_dataset", "**", "ground_truth.json"), recursive=True)
    gt_files = [f for f in gt_files if json.load(open(f))["arch"] == "FINFET"]
    
    res_base = evaluate_script("inference.py", gt_files)
    res_v2 = evaluate_script("inference_periodic_v2.py", gt_files)
    
    b_a5, b_a50, b_me, b_md, b_maex, b_maey, b_a5x, b_a5y, b_a50x, b_a50y = get_stats(res_base)
    v_a5, v_a50, v_me, v_md, v_maex, v_maey, v_a5x, v_a5y, v_a50x, v_a50y = get_stats(res_v2)
    
    # Save CSV
    with open("periodic_v2_results.csv", 'w', encoding='utf-8') as f:
        f.write("pair_id,gt_x,gt_y,base_px,base_py,base_err,v2_px,v2_py,v2_err,improvement\n")
        for b, v in zip(res_base, res_v2):
            imp = b['error_px'] - v['error_px']
            f.write(f"{b['pair_id']},{b['gt_x']},{b['gt_y']},{b['pred_x']},{b['pred_y']},{b['error_px']:.2f},"
                    f"{v['pred_x']},{v['pred_y']},{v['error_px']:.2f},{imp:.2f}\n")
            
    # Worst case analysis
    pairs = []
    for b, v in zip(res_base, res_v2):
        imp = b['error_px'] - v['error_px']
        pairs.append({
            "id": b['pair_id'], "gt": (b['gt_x'], b['gt_y']), 
            "b_pred": (b['pred_x'], b['pred_y']), "v_pred": (v['pred_x'], v['pred_y']),
            "b_err": b['error_px'], "v_err": v['error_px'], "imp": imp
        })
        
    print("\n10 LARGEST IMPROVEMENTS:")
    for p in sorted(pairs, key=lambda x: x["imp"], reverse=True)[:10]:
        print(f"{p['id']:<11} | GT:{p['gt']} | Base:{p['b_pred']} err:{p['b_err']:.1f} | V2:{p['v_pred']} err:{p['v_err']:.1f} | Imp: +{p['imp']:.1f}")
        
    print("\n10 LARGEST REGRESSIONS:")
    for p in sorted(pairs, key=lambda x: x["imp"])[:10]:
        print(f"{p['id']:<11} | GT:{p['gt']} | Base:{p['b_pred']} err:{p['b_err']:.1f} | V2:{p['v_pred']} err:{p['v_err']:.1f} | Imp: {p['imp']:.1f}")
        
    print("\n============================================================")
    print("PERIODICITY-AWARE V2 A/B TEST")
    print("============================================================")
    
    print("BASELINE:")
    print(f"Acc@5: {b_a5:.1f}%")
    print(f"Acc@50: {b_a50:.1f}%")
    print(f"Mean error: {b_me:.2f} px")
    print(f"Median error: {b_md:.2f} px\n")
    
    print("PERIODIC V2:")
    print(f"Acc@5: {v_a5:.1f}%")
    print(f"Acc@50: {v_a50:.1f}%")
    print(f"Mean error: {v_me:.2f} px")
    print(f"Median error: {v_md:.2f} px\n")
    
    print("DELTA:")
    print(f"Acc@5: {v_a5 - b_a5:+.1f}%")
    print(f"Acc@50: {v_a50 - b_a50:+.1f}%")
    print(f"Mean error: {v_me - b_me:+.2f} px")
    print(f"Median error: {v_md - b_md:+.2f} px\n")
    
    print("MAE-X:")
    print(f"Baseline: {b_maex:.2f}")
    print(f"V2: {v_maex:.2f}\n")
    
    print("MAE-Y:")
    print(f"Baseline: {b_maey:.2f}")
    print(f"V2: {v_maey:.2f}\n")
    print("============================================================\n")
    
    print("1. Did periodic alias grouping help?")
    print("2. Did larger context help?")
    print("3. Did local phase correlation help?")
    print("4. Did V2 reduce large periodic jumps?")
    print("5. Did V2 improve both X and Y?")
    print("6. Did V2 improve the overall benchmark?")
    print("7. Should V2 replace inference.py?")
    
    # Recommendation logic
    if (v_a5 > b_a5) and (v_me < b_me * 0.9):
        print("\nACCEPT")
    else:
        print("\nREJECT")
        
    report = f"""# PERIODIC V2 REPORT
## Baseline
- Acc@5: {b_a5:.1f}%
- Mean err: {b_me:.1f}
## V2
- Acc@5: {v_a5:.1f}%
- Mean err: {v_me:.1f}
"""
    with open("periodic_v2_report.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
