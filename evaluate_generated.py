import os
import json
import math
import subprocess
import sys
import glob

def find_ground_truths(base_dir):
    return glob.glob(os.path.join(base_dir, "**", "ground_truth.json"), recursive=True)

def parse_inference_output(output):
    # Try to find the JSON string. There might be spaces, newlines, etc.
    try:
        # Find first '{' and last '}'
        start_idx = output.find('{')
        end_idx = output.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_str = output[start_idx:end_idx+1]
            # Replace whitespace to handle weird terminal wraps
            import re
            json_str = re.sub(r'\s+', '', json_str)
            
            # Since removing all spaces might break things like "arch_detected":"FINFET", 
            # let's be more careful: only remove whitespace outside of quotes, or just rely on standard json.loads
            # The user output had spaces inside keys: "cand          didates_evaluated"
            # It's better to just do a smart regex fix or rely on standard parsing if we can.
            
            # Actually, standard json.loads might fail if there are weird spaces in keys. Let's try fixing it.
            # A safer approach for the provided example:
            fixed_str = output[start_idx:end_idx+1].replace('\n', '').replace('\r', '')
            # Fix large gaps of spaces inside words (like "cand         didates_evaluated")
            fixed_str = re.sub(r'([a-zA-Z])\s{2,}([a-zA-Z])', r'\1\2', fixed_str)
            return json.loads(fixed_str)
    except Exception as e:
        pass
    return None

def main():
    base_dir = "generated_dataset"
    gt_files = find_ground_truths(base_dir)
    
    results = []
    
    print(f"Found {len(gt_files)} pairs. Starting evaluation...\n")
    
    for gt_path in gt_files:
        pair_dir = os.path.dirname(gt_path)
        ref_path = os.path.join(pair_dir, "reference.png")
        search_path = os.path.join(pair_dir, "search.png")
        
        # Read GT
        with open(gt_path, 'r', encoding='utf-8') as f:
            try:
                gt = json.load(f)
            except Exception as e:
                print(f"[FAIL] {gt_path}: Invalid ground truth JSON. Skipping.")
                continue
                
        pair_id = gt.get("pair_id", "unknown")
        arch = gt.get("arch", "UNKNOWN")
        split = gt.get("split", "unknown")
        center_x = gt.get("center_x", 0)
        center_y = gt.get("center_y", 0)
        
        if not os.path.exists(ref_path) or not os.path.exists(search_path):
            print(f"[FAIL] {pair_id}: Missing reference or search image.")
            continue
            
        cmd = [sys.executable, "inference.py", ref_path, search_path, "--arch", arch, "--json"]
        
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
            
            if proc.returncode != 0:
                print(f"[FAIL] {pair_id}: Non-zero exit code {proc.returncode}.")
                continue
                
            pred = parse_inference_output(proc.stdout)
            if not pred:
                print(f"[FAIL] {pair_id}: Could not parse inference JSON.")
                continue
                
            pred_x = pred.get("x", 0)
            pred_y = pred.get("y", 0)
            conf = pred.get("confidence", 0.0)
            elapsed = pred.get("elapsed_ms", 0.0)
            cands = pred.get("candidates_evaluated", 0)
            
            error_px = math.sqrt((pred_x - center_x)**2 + (pred_y - center_y)**2)
            
            results.append({
                "pair_id": pair_id,
                "split": split,
                "arch": arch,
                "gt_x": center_x,
                "gt_y": center_y,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "error_px": error_px,
                "acc5": 1 if error_px <= 5 else 0,
                "acc50": 1 if error_px <= 50 else 0,
                "confidence": conf,
                "elapsed_ms": elapsed,
                "candidates_evaluated": cands
            })
            
            # print(f"Evaluated {pair_id}: err={error_px:.1f}")
            
        except subprocess.TimeoutExpired:
            print(f"[FAIL] {pair_id}: Subprocess timeout (exceeded 10s).")
            continue
        except Exception as e:
            print(f"[FAIL] {pair_id}: Unexpected error: {e}")
            continue

    if not results:
        print("No valid results collected.")
        sys.exit(1)
        
    # Calculate aggregates
    def get_stats(subset):
        if not subset:
            return 0, 0.0, 0.0, 0.0, 0.0
        acc5 = sum(r["acc5"] for r in subset) / len(subset) * 100
        acc50 = sum(r["acc50"] for r in subset) / len(subset) * 100
        errs = [r["error_px"] for r in subset]
        mean_err = sum(errs) / len(errs)
        median_err = sorted(errs)[len(errs)//2]
        return len(subset), acc5, acc50, mean_err, median_err

    # Create groups
    groups = {
        ("train", "FINFET"): [r for r in results if r["split"] == "train" and r["arch"] == "FINFET"],
        ("validation", "FINFET"): [r for r in results if r["split"] == "validation" and r["arch"] == "FINFET"],
        ("test", "FINFET"): [r for r in results if r["split"] == "test" and r["arch"] == "FINFET"],
        ("ALL", "FINFET"): [r for r in results if r["arch"] == "FINFET"],
        
        ("train", "DRAM"): [r for r in results if r["split"] == "train" and r["arch"] == "DRAM"],
        ("validation", "DRAM"): [r for r in results if r["split"] == "validation" and r["arch"] == "DRAM"],
        ("test", "DRAM"): [r for r in results if r["split"] == "test" and r["arch"] == "DRAM"],
        ("ALL", "DRAM"): [r for r in results if r["arch"] == "DRAM"],
    }
    
    print("\nSplit       Arch      Pairs   Acc@5   Acc@50   MeanErr   MedianErr")
    print("-" * 68)
    for k in [
        ("train", "DRAM"), ("validation", "DRAM"), ("test", "DRAM"), ("ALL", "DRAM"),
        ("train", "FINFET"), ("validation", "FINFET"), ("test", "FINFET"), ("ALL", "FINFET")
    ]:
        subset = groups[k]
        if subset:
            n, a5, a50, me, md = get_stats(subset)
            print(f"{k[0]:<11} {k[1]:<9} {n:<7} {a5:<7.1f} {a50:<8.1f} {me:<9.1f} {md:<9.1f}")
            
    print("\n10 Worst-Performing Pairs:")
    print("pair_id      | split      | arch   | GT          | prediction  | error")
    print("-" * 75)
    worst = sorted(results, key=lambda x: x["error_px"], reverse=True)[:10]
    for w in worst:
        gt_str = f"({w['gt_x']},{w['gt_y']})"
        pr_str = f"({w['pred_x']},{w['pred_y']})"
        print(f"{w['pair_id']:<12} | {w['split']:<10} | {w['arch']:<6} | {gt_str:<11} | {pr_str:<11} | {w['error_px']:.1f}")

    # Save to CSV
    csv_file = "generated_dataset_evaluation.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("pair_id,split,arch,gt_x,gt_y,pred_x,pred_y,error_px,acc5,acc50,confidence,elapsed_ms,candidates_evaluated\n")
        for r in results:
            f.write(f"{r['pair_id']},{r['split']},{r['arch']},{r['gt_x']},{r['gt_y']},{r['pred_x']},{r['pred_y']},"
                    f"{r['error_px']:.4f},{r['acc5']},{r['acc50']},{r['confidence']},{r['elapsed_ms']},{r['candidates_evaluated']}\n")

    print("\n============================================================")
    print("DriftSense-AI Generated Dataset Evaluation")
    print("============================================================")
    total_n, total_a5, total_a50, total_me, total_md = get_stats(results)
    print(f"Total pairs:  {total_n}")
    print(f"Acc@5px:      {total_a5:.1f}%")
    print(f"Acc@50px:     {total_a50:.1f}%")
    print(f"Mean error:   {total_me:.2f} px")
    print(f"Median error: {total_md:.2f} px")
    print("============================================================\n")

if __name__ == "__main__":
    main()
