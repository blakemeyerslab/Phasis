#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PhasMatch - Match Phasis calls (cals.tsv) to a ground-truth reference (BED 0-based or GFF/GTF 1-based).
# Reports BOTH views:
#   - ID-centric metrics (dedup by prediction identifier)
#   - Reference-centric metrics (dedup by reference feature)
#
# Note on Reference-centric precision/F1:
#   Because predictions and reference features can be many-to-many, there is no native FP on the
#   reference side. We report a pragmatic precision_Reference by counting unmatched prediction
#   identifiers as false positives against TP_Reference, and compute F1_Reference from that precision
#   and recall_Reference. This keeps the recall denominator purely reference-based while still
#   penalizing unmatched predictions in a single summary number.
#
# Identity check (Reference side) that should always hold:
#   TP_Reference + FN_Reference == total_Reference

import sys, os, argparse, datetime, csv
from collections import defaultdict

REQUIRED_CALL_COLS = {"identifier","achr","start","end","alib"}

def _read_tsv_header(path):
    with open(path, "r", newline="") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            return line.rstrip("\n").split("\t")
    return []

def parse_phasis_cals(path, one_based=True):
    header = _read_tsv_header(path)
    if not header:
        raise ValueError("Empty or invalid Phasis cals file: %s" % path)
    cols = {name: idx for idx, name in enumerate(header)}
    missing = REQUIRED_CALL_COLS - set(cols.keys())
    if missing:
        raise ValueError("Missing required columns in phasis file: %s" % sorted(missing))

    out = []
    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)  # skip header
        for ent in reader:
            if not ent:
                continue
            try:
                chrom = str(ent[cols["achr"]])
                start = int(ent[cols["start"]]) - (1 if one_based else 0)  # to 0-based
                end   = int(ent[cols["end"]])                               # half-open end
            except Exception:
                # skip malformed rows
                continue
            row = {
                "identifier": ent[cols["identifier"]],
                "chrom": chrom,
                "start": start,
                "end":   end,
                "library": ent[cols["alib"]],
                "phasis_score": ent[cols["phasis_score"]] if "phasis_score" in cols else "",
                "Peak_Howell_score": ent[cols["Peak_Howell_score"]] if "Peak_Howell_score" in cols else "",
                "Peak_Howell_score_strict": ent[cols["Peak_Howell_score_strict"]] if "Peak_Howell_score_strict" in cols else "",
            }
            out.append(row)
    return out

def detect_ref_format(path):
    b = os.path.basename(path).lower()
    if b.endswith(".bed"): return "bed"
    if b.endswith(".gff") or b.endswith(".gff3"): return "gff"
    if b.endswith(".gtf"): return "gtf"
    with open(path, "r", newline="") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            return "gff" if len(line.rstrip("\n").split("\t")) >= 9 else "bed"
    return "bed"

def parse_bed_reference(path):
    out = []
    with open(path, "r", newline="") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                chrom = p[0]; start = int(p[1]); end = int(p[2])
            except ValueError:
                continue
            name = p[3] if len(p) >= 4 else "."
            out.append({"chrom":chrom, "start":start, "end":end, "name":name})
    return out

def _name_from_gff_attr(attr):
    if not attr: return "."
    fields = [x.strip() for x in attr.split(";") if x.strip()]
    kv = {}
    for f in fields:
        if "=" in f: k, v = f.split("=", 1)
        elif " " in f: k, v = f.split(" ", 1)
        else:
            continue
        kv[k.strip()] = v.strip().strip('"')
    for key in ("ID","Name","gene_id","transcript_id"):
        if key in kv and kv[key]:
            return kv[key]
    return "."

def parse_gff_like_reference(path):
    out = []
    with open(path, "r", newline="") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9:
                continue
            try:
                chrom = p[0]; start1 = int(p[3]); end1 = int(p[4])
            except ValueError:
                continue
            start0 = start1 - 1
            end0   = end1
            name   = _name_from_gff_attr(p[8])
            out.append({"chrom":chrom, "start":start0, "end":end0, "name":name})
    return out

def parse_reference(path):
    return parse_bed_reference(path) if detect_ref_format(path)=="bed" else parse_gff_like_reference(path)

def halfopen_overlap(a_start, a_end, b_start, b_end):
    # half-open [start,end) overlap
    return (a_start < b_end) and (b_start < a_end)

def ref_key(tr):
    # Unique reference feature key: used for FN de-dup and reference-centric TP
    return (tr["chrom"], int(tr["start"]), int(tr["end"]), tr["name"])

def match_phasis_to_reference(phasis_rows, ref_rows, flank):
    # Index reference by chromosome; collect all unique reference keys
    ref_by_chr = defaultdict(list)
    all_ref_keys = set()
    for i, tr in enumerate(ref_rows):
        ref_by_chr[tr["chrom"]].append((i, tr))
        all_ref_keys.add(ref_key(tr))

    matched_ref_keys = set()
    rows = []

    # Per-identifier aggregation to prevent TP/FP inflation in id-centric metrics
    id_to_any_match = defaultdict(bool)

    for r in phasis_rows:
        chrom = r["chrom"]; start = int(r["start"]); end = int(r["end"])
        qS = max(0, start - flank); qE = end + 1 + flank  # prediction window (half-open)
        matches = []
        if chrom in ref_by_chr:
            for idx, tr in ref_by_chr[chrom]:
                tS = int(tr["start"]); tE = int(tr["end"])
                if halfopen_overlap(qS, qE, tS, tE):
                    matches.append((idx, tr))

        if matches:
            id_to_any_match[r["identifier"]] = True
            for idx, tr in matches:
                matched_ref_keys.add(ref_key(tr))
                rows.append({
                    "status": "TP",
                    "ph_id": r["identifier"],
                    "ph_chrom": chrom,
                    "ph_start": start,
                    "ph_end":   end,
                    "ph_library": r["library"],
                    "phasis_score": r.get("phasis_score",""),
                    "Peak_Howell_score": r.get("Peak_Howell_score",""),
                    "Peak_Howell_score_strict": r.get("Peak_Howell_score_strict",""),
                    "ref_name": tr["name"],
                    "ref_chrom": tr["chrom"],
                    "ref_start": int(tr["start"]),
                    "ref_end":   int(tr["end"]),
                })
        else:
            # keep per-row FP for auditing; id-centric metrics de-dup by identifier
            rows.append({
                "status": "FP",
                "ph_id": r["identifier"],
                "ph_chrom": chrom,
                "ph_start": start,
                "ph_end":   end,
                "ph_library": r["library"],
                "phasis_score": r.get("phasis_score",""),
                "Peak_Howell_score": r.get("Peak_Howell_score",""),
                "Peak_Howell_score_strict": r.get("Peak_Howell_score_strict",""),
                "ref_name": ".",
                "ref_chrom": ".",
                "ref_start": -1,
                "ref_end":   -1,
            })

    # ---- Reference-centric FN rows: one per unmatched reference feature ----
    unmatched_ref_keys = all_ref_keys - matched_ref_keys
    for chrom, items in ref_by_chr.items():
        for _, tr in items:
            if ref_key(tr) in unmatched_ref_keys:
                rows.append({
                    "status": "FN",
                    "ph_id": ".",
                    "ph_chrom": ".",
                    "ph_start": -1,
                    "ph_end":   -1,
                    "ph_library": ".",
                    "phasis_score": "",
                    "Peak_Howell_score": "",
                    "Peak_Howell_score_strict": "",
                    "ref_name": tr["name"],
                    "ref_chrom": tr["chrom"],
                    "ref_start": int(tr["start"]),
                    "ref_end":   int(tr["end"]),
                })

    # ---- Metrics (ID-CENTRIC and REFERENCE-CENTRIC) ----
    all_ids = {r["identifier"] for r in phasis_rows}
    tp_ids  = {i for i, anym in id_to_any_match.items() if anym}
    fp_ids  = all_ids - tp_ids

    # id-centric metrics
    precision_id = (len(tp_ids) / (len(tp_ids) + len(fp_ids))) if (len(tp_ids) + len(fp_ids)) > 0 else 0.0
    recall_id    = (len(tp_ids) / (len(tp_ids) + len(unmatched_ref_keys))) if (len(tp_ids) + len(unmatched_ref_keys)) > 0 else 0.0
    f1_id        = (2 * precision_id * recall_id / (precision_id + recall_id)) if (precision_id + recall_id) > 0 else 0.0

    # reference-centric metrics
    TP_Reference   = len(matched_ref_keys)
    total_Reference= len(all_ref_keys)
    FN_Reference   = len(unmatched_ref_keys)
    recall_Reference = (TP_Reference / total_Reference) if total_Reference > 0 else 0.0

    # Use unmatched prediction identifiers as FP against TP_Reference for a pragmatic precision_Reference
    precision_Reference = (TP_Reference / (TP_Reference + len(fp_ids))) if (TP_Reference + len(fp_ids)) > 0 else 0.0
    f1_Reference        = (2 * precision_Reference * recall_Reference / (precision_Reference + recall_Reference)) if (precision_Reference + recall_Reference) > 0 else 0.0

    # Per-library TP counts (unique prediction identifiers per library)
    tp_ids_by_lib = defaultdict(set)
    for row in rows:
        if row["status"] == "TP":
            tp_ids_by_lib[row["ph_library"]].add(row["ph_id"])
    tp_by_lib_unique_ids = {lib: len(s) for lib, s in tp_ids_by_lib.items()}

    metrics = {
        # id-centric
        "TP_ids": len(tp_ids),
        "FP_ids": len(fp_ids),
        "precision_id": precision_id,
        "recall_id": recall_id,
        "f1_id": f1_id,
        "total_unique_ids": len(all_ids),
        # reference-centric
        "TP_Reference": TP_Reference,
        "FN_Reference": FN_Reference,
        "total_Reference": total_Reference,
        "recall_Reference": recall_Reference,
        "precision_Reference": precision_Reference,
        "f1_Reference": f1_Reference,
        # sanity
        "matched_reference_plus_FN_equals_total": (TP_Reference + FN_Reference == total_Reference)
    }
    return rows, metrics, tp_by_lib_unique_ids

def write_matches_tsv(rows, path):
    cols = ["status",
            "ph_id","ph_chrom","ph_start","ph_end","ph_library","phasis_score","Peak_Howell_score","Peak_Howell_score_strict",
            "ref_name","ref_chrom","ref_start","ref_end"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c,"") for c in cols])

def write_summary_txt(summary_path, phasis_path, reference_path, flank, metrics, tp_by_lib_unique_ids):
    with open(summary_path, "w") as sm:
        sm.write("Purpose: Match Phasis cals.tsv vs Reference (BED or GFF/GTF).\n")
        sm.write(f"phasis file: {os.path.abspath(phasis_path)}\n")
        sm.write(f"reference: {os.path.abspath(reference_path)}\n")
        sm.write(f"flanksize: {flank} nt\n\n")

        # Predictions (ID-centric)
        sm.write("== Predictions (ID-centric, de-duplicated by identifier) ==\n")
        sm.write(f"Unique identifiers (predictions): {metrics['total_unique_ids']}\n")
        sm.write(f"TP (unique ids): {metrics['TP_ids']}\n")
        sm.write(f"FP (unique ids): {metrics['FP_ids']}\n")
        sm.write(f"precision (ID-centric): {metrics['precision_id']:.6f}\n")
        sm.write(f"recall    (ID-centric): {metrics['recall_id']:.6f}\n")
        sm.write(f"F1        (ID-centric): {metrics['f1_id']:.6f}\n\n")

        # Reference (feature-centric)
        sm.write("== Reference (feature-centric, de-duplicated by Reference feature) ==\n")
        sm.write(f"Total Reference features: {metrics['total_Reference']}\n")
        sm.write(f"TP (Reference features matched): {metrics['TP_Reference']}\n")
        sm.write(f"FN (Reference features unmatched): {metrics['FN_Reference']}\n")
        sm.write(f"precision (Reference-centric): {metrics['precision_Reference']:.6f}\n")
        sm.write(f"recall    (Reference-centric): {metrics['recall_Reference']:.6f}\n")
        sm.write(f"F1        (Reference-centric): {metrics['f1_Reference']:.6f}\n")
        sm.write(f"Sanity: TP_Reference + FN_Reference == total_Reference ? {metrics['matched_reference_plus_FN_equals_total']}\n\n")

        sm.write("# True positives per library (unique identifiers):\n")
        if not tp_by_lib_unique_ids:
            sm.write("(none)\n")
        else:
            for lib, cnt in sorted(tp_by_lib_unique_ids.items()):
                sm.write(f"{lib}\t{cnt}\n")


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Match Phasis calls.tsv to reference (BED or GFF/GTF). Reports both ID-centric and Reference-centric metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("phasis_cals", help="Phasis cals.tsv (needs: identifier, achr, start, end, alib).")
    ap.add_argument("reference", help="Reference intervals (BED 0-based or GFF/GTF 1-based).")
    ap.add_argument("-o", "--out-prefix", default=None, help="Output prefix.")
    ap.add_argument("--phasis-one-based",  dest="phasis_one_based", action="store_true",
                    help="Treat Phasis coordinates as 1-based (subtract 1 from start). Default.")
    ap.add_argument("--phasis-zero-based", dest="phasis_one_based", action="store_false",
                    help="Treat Phasis coordinates as already 0-based.")
    ap.set_defaults(phasis_one_based=True)
    ap.add_argument("--flank", type=int, default=250, help="Flank (nt) to expand Phasis intervals before matching.")
    return ap

def main():
    if len(sys.argv) == 1:
        print("Matches Phasis cals.tsv vs Reference (BED/GFF/GTF). Reports ID-centric and Reference-centric metrics.")
        print("Outputs: <prefix>.matches.tsv and <prefix>.summary.txt. Use -h for full help.\n")
        build_argparser().print_help()
        sys.exit(2)

    ap = build_argparser()
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%m_%d_%H_%M")
    prefix = args.out_prefix or f"match_{ts}"

    phasis_rows = parse_phasis_cals(args.phasis_cals, one_based=args.phasis_one_based)
    ref_rows    = parse_reference(args.reference)

    match_rows, metrics, tp_by_lib_unique = match_phasis_to_reference(phasis_rows, ref_rows, flank=args.flank)

    match_path   = f"{prefix}.matches.tsv"
    summary_path = f"{prefix}.summary.txt"
    write_matches_tsv(match_rows, match_path)
    write_summary_txt(summary_path, args.phasis_cals, args.reference, args.flank, metrics, tp_by_lib_unique)

    # Console report
    print(f"[OK] wrote {match_path}")
    print(f"[OK] wrote {summary_path}")
    print("[METRICS] ")
    print(f"ID-centric: TP(ids)={metrics['TP_ids']} FP(ids)={metrics['FP_ids']} "
          f"| Precision={metrics['precision_id']:.4f} Recall={metrics['recall_id']:.4f} F1={metrics['f1_id']:.4f} ; ")
    print(f"Reference-centric: TP_Reference={metrics['TP_Reference']} FN_Reference={metrics['FN_Reference']} total_Reference={metrics['total_Reference']} "
    f"| Precision_Reference={metrics['precision_Reference']:.4f} Recall_Reference={metrics['recall_Reference']:.4f} F1_Reference={metrics['f1_Reference']:.4f} "
    f"| Sanity(TP_Reference + FN_Reference == total_Reference)={metrics['matched_reference_plus_FN_equals_total']}")

if __name__ == "__main__":
    main()
