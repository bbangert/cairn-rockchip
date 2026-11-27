#!/usr/bin/env python3
"""Digest one conversion: op targets from the operator table in NAME.out (the
captured stdout; NAME.build.log has none), build warnings, cosine of the last
three layers from accuracy_analysis. usage: summarize.py NAME...
optable and warnings are what convert.py records in NAME.json."""
import collections, json, os, re, sys

# the graph's own boundary: InputOperator is always CPU, and is not an op the model runs
BOUNDARY = ("InputOperator", "OutputOperator")


def optable(lines):
    """-> [(id, op, dtype, target)] of the last operator table (the toolkit prints one per pass).

    A line inside the table that is neither a row nor its rules raises: dropping it would let an op on
    CPU in a row of another shape pass convert.py's --refuse-cpu."""
    rows, intable = [], False
    for ln in lines:
        if "ID   OpType" in ln and "Target" in ln:
            intable, rows = True, []; continue   # keep the last table
        if intable:
            m = re.match(r"D RKNN: \[[\d:.]+\] (\d+)\s+(\S+)\s+(\S+)\s+(NPU|CPU|GPU)\s", ln)
            if m: rows.append(m.groups())
            elif re.match(r"D RKNN: \[[\d:.]+\] -+\s*$", ln):
                if rows: intable = False  # the rule under the header comes before any row
            else: raise ValueError(f"unreadable operator table line: {ln[:160]!r}")
    if intable: raise ValueError("operator table has no closing rule")
    return rows


def off_npu(rows):
    return collections.Counter(f"{r[1]}[{r[2]}]" for r in rows if r[3] != "NPU" and r[1] not in BOUNDARY)


def warnings(lines):
    return collections.Counter(re.sub(r"\[[\d:.]+\] ", "", l)[:150] for l in lines if l.startswith("W "))


if __name__ == "__main__":
    W = os.environ.get("RKNN_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "work", "rknn")
    for name in sys.argv[1:]:
        log = open(os.path.join(W, name + ".out"), errors="replace").read().splitlines()
        res = json.load(open(os.path.join(W, name + ".json")))
        rows = optable(log)
        tgt = collections.Counter(r[3] for r in rows)
        cpu = collections.Counter(f"{r[1]}[{r[2]}]" for r in rows if r[3] != "NPU")
        warns = warnings(log)
        last = [r for r in rows if r[1] == "OutputOperator"]
        print(f"== {name}  steps={res['steps']} size_mb={res.get('size_mb')} sim_out={res.get('sim_outputs')}")
        print("   targets", dict(tgt), "CPU/other ops", dict(cpu), "outputs", [(r[2], r[3]) for r in last])
        for w, n in warns.most_common(8): print(f"   warn x{n}: {w}")
        ea = os.path.join(W, name + ".snap", "error_analysis.txt")
        if os.path.exists(ea):
            L = [l for l in open(ea) if re.search(r"\d\.\d{5} \|", l)]
            print("   cos(entire|single) last 3 layers:"); [print("     ", l.rstrip()) for l in L[-3:]]
