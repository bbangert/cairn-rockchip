#!/usr/bin/env python3
"""Hash a rung directory. usage: manifest.py DIR [BUILD_DIR]
Writes DIR/MANIFEST.sha256 (sha256sum format, bare names: `sha256sum -c` works
from inside DIR, busybox included) and DIR/MANIFEST.json. Build facts come from
BUILD_DIR/NAME.json (default DIR/../final/build); a rung's twin is the .onnx in
DIR whose hash is the one the rung was converted from. Exits non-zero, writing
nothing, unless every .rknn has a build JSON with all steps ok, no recorded
simulator error, no failed parity verdict, and an rknn_sha256 equal to the
file's hash.
The manifest is a public model repo's licence statement, so MIT is never read
off a name (the GPL-3.0 YOLOv9 shares it): a twin is MIT only when
BUILD_DIR/X.onnx.json, written by export_mit_yolov9.py, records the twin's own
hash and names the model, commit and checkpoint hash; an .rknn inherits licence
and `source` from its twin. A yolov9* file without that provenance refuses.
Otherwise yolox* is Apache-2.0, the rest AGPL-3.0."""
import hashlib, json, os, sys

d = os.path.abspath(sys.argv[1])
build = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "..", "final", "build"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
files = sorted(f for f in os.listdir(d) if not f.startswith("MANIFEST"))
assert files and all(f.endswith((".rknn", ".onnx")) for f in files), files
hashes = {f: sha(os.path.join(d, f)) for f in files}
twins = {h: f for f, h in hashes.items() if f.endswith(".onnx")}
rows, bad = [], []
mit = {}
for f in files:
    sp = os.path.join(build, f + ".json")
    if not f.endswith(".onnx") or not os.path.exists(sp): continue
    try: sc = json.load(open(sp))
    except ValueError: sc = None
    if not isinstance(sc, dict) or sc.get("onnx_sha256") != hashes[f]:
        bad.append(f"{f}: sidecar {sp} does not record this file's sha256"); continue
    # a hash alone would label the file MIT with nothing saying which checkpoint it came from
    miss = [k for k, ok in (("name", sc.get("name") in ("v9-t", "v9-s", "v9-m")),
                            ("mit_commit", isinstance(sc.get("mit_commit"), str) and sc["mit_commit"]),
                            ("checkpoint_sha256", isinstance(sc.get("checkpoint_sha256"), str) and sc["checkpoint_sha256"])) if not ok]
    if miss: bad.append(f"{f}: sidecar {sp} lacks a valid {', '.join(miss)}"); continue
    mit[f] = {"repo": "MultimediaTechLab/YOLO"} | {k: sc[o] for k, o in (
        ("commit", "mit_commit"), ("checkpoint", "name"), ("checkpoint_sha256", "checkpoint_sha256"))}
for f in files:
    row = {"file": f, "sha256": hashes[f], "bytes": os.path.getsize(os.path.join(d, f)),
           "role": "rknn" if f.endswith(".rknn") else "twin",
           "license": "Apache-2.0" if f.startswith("yolox") else "AGPL-3.0"}
    if f in mit: row.update(license="MIT", source=mit[f])
    elif row["role"] == "twin" and f.startswith("yolov9"): bad.append(f"{f}: MIT sidecar {build}/{f}.json missing"); continue
    if row["role"] == "rknn":
        jp = os.path.join(build, f[:-5] + ".json")
        if not os.path.exists(jp): bad.append(f"{f}: no build json {jp}"); continue
        b = json.load(open(jp))
        # the json is keyed by name only; the hash is what says it describes these bytes
        if any(v != "ok" for v in b["steps"].values()): bad.append(f"{f}: build steps not ok {b['steps']}"); continue
        # convert.py records a simulator/parity exception here while every step stays ok
        if isinstance(b.get("sim_outputs"), str): bad.append(f"{f}: build sim_outputs {b['sim_outputs']}"); continue
        # a failed parity verdict fails convert.py's exit, not a step, and leaves the .rknn on disk
        if b.get("parity_pass") is False: bad.append(f"{f}: build failed its parity verdict"); continue
        if b.get("rknn_sha256") != hashes[f]: bad.append(f"{f}: build json rknn_sha256 {b.get('rknn_sha256')} != file {hashes[f]}"); continue
        tw = twins[b["source_sha256"]]
        if tw in mit: row.update(license="MIT", source=mit[tw])
        elif f.startswith("yolov9"): bad.append(f"{f}: MIT sidecar for its twin {tw} missing"); continue
        row.update(twin=tw, toolkit=b["toolkit"], target=b["target_platform"], mode=b["mode"],
                   quantized_algorithm=b["quantized_algorithm"], calib_set_id=b["calib_set_id"],
                   calib_policy=b["calib_policy"], calib_count=b.get("calib_count"), source_sha256=b["source_sha256"])
    rows.append(row)
if bad: sys.exit("manifest not written:\n" + "\n".join(bad))
open(os.path.join(d, "MANIFEST.sha256"), "w").write("".join(f"{h}  {f}\n" for f, h in hashes.items()))
json.dump(rows, open(os.path.join(d, "MANIFEST.json"), "w"), indent=1)
print(len(rows), "files ->", d)
