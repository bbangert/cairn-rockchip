#!/usr/bin/env python3
"""The COCO train2017 calibration subset: which images, and fetching only those.

`select` draws N image ids from instances_train2017.json with a fixed seed and writes them in draw
order, ten to a line, so the first K ids of the file are the K-image subset (nested subsets for the count sweep).
The id file is tracked; the annotation file is needed only to redraw it. `fetch` downloads each
listed image by itself (about 160 kB each) into DIR, which must be new or empty, and writes
DIR/MANIFEST.sha256 over what arrived.
usage: coco_calib.py select ANN_TRAIN_JSON [--n 500] [--seed S] [--out coco_calib_ids.txt]
       coco_calib.py fetch IDS_FILE DIR
"""
import argparse, hashlib, json, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IDS = os.path.join(HERE, "coco_calib_ids.txt")
SEED = 20261127
BASE = "http://images.cocodataset.org/train2017"  # the dataset's own host; no model is fetched from it


def draw(image_ids, n, seed=SEED):
    ids = np.array(sorted(set(int(i) for i in image_ids)), np.int64)
    # a negative n would reach the slice below and take all but the last -n ids
    if not 0 <= n <= len(ids):
        raise ValueError(f"asked for {n} images, the file has {len(ids)}")
    # sorted first: the draw must not depend on the order the annotation file lists images in
    return [int(i) for i in np.random.default_rng(seed).permutation(ids)[:n]]


def read_ids(path, n=None):
    ids = [int(t) for ln in open(path) if not ln.startswith("#") for t in ln.split()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ids")
    if n is not None and not 0 <= n <= len(ids):
        raise ValueError(f"{path} lists {len(ids)} ids, asked for {n}")
    return ids if n is None else ids[:n]


def file_name(image_id):
    return f"{image_id:012d}.jpg"


def cmd_select(a):
    with open(a.ann, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(a.ann) as f:
        images = json.load(f)["images"]
    ids = draw([im["id"] for im in images], a.n, a.seed)
    with open(a.out, "w") as f:
        f.write(f"# {a.n} of {len(images)} train2017 ids, numpy default_rng({a.seed}).permutation of the sorted ids;\n"
                f"# instances_train2017.json sha256 {digest}. First K ids = the K-image subset.\n")
        f.writelines(" ".join(map(str, ids[i:i + 10])) + "\n" for i in range(0, len(ids), 10))
    print(f"{len(ids)} ids -> {a.out}")


def cmd_fetch(a):
    ids = read_ids(a.ids)
    os.makedirs(a.dir, exist_ok=True)
    # Every entry, not only .jpg: prep_calib takes a stray .png in this directory as a calibration frame.
    names = {file_name(i) for i in ids}
    if not set(os.listdir(a.dir)) <= names | {n + ".part" for n in names} | {"MANIFEST.sha256"}:
        sys.exit(f"{a.dir} holds files this list does not name; give a new or empty directory")
    for n, i in enumerate(ids, 1):
        dest = os.path.join(a.dir, file_name(i))
        if os.path.exists(dest):
            continue
        tmp = dest + ".part"
        subprocess.run(["curl", "-fsS", "--retry", "3", "--max-time", "60", "-o", tmp, f"{BASE}/{file_name(i)}"], check=True)
        with open(tmp, "rb") as f:
            if f.read(3) != b"\xff\xd8\xff":
                os.remove(tmp)
                sys.exit(f"{file_name(i)}: not a JPEG")
        os.replace(tmp, dest)
        if n % 100 == 0:
            print(f"{n}/{len(ids)}", flush=True)
    with open(os.path.join(a.dir, "MANIFEST.sha256"), "w") as m:
        for i in ids:
            with open(os.path.join(a.dir, file_name(i)), "rb") as f:
                m.write(f"{hashlib.sha256(f.read()).hexdigest()}  {file_name(i)}\n")
    print(f"{len(ids)} images in {a.dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("usage:")[0].strip())
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("ann")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--out", default=IDS)
    f = sub.add_parser("fetch")
    f.add_argument("ids")
    f.add_argument("dir")
    a = ap.parse_args(argv)
    (cmd_select if a.cmd == "select" else cmd_fetch)(a)


if __name__ == "__main__":
    main()
