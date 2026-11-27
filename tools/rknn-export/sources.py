#!/usr/bin/env python3
"""Fetch third-party code, weights and eval data exactly as recorded in sources.lock.

Trust: v9-t's hash equals the digest GitHub publishes for the release asset. GitHub publishes
none for v9-s and v9-m, so theirs are trust-on-first-use, like the COCO hashes from
cocodataset.org over plain http (its official link, cross-checked by content counts). What
vouches for all three checkpoints is the result: each reproduces its published COCO score.
The first COCO fetch records the hashes, and the per-file val2017 manifest, in the lock.
Nothing fetched here enters git; work/ and venv/ are ignored, as directories or symlinks.
A tree counts as present only when a stamp written after a successful check says so: the
venv's holds the sha256 of the requirements files, the COCO images' the manifest digest in
the lock. `verify` re-hashes every fetched file, compares the venv's pip freeze with the
requirements (the stamp alone is only the fetch fast path), and fails on anything absent
unless --allow-absent.
usage: sources.py {mit|coco|verify [--allow-absent]|venv-stamp}
"""
import hashlib, json, os, re, shutil, subprocess, sys, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.path.join(HERE, "sources.lock")
MIT = os.path.join(HERE, "work", "mit")
COCO = os.path.join(HERE, "work", "coco")
BUILD_REQ = os.path.join(HERE, "requirements-mit-build.txt")
MIT_REQ = os.path.join(HERE, "requirements-mit.txt")
VENV_STAMP = ".requirements.sha256"
COCO_STAMP = ".images-manifest.sha256"
HEADROOM = 20 * 2**30  # the volume is shared with model builds; a full disk there corrupts a conversion


class SourceError(Exception):
    pass


def load_lock(path=LOCK):
    with open(path) as f:
        return json.load(f)


def save_lock(lock, path=LOCK):
    with open(path, "w") as f:
        json.dump(lock, f, indent=2)
        f.write("\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path, nbytes, sha256):
    got = os.path.getsize(path)
    if got != nbytes:
        raise SourceError(f"{path}: {got} bytes, lock says {nbytes}")
    digest = sha256_file(path)
    if digest != sha256:
        raise SourceError(f"{path}: sha256 {digest}, lock says {sha256}")


def safe_member(name, dest):
    # Checked on the raw name: zipfile.extract drops ".." components, so a/../x would be accepted
    # here after normalising yet land at a/x. A backslash is refused too: it separates components
    # on Windows but is a filename character on POSIX, and the two readings must not differ.
    if "\\" in name or ".." in name.split("/") or os.path.isabs(name):
        raise SourceError(f"unsafe archive member {name!r}")
    target = os.path.normpath(os.path.join(dest, name))
    # Resolved, not lexical: an existing symlink under dest (or at the member's own path) would
    # carry the write elsewhere. dest itself may be a symlink (work/ is), hence realpath on both.
    root = os.path.realpath(dest)
    if os.path.commonpath([root, os.path.realpath(target)]) != root:
        raise SourceError(f"archive member {name!r} escapes {dest}")
    return target


def check_disk(free, download, extracted, headroom=HEADROOM):
    need = download + extracted + headroom
    if free < need:
        raise SourceError(f"free space {free} < needed {need} (download {download} + extracted {extracted} + headroom {headroom})")


def free_bytes(path):
    # work/ may be a symlink onto another volume: measure where the bytes will land.
    p = os.path.abspath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(os.path.realpath(p)).free


def download(url, dest):
    # Returns (sha256, md5): md5 is only for cross-checking against what the publisher lists.
    # The rename keeps a killed download from looking like a finished one.
    sha, md5 = hashlib.sha256(), hashlib.md5()
    part = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                sha.update(chunk)
                md5.update(chunk)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
    os.replace(part, dest)
    return sha.hexdigest(), md5.hexdigest()


def run(*cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def step(msg):
    print(msg, flush=True)


def check_clone(mit, clone):
    head = subprocess.run(["git", "-C", clone, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != mit["commit"]:
        raise SourceError(f"{clone} is at {head}, lock says {mit['commit']}")
    dirty = subprocess.run(["git", "-C", clone, "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    if dirty.strip():
        raise SourceError(f"{clone} has local changes:\n{dirty}")
    return head


def sync_clone(mit):
    clone = os.path.join(MIT, "YOLO")
    if not os.path.isdir(clone):
        os.makedirs(MIT, exist_ok=True)
        run("git", "clone", "--quiet", mit["url"], clone)
        run("git", "-C", clone, "checkout", "--quiet", "--detach", mit["commit"])
        step(f"clone: {mit['repo']} at {mit['commit']}")
    step(f"clone: {check_clone(mit, clone)} clean")


def sync_checkpoints(mit):
    wdir = os.path.join(MIT, "weights")
    os.makedirs(wdir, exist_ok=True)
    for c in mit["checkpoints"]:
        path = os.path.join(wdir, c["name"])
        fetched = not os.path.exists(path)
        if fetched:
            download(f"{mit['release_url']}/{c['name']}", path)
        try:
            verify_file(path, c["bytes"], c["sha256"])
        except SourceError:
            if fetched:
                os.remove(path)  # the next run downloads again instead of failing on the same bad file
            raise
        step(f"weights: {c['name']} {'downloaded, ' if fetched else ''}verified")


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def pins(path):
    out = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"([A-Za-z0-9._-]+)==(\S+?)(?:\s|\\|$)", line)
            if m:
                out[norm(m[1])] = m[2]
    return out


def requirements_digest():
    h = hashlib.sha256()
    for path in (BUILD_REQ, MIT_REQ):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def write_stamp(path, digest):
    with open(path, "w") as f:
        f.write(digest + "\n")


def read_stamp(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def check_venv(venv):
    got = read_stamp(os.path.join(venv, VENV_STAMP))
    if got == requirements_digest():
        return
    why = "has no install stamp (an install that did not finish?)" if got is None else "was installed from other requirements"
    # Not deleted here: the venv is hours of someone's installs.
    raise SourceError(f"{venv} {why}; rm -r it and rerun `sources.py mit`")


def check_installed_venv(venv):
    # The stamp only says an install finished once; verify also asks what is installed now.
    check_venv(venv)
    diff = venv_diff(venv)
    if diff:
        raise SourceError(f"{venv} no longer matches the requirements: {diff[:8]}")


def sync_venv():
    venv = os.path.join(MIT, "venv")
    if os.path.isdir(venv):
        check_venv(venv)
        step("venv: present, installed from the current requirements")
        return
    pip = os.path.join(venv, "bin", "pip")
    run("python3.11", "-m", "venv", venv)
    # antlr4-python3-runtime is sdist-only: build it with the pinned tooling, not with whatever an isolated build env resolves.
    run(pip, "install", "--require-hashes", "-r", BUILD_REQ)
    run(pip, "install", "--require-hashes", "--no-build-isolation", "-r", MIT_REQ)
    write_stamp(os.path.join(venv, VENV_STAMP), requirements_digest())
    step("venv: created from requirements-mit-build.txt and requirements-mit.txt")


def freeze(venv):
    pip = os.path.join(venv, "bin", "pip")
    try:
        out = subprocess.run([pip, "freeze", "--all"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise SourceError(f"{venv}: cannot run {pip} freeze ({e})")
    return out.splitlines()


def venv_diff(venv):
    want = {**pins(BUILD_REQ), **pins(MIT_REQ)}
    got, odd = {}, []
    for line in freeze(venv):
        name, eq, ver = line.partition("==")
        if eq:
            got[norm(name)] = ver
        else:
            odd.append(line)
    # wheel only builds the sdist; a venv installed before it was pinned lacks it
    return sorted(odd + [k for k in want.keys() | got.keys() if k != "wheel" and want.get(k) != got.get(k)])


def cmd_venv_stamp(lock):
    venv = os.path.join(MIT, "venv")
    if not os.path.isdir(venv):
        raise SourceError(f"{venv} does not exist")
    diff = venv_diff(venv)
    if diff:
        raise SourceError(f"{venv} does not match the requirements, refusing to stamp it: {diff[:8]}")
    write_stamp(os.path.join(venv, VENV_STAMP), requirements_digest())
    step("venv: matches the requirements, stamped")


def cmd_mit(lock):
    sync_clone(lock["mit"])
    sync_checkpoints(lock["mit"])
    sync_venv()


def image_manifest(imgs):
    # name + content hash per file: a count alone trusts the truncated last file of a killed extraction.
    h, n = hashlib.sha256(), 0
    for name in sorted(os.listdir(imgs)):
        if name.endswith(".jpg"):
            h.update(f"{name} {sha256_file(os.path.join(imgs, name))}\n".encode())
            n += 1
    return n, h.hexdigest()


def coco_extracted(coco):
    imgs = os.path.join(COCO, "val2017")
    ann = os.path.join(COCO, coco["annotations_file"]["path"])
    n = len([f for f in os.listdir(imgs) if f.endswith(".jpg")]) if os.path.isdir(imgs) else 0
    return n == coco["images"] and os.path.isfile(ann)


def coco_present(coco):
    want = coco["images_manifest"]["sha256"]
    return want is not None and coco_extracted(coco) and read_stamp(os.path.join(COCO, COCO_STAMP)) == want


def verify_images(coco):
    m = coco["images_manifest"]
    imgs = os.path.join(COCO, "val2017")
    n, digest = image_manifest(imgs)
    if m["sha256"] is None:
        raise SourceError("the lock records no val2017 manifest digest")
    if (n, digest) != (m["files"], m["sha256"]):
        raise SourceError(f"{imgs}: {n} jpgs with manifest {digest}, lock says {m['files']} with {m['sha256']}; "
                          f"an image is missing, truncated or altered: rm -r it and rerun `sources.py coco`")


def check_images(lock):
    coco = lock["coco"]
    m = coco["images_manifest"]
    if m["sha256"] is None:
        n, digest = image_manifest(os.path.join(COCO, "val2017"))
        m["files"], m["sha256"] = n, digest
        save_lock(lock)
        step(f"val2017: manifest {digest} over {n} jpgs recorded in the lock (first fetch)")
    verify_images(coco)
    write_stamp(os.path.join(COCO, COCO_STAMP), m["sha256"])
    step(f"val2017: {m['files']} jpgs match the manifest in the lock")


def fetch_archive(lock, a):
    path = os.path.join(COCO, a["name"])
    md5 = None
    fetched = not os.path.exists(path)

    def bad(msg):
        if fetched:
            os.remove(path)
            msg += "; deleted, rerun to download again"
        else:
            msg += f"; delete {path} to download it again"
        raise SourceError(msg)

    if fetched:
        step(f"download: {a['url']}")
        sha, md5 = download(a["url"], path)
    else:
        sha = sha256_file(path)
    if os.path.getsize(path) != a["bytes"]:
        bad(f"{path}: {os.path.getsize(path)} bytes, lock says {a['bytes']}")
    if a["sha256"] is None:
        a["sha256"] = sha
        save_lock(lock)
        step(f"{a['name']}: sha256 {sha} recorded in the lock (first fetch)")
    elif a["sha256"] != sha:
        bad(f"{path}: sha256 {sha}, lock says {a['sha256']}")
    else:
        step(f"{a['name']}: sha256 verified")
    if md5:
        step(f"{a['name']}: md5 {md5}")


def extract(a):
    only = a["members"]
    with zipfile.ZipFile(os.path.join(COCO, a["name"])) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir() and (only is None or i.filename in only)]
        if only is not None and sorted(names) != sorted(only):
            raise SourceError(f"{a['name']} lacks {set(only) - set(names)}")
        for n in names:
            safe_member(n, COCO)  # all checked before any is written
        for n in names:
            z.extract(n, COCO)
    step(f"extract: {a['name']} ({len(names)} files)")


def check_counts(lock):
    coco = lock["coco"]
    imgs = [f for f in os.listdir(os.path.join(COCO, "val2017")) if f.endswith(".jpg")]
    if len(imgs) != coco["images"]:
        raise SourceError(f"val2017 has {len(imgs)} jpgs, expected {coco['images']}")
    ann = os.path.join(COCO, coco["annotations_file"]["path"])
    with open(ann) as f:
        d = json.load(f)
    if len(d["images"]) != coco["images"] or len(d["categories"]) != coco["categories"]:
        raise SourceError(f"{ann}: {len(d['images'])} images / {len(d['categories'])} categories, expected {coco['images']} / {coco['categories']}")
    step(f"counts: {len(imgs)} jpgs, {len(d['images'])} images and {len(d['categories'])} categories in the json")


def record_annotations(lock):
    af = lock["coco"]["annotations_file"]
    path = os.path.join(COCO, af["path"])
    if af["sha256"] is None:
        af["bytes"], af["sha256"] = os.path.getsize(path), sha256_file(path)
        save_lock(lock)
        step(f"{af['path']}: {af['bytes']} bytes, sha256 {af['sha256']} recorded in the lock (first fetch)")
    else:
        verify_file(path, af["bytes"], af["sha256"])
        step(f"{af['path']}: verified")


def cmd_coco(lock, free=None):
    coco = lock["coco"]
    if coco_present(coco):
        step("coco: val2017 and instances_val2017.json already present")
        record_annotations(lock)
        return
    if coco_extracted(coco):
        step("coco: extracted tree without a completion stamp; checking it")
        check_images(lock)
        record_annotations(lock)
        return
    arcs = coco["archives"]
    dl = sum(a["bytes"] for a in arcs)
    ex = sum(a["extracted_bytes"] for a in arcs)
    free = free_bytes(COCO) if free is None else free
    step(f"coco: about to fetch {dl} bytes, extracting about {ex} bytes; {free} bytes free")
    check_disk(free, dl, ex)
    os.makedirs(COCO, exist_ok=True)
    for a in arcs:
        fetch_archive(lock, a)
    for a in arcs:
        extract(a)
    check_counts(lock)
    check_images(lock)
    record_annotations(lock)
    for a in arcs:
        os.remove(os.path.join(COCO, a["name"]))
    step("coco: zips deleted; their hashes stay in sources.lock")


def cmd_verify(lock, allow_absent=False):
    bad = absent = 0

    def check(label, path, fn, required=True):
        nonlocal bad, absent
        if not os.path.exists(path):
            step(f"{label}: absent")
            absent += required
            return
        try:
            fn()
            step(f"{label}: ok")
        except SourceError as e:
            bad += 1
            step(f"{label}: MISMATCH {e}")

    def file_check(path, nbytes, sha):
        def fn():
            if sha is None:
                raise SourceError(f"{path}: present but the lock records no sha256")
            verify_file(path, nbytes, sha)
        return fn

    mit, coco = lock["mit"], lock["coco"]
    clone = os.path.join(MIT, "YOLO")
    check("YOLO clone", clone, lambda: check_clone(mit, clone))
    for c in mit["checkpoints"]:
        path = os.path.join(MIT, "weights", c["name"])
        check(c["name"], path, file_check(path, c["bytes"], c["sha256"]))
    venv = os.path.join(MIT, "venv")
    check("venv", venv, lambda: check_installed_venv(venv))
    check("val2017", os.path.join(COCO, "val2017"), lambda: verify_images(coco))
    af = coco["annotations_file"]
    path = os.path.join(COCO, af["path"])
    check(af["path"], path, file_check(path, af["bytes"], af["sha256"]))
    for a in coco["archives"]:  # deleted after extraction, so absent is the normal state
        path = os.path.join(COCO, a["name"])
        check(a["name"], path, file_check(path, a["bytes"], a["sha256"]), required=False)
    if bad:
        raise SourceError(f"{bad} item(s) differ from the lock")
    if absent and not allow_absent:
        raise SourceError(f"{absent} item(s) absent; fetch them, or pass --allow-absent")


def main(argv):
    cmds = {"mit": cmd_mit, "coco": cmd_coco, "verify": cmd_verify, "venv-stamp": cmd_venv_stamp}
    args = argv[1:]
    if not args or args[0] not in cmds or args[1:] not in ([], ["--allow-absent"] if args[0] == "verify" else []):
        sys.exit(__doc__.strip().splitlines()[-1])
    try:
        cmds[args[0]](load_lock(), *([True] if args[1:] else []))
    except SourceError as e:
        sys.exit(f"error: {e}")
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: {' '.join(map(str, e.cmd))} exited {e.returncode}")


if __name__ == "__main__":
    main(sys.argv)
