import hashlib
import io
import os
import re
import subprocess
import zipfile

import pytest

import sources


def test_lock_names_the_three_checkpoints_with_digests():
    ck = sources.load_lock()["mit"]["checkpoints"]
    assert [c["name"] for c in ck] == ["v9-t.pt", "v9-s.pt", "v9-m.pt"]
    for c in ck:
        assert re.fullmatch(r"[0-9a-f]{64}", c["sha256"])
        assert c["bytes"] > 0


def test_lock_pins_a_full_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", sources.load_lock()["mit"]["commit"])


def test_verify_file(tmp_path):
    p = tmp_path / "w.bin"
    p.write_bytes(b"abc")
    good = sources.sha256_file(p)
    sources.verify_file(p, 3, good)
    with pytest.raises(sources.SourceError, match="sha256"):
        sources.verify_file(p, 3, "0" * 64)
    with pytest.raises(sources.SourceError, match="bytes"):
        sources.verify_file(p, 4, good)


@pytest.mark.parametrize("name", ["../x", "/abs/x", "a/../../x", ".."])
def test_safe_member_rejects_escapes(name, tmp_path):
    with pytest.raises(sources.SourceError):
        sources.safe_member(name, tmp_path)


def test_safe_member_accepts_the_dataset_layout(tmp_path):
    assert sources.safe_member("val2017/000000000139.jpg", tmp_path) == str(tmp_path / "val2017" / "000000000139.jpg")


def test_check_disk_counts_the_headroom():
    sources.check_disk(free=10 + 5 + 7, download=10, extracted=5, headroom=7)
    with pytest.raises(sources.SourceError, match="free space"):
        sources.check_disk(free=10 + 5 + 7 - 1, download=10, extracted=5, headroom=7)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members:
            z.writestr(name, data)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    mit, coco = tmp_path / "mit", tmp_path / "coco"
    mit.mkdir()
    coco.mkdir()
    monkeypatch.setattr(sources, "MIT", str(mit))
    monkeypatch.setattr(sources, "COCO", str(coco))
    return mit, coco


@pytest.mark.parametrize("evil", ["../evil", "{abs}"])
def test_extract_with_one_unsafe_member_writes_nothing(roots, tmp_path, evil):
    _, coco = roots
    evil = evil.format(abs=tmp_path / "abs-evil")
    make_zip(coco / "a.zip", [("good.txt", b"x"), (evil, b"y")])
    with pytest.raises(sources.SourceError, match="unsafe|escapes"):
        sources.extract({"name": "a.zip", "members": None})
    assert [p.name for p in coco.rglob("*")] == ["a.zip"]
    assert not (tmp_path / "evil").exists() and not (tmp_path / "abs-evil").exists()


def test_extract_keeps_to_the_listed_members(roots):
    _, coco = roots
    make_zip(coco / "a.zip", [("annotations/i.json", b"{}"), ("annotations/other.json", b"{}")])
    sources.extract({"name": "a.zip", "members": ["annotations/i.json"]})
    assert (coco / "annotations" / "i.json").exists() and not (coco / "annotations" / "other.json").exists()
    with pytest.raises(sources.SourceError, match="lacks"):
        sources.extract({"name": "a.zip", "members": ["annotations/missing.json"]})


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_renames_the_part_file_into_place(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.urllib.request, "urlopen", lambda url, timeout: FakeResponse(b"abc"))
    dest = tmp_path / "w.bin"
    got = sources.download("http://example.invalid/w.bin", str(dest))
    assert got == (sha(b"abc"), hashlib.md5(b"abc").hexdigest())
    assert dest.read_bytes() == b"abc" and os.listdir(tmp_path) == ["w.bin"]


def test_an_interrupted_download_leaves_neither_part_nor_dest(tmp_path, monkeypatch):
    class Dies(FakeResponse):
        def read(self, n=-1):
            raise ConnectionResetError("cut")

    monkeypatch.setattr(sources.urllib.request, "urlopen", lambda url, timeout: Dies(b"abc"))
    with pytest.raises(ConnectionResetError):
        sources.download("http://example.invalid/w.bin", str(tmp_path / "w.bin"))
    assert os.listdir(tmp_path) == []


def fake_download(data):
    def fn(url, dest):
        with open(dest, "wb") as f:
            f.write(data)
        return sha(data), hashlib.md5(data).hexdigest()
    return fn


@pytest.mark.parametrize("nbytes,digest", [(4, sha(b"abc")), (3, "0" * 64)])
def test_a_bad_checkpoint_download_is_deleted_and_raises(roots, monkeypatch, nbytes, digest):
    mit, _ = roots
    monkeypatch.setattr(sources, "download", fake_download(b"abc"))
    spec = {"release_url": "http://example.invalid", "checkpoints": [{"name": "w.pt", "bytes": nbytes, "sha256": digest}]}
    with pytest.raises(sources.SourceError):
        sources.sync_checkpoints(spec)
    assert not (mit / "weights" / "w.pt").exists()


def test_a_bad_checkpoint_already_on_disk_is_reported_and_left(roots, monkeypatch):
    mit, _ = roots
    (mit / "weights").mkdir()
    (mit / "weights" / "w.pt").write_bytes(b"abc")
    monkeypatch.setattr(sources, "download", lambda *a: pytest.fail("must not download over an existing file"))
    spec = {"release_url": "x", "checkpoints": [{"name": "w.pt", "bytes": 3, "sha256": "0" * 64}]}
    with pytest.raises(sources.SourceError, match="sha256"):
        sources.sync_checkpoints(spec)
    assert (mit / "weights" / "w.pt").exists()


@pytest.mark.parametrize("field", ["bytes", "sha256"])
def test_a_bad_coco_archive_download_is_deleted(roots, monkeypatch, field):
    _, coco = roots
    monkeypatch.setattr(sources, "download", fake_download(b"abc"))
    a = {"name": "v.zip", "url": "http://example.invalid/v.zip", "bytes": 3, "sha256": sha(b"abc")}
    a[field] = 4 if field == "bytes" else "0" * 64
    with pytest.raises(sources.SourceError, match="deleted"):
        sources.fetch_archive({}, a)
    assert not (coco / "v.zip").exists()


def test_a_bad_coco_archive_already_on_disk_is_named_not_deleted(roots):
    _, coco = roots
    (coco / "v.zip").write_bytes(b"abc")
    a = {"name": "v.zip", "url": "x", "bytes": 3, "sha256": "0" * 64}
    with pytest.raises(sources.SourceError, match="delete .*v.zip"):
        sources.fetch_archive({}, a)
    assert (coco / "v.zip").exists()


def test_free_bytes_measures_the_nearest_existing_ancestor(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(sources.shutil, "disk_usage", lambda p: seen.append(p) or type("U", (), {"free": 7})())
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    assert sources.free_bytes(str(tmp_path / "link" / "not" / "yet")) == 7
    assert seen == [os.path.realpath(real)]


def git(*args, cwd):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=cwd,
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def tree(roots, monkeypatch):
    """A complete, good work/ in tmp dirs, and the lock that describes it."""
    mit, coco = roots
    clone = mit / "YOLO"
    clone.mkdir()
    git("init", "-q", cwd=clone)
    (clone / "f").write_text("x")
    git("add", "f", cwd=clone)
    git("commit", "-q", "-m", "c", cwd=clone)
    (mit / "weights").mkdir()
    (mit / "weights" / "w.pt").write_bytes(b"abc")
    (mit / "venv").mkdir()
    sources.write_stamp(str(mit / "venv" / sources.VENV_STAMP), sources.requirements_digest())
    (coco / "val2017").mkdir()
    (coco / "annotations").mkdir()
    for n in ("1.jpg", "2.jpg"):
        (coco / "val2017" / n).write_bytes(n.encode() * 10)
    (coco / "annotations" / "i.json").write_bytes(b"{}")
    n, digest = sources.image_manifest(str(coco / "val2017"))
    lock = {
        "mit": {"commit": git("rev-parse", "HEAD", cwd=clone),
                "checkpoints": [{"name": "w.pt", "bytes": 3, "sha256": sha(b"abc")}]},
        "coco": {"images": 2, "images_manifest": {"files": n, "sha256": digest},
                 "annotations_file": {"path": "annotations/i.json", "bytes": 2, "sha256": sha(b"{}")},
                 "archives": [{"name": "v.zip", "bytes": 1, "sha256": "0" * 64}]},
    }
    monkeypatch.setattr(sources, "load_lock", lambda: lock)
    monkeypatch.setattr(sources, "save_lock", lambda lk, path=None: None)
    return lock


def run_main(*args):
    return sources.main(["sources.py", *args])


def test_verify_passes_on_a_good_tree(tree, capsys):
    run_main("verify")
    out = capsys.readouterr().out
    assert "MISMATCH" not in out and "v.zip: absent" in out  # the zips are deleted after extraction


def test_verify_on_an_empty_work_fails_unless_absent_is_allowed(roots, tree, tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    (empty / "mit").mkdir(parents=True)
    (empty / "coco").mkdir()
    monkeypatch.setattr(sources, "MIT", str(empty / "mit"))
    monkeypatch.setattr(sources, "COCO", str(empty / "coco"))
    with pytest.raises(SystemExit, match="absent"):
        run_main("verify")
    run_main("verify", "--allow-absent")


@pytest.mark.parametrize("breakage", ["weights", "jpg", "annotations", "stamp", "clone-dirty", "clone-commit"])
def test_verify_fails_on_anything_present_and_wrong_even_with_allow_absent(tree, roots, breakage):
    mit, coco = roots
    {
        "weights": lambda: (mit / "weights" / "w.pt").write_bytes(b"abd"),
        "jpg": lambda: (coco / "val2017" / "2.jpg").write_bytes(b"trunc"),
        "annotations": lambda: (coco / "annotations" / "i.json").write_bytes(b"[]"),
        "stamp": lambda: sources.write_stamp(str(mit / "venv" / sources.VENV_STAMP), "0" * 64),
        "clone-dirty": lambda: (mit / "YOLO" / "f").write_text("changed"),
        "clone-commit": lambda: tree["mit"].update(commit="0" * 40),
    }[breakage]()
    for flags in ((), ("--allow-absent",)):
        with pytest.raises(SystemExit, match="differ"):
            run_main("verify", *flags)


def test_verify_rejects_unknown_arguments(tree):
    with pytest.raises(SystemExit, match="usage"):
        run_main("verify", "--bogus")
    with pytest.raises(SystemExit, match="usage"):
        run_main("coco", "--allow-absent")


def test_one_truncated_jpg_breaks_the_manifest(tree, roots):
    _, coco = roots
    sources.verify_images(tree["coco"])
    with open(coco / "val2017" / "1.jpg", "r+b") as f:
        f.truncate(5)
    with pytest.raises(sources.SourceError, match="truncated"):
        sources.verify_images(tree["coco"])


def test_coco_fast_path_needs_the_completion_stamp(tree, roots, monkeypatch):
    _, coco = roots
    assert not sources.coco_present(tree["coco"])
    monkeypatch.setattr(sources, "download", lambda *a: pytest.fail("an extracted tree must not be downloaded"))
    sources.cmd_coco(tree)  # extracted but unstamped: re-checked, stamped, nothing fetched
    assert sources.coco_present(tree["coco"])
    assert sources.read_stamp(str(coco / sources.COCO_STAMP)) == tree["coco"]["images_manifest"]["sha256"]


def test_coco_fast_path_does_not_rehash(tree, roots, monkeypatch):
    sources.cmd_coco(tree)
    monkeypatch.setattr(sources, "image_manifest", lambda *a: pytest.fail("fast path must not re-hash"))
    sources.cmd_coco(tree)


def test_coco_check_on_a_damaged_unstamped_tree_fails_and_stamps_nothing(tree, roots):
    _, coco = roots
    (coco / "val2017" / "1.jpg").write_bytes(b"trunc")
    with pytest.raises(sources.SourceError, match="truncated"):
        sources.cmd_coco(tree)
    assert not (coco / sources.COCO_STAMP).exists()


def test_first_fetch_records_the_manifest(tree, roots):
    tree["coco"]["images_manifest"] = {"files": None, "sha256": None}
    sources.check_images(tree)
    m = tree["coco"]["images_manifest"]
    assert m["files"] == 2 and re.fullmatch(r"[0-9a-f]{64}", m["sha256"])


def test_the_lock_records_the_val2017_manifest():
    m = sources.load_lock()["coco"]["images_manifest"]
    assert m["files"] == 5000 and re.fullmatch(r"[0-9a-f]{64}", m["sha256"])


def test_venv_stamp_states(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    with pytest.raises(sources.SourceError, match="no install stamp.*rm -r"):
        sources.check_venv(str(venv))
    sources.write_stamp(str(venv / sources.VENV_STAMP), "0" * 64)
    with pytest.raises(sources.SourceError, match="other requirements.*rm -r"):
        sources.check_venv(str(venv))
    sources.write_stamp(str(venv / sources.VENV_STAMP), sources.requirements_digest())
    sources.check_venv(str(venv))


def test_the_digest_follows_either_requirements_file(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("x")
    b.write_text("y")
    monkeypatch.setattr(sources, "BUILD_REQ", str(a))
    monkeypatch.setattr(sources, "MIT_REQ", str(b))
    before = sources.requirements_digest()
    b.write_text("z")
    assert sources.requirements_digest() != before


def test_sync_venv_refuses_an_unstamped_venv_without_touching_it(roots, monkeypatch):
    mit, _ = roots
    (mit / "venv").mkdir()
    (mit / "venv" / "half").write_text("x")
    monkeypatch.setattr(sources, "run", lambda *a, **k: pytest.fail("must not reinstall"))
    with pytest.raises(sources.SourceError, match="rm -r"):
        sources.sync_venv()
    assert (mit / "venv" / "half").exists()


def test_sync_venv_installs_the_build_tools_first_then_stamps(roots, monkeypatch):
    mit, _ = roots
    calls = []

    def fake_run(*cmd, **kw):
        calls.append(cmd)
        if cmd[1:3] == ("-m", "venv"):
            os.makedirs(cmd[3])

    monkeypatch.setattr(sources, "run", fake_run)
    sources.sync_venv()
    venv_cmd, build, main = calls
    assert build[1:4] == ("install", "--require-hashes", "-r") and build[-1] == sources.BUILD_REQ
    assert "--no-build-isolation" in main and "--require-hashes" in main and main[-1] == sources.MIT_REQ
    assert "--no-build-isolation" not in build
    sources.check_venv(str(mit / "venv"))


def test_sync_venv_failed_install_leaves_no_stamp(roots, monkeypatch):
    mit, _ = roots

    def fake_run(*cmd, **kw):
        if cmd[1:3] == ("-m", "venv"):
            os.makedirs(cmd[3])
        else:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(sources, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        sources.sync_venv()
    assert not (mit / "venv" / sources.VENV_STAMP).exists()
    with pytest.raises(sources.SourceError, match="no install stamp"):
        sources.sync_venv()


def test_venv_stamp_command_checks_the_freeze_against_the_pins(roots, monkeypatch):
    mit, _ = roots
    (mit / "venv").mkdir()
    want = {**sources.pins(sources.BUILD_REQ), **sources.pins(sources.MIT_REQ)}
    frozen = [f"{k}=={v}" for k, v in want.items() if k != "wheel"]  # a venv from before wheel was pinned
    monkeypatch.setattr(sources, "freeze", lambda venv: frozen)
    sources.cmd_venv_stamp({})
    sources.check_venv(str(mit / "venv"))

    os.remove(mit / "venv" / sources.VENV_STAMP)
    monkeypatch.setattr(sources, "freeze", lambda venv: frozen[1:] + ["numpy==1.0"] + ["x @ file:///y"])
    with pytest.raises(sources.SourceError, match="refusing to stamp"):
        sources.cmd_venv_stamp({})
    assert not (mit / "venv" / sources.VENV_STAMP).exists()


@pytest.mark.parametrize("path", [sources.BUILD_REQ, sources.MIT_REQ])
def test_every_requirement_is_hash_pinned(path):
    lines = open(path).read().splitlines()
    reqs = [i for i, l in enumerate(lines) if "==" in l and not l.startswith("#")]
    assert reqs
    for i in reqs:
        assert lines[i].endswith(" \\"), lines[i]
        j = i + 1
        assert lines[j].lstrip().startswith("--hash=sha256:"), lines[i]
        while lines[j].endswith("\\"):
            j += 1
            assert lines[j].lstrip().startswith("--hash=sha256:"), lines[i]


def test_pins_reads_local_version_labels():
    assert sources.pins(sources.MIT_REQ)["torch"] == "2.4.1+cpu"
