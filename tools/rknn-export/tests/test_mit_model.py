import hashlib
import os
import socket
import sys

import pytest

import mit_model


def test_block_network_refuses_connections_until_undone():
    # a listener on loopback: a refusal here can only come from the patch
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    addr = srv.getsockname()
    undo = mit_model.block_network()
    try:
        with pytest.raises(mit_model.NetworkBlocked):
            socket.create_connection(addr, timeout=1)
        with pytest.raises(mit_model.NetworkBlocked):
            socket.socket().connect_ex(addr)
    finally:
        undo()
    socket.create_connection(addr, timeout=1).close()
    srv.close()


def test_load_rejects_unknown_model_name():
    with pytest.raises(ValueError):
        mit_model.load("v9-c", "/nope.pt", "/nope")


def test_strict_load_names_missing_and_stray_keys(tmp_path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    layers = torch.nn.ModuleList([torch.nn.Conv2d(1, 2, 1), torch.nn.Conv2d(2, 2, 1)])
    net = SimpleNamespace(model=layers)
    good = {k: v for k, v in layers.state_dict().items()}

    def ckpt(d):
        p = str(tmp_path / "w.pt")
        torch.save(d, p)
        return p

    rep = mit_model._load_strict(net, ckpt({**good, "2.weight": torch.zeros(1)}))
    assert rep["loaded"] == 4 and rep["unused"] == 1 and rep["unused_layers"] == [2]
    with pytest.raises(ValueError, match="1.bias"):
        mit_model._load_strict(net, ckpt({k: v for k, v in good.items() if k != "1.bias"}))
    with pytest.raises(ValueError, match="0.weight"):
        mit_model._load_strict(net, ckpt({**good, "0.weight": torch.zeros(5)}))
    with pytest.raises(ValueError, match="1.extra"):
        mit_model._load_strict(net, ckpt({**good, "1.extra": torch.zeros(1)}))


def test_block_network_undo_restores_wandb_mode(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    undo = mit_model.block_network()
    assert os.environ["WANDB_MODE"] == "disabled"
    undo()
    assert "WANDB_MODE" not in os.environ
    monkeypatch.setenv("WANDB_MODE", "online")
    mit_model.block_network()()
    assert os.environ["WANDB_MODE"] == "online"


@pytest.fixture
def weights(tmp_path, monkeypatch):
    data = b"weights"
    (tmp_path / "v9-t.pt").write_bytes(data)
    lock = {"mit": {"commit": "abc", "checkpoints": [{"name": "v9-t.pt", "sha256": hashlib.sha256(data).hexdigest()},
                                                      {"name": "v9-s.pt", "sha256": "0" * 64}]}}
    monkeypatch.setattr(mit_model, "WEIGHTS", str(tmp_path))
    monkeypatch.setattr(mit_model.sources, "load_lock", lambda: lock)
    return tmp_path, lock


def test_checkpoint_returns_path_sha_and_commit(weights):
    path, sha, commit = mit_model.checkpoint("v9-t")
    assert path == str(weights[0] / "v9-t.pt") and sha == weights[1]["mit"]["checkpoints"][0]["sha256"]
    assert commit == "abc"


def test_checkpoint_name_missing_from_lock(weights):
    with pytest.raises(ValueError, match="v9-m.pt is not in sources.lock"):
        mit_model.checkpoint("v9-m")


def test_checkpoint_file_missing(weights):
    with pytest.raises(ValueError, match="v9-s.pt: checkpoint missing"):
        mit_model.checkpoint("v9-s")


def test_checkpoint_sha_mismatch(weights):
    (weights[0] / "v9-t.pt").write_bytes(b"other")
    with pytest.raises(ValueError, match="lock says"):
        mit_model.checkpoint("v9-t")


def _fake_net(torch, neck):
    # _fit_class_neck rebuilds the convs with type(class_conv[0])(cin, cout, 3)
    class Conv(torch.nn.Module):
        def __init__(self, cin, cout, k):
            super().__init__()
            self.conv = torch.nn.Conv2d(cin, cout, k)

    head = torch.nn.Module()
    head.class_conv = torch.nn.Sequential(Conv(4, neck, 3), Conv(neck, neck, 3), torch.nn.Conv2d(neck, 80, 1))
    layer = torch.nn.Module()
    layer.layer_type = "MultiheadDetection"
    layer.heads = torch.nn.ModuleList([head])
    other = torch.nn.Module()
    return type("Net", (), {"model": [other, layer]})(), head


def test_fit_class_neck_follows_checkpoint_width():
    torch = pytest.importorskip("torch")
    net, head = _fake_net(torch, 128)
    ckpt = {"1.heads.0.class_conv.0.conv.weight": torch.zeros(64, 4, 3, 3)}
    assert mit_model._fit_class_neck(net, ckpt) == {"1.heads.0": 64}
    first, second, last = head.class_conv
    assert (first.conv.in_channels, first.conv.out_channels) == (4, 64)
    assert (second.conv.in_channels, second.conv.out_channels) == (64, 64)
    assert (last.in_channels, last.out_channels) == (64, 80)


def test_fit_class_neck_leaves_matching_width_and_other_keys_alone():
    torch = pytest.importorskip("torch")
    net, head = _fake_net(torch, 128)
    before = head.class_conv
    ckpt = {"1.heads.0.class_conv.0.conv.weight": torch.zeros(128, 4, 3, 3)}
    assert mit_model._fit_class_neck(net, ckpt) == {} and head.class_conv is before
    assert mit_model._fit_class_neck(net, {}) == {} and head.class_conv is before


def _git(*args, cwd):
    import subprocess

    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=cwd,
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    clone = tmp_path / "YOLO"
    clone.mkdir()
    _git("init", "-q", cwd=clone)
    (clone / "f").write_text("x")
    _git("add", "f", cwd=clone)
    _git("commit", "-q", "-m", "c", cwd=clone)
    lock = {"mit": {"commit": _git("rev-parse", "HEAD", cwd=clone)}}
    monkeypatch.setattr(mit_model, "MIT_DIR", str(clone))
    monkeypatch.setattr(mit_model.sources, "load_lock", lambda: lock)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return clone


def _refuses(clone, match):
    with pytest.raises(SystemExit, match=match):
        mit_model._build("v9-t", str(clone))
    assert str(clone) not in sys.path  # refused before the upstream import path is set up


def test_build_refuses_a_dirty_pinned_clone(pinned):
    (pinned / "f").write_text("edited")
    _refuses(pinned, "not the pinned one.*local changes")


def test_build_refuses_a_pinned_clone_on_another_commit(pinned):
    (pinned / "g").write_text("y")
    _git("add", "g", cwd=pinned)
    _git("commit", "-q", "-m", "d", cwd=pinned)
    _refuses(pinned, "not the pinned one.*lock says")


def test_build_refuses_a_pinned_clone_that_is_not_a_repo(pinned, tmp_path):
    import shutil

    shutil.rmtree(pinned / ".git")
    _refuses(pinned, "not the pinned one")


def test_build_does_not_check_other_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(mit_model.sources, "load_lock", lambda: pytest.fail("checked a non-pinned dir"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        mit_model._build("v9-t", str(tmp_path))
    except Exception:
        pass  # past the check; whatever stops it after that is the upstream import or config
