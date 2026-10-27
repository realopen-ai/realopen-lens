"""Release gates reject incomplete signing and unsuccessful acceptance evidence."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "packaging" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"lens_packaging_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "arguments",
    [
        ["--release"],
        ["--release", "--identity", "-", "--notary-profile", "profile"],
        ["--identity", "Developer ID Application: Example"],
        ["--notary-profile", "profile"],
    ],
)
def test_release_requires_developer_identity_and_notary_profile(arguments):
    with pytest.raises(SystemExit):
        script("build_macos").options(arguments)


def test_development_and_release_modes_are_explicit():
    build = script("build_macos")
    assert not build.options([]).release
    assert build.options(
        [
            "--release",
            "--identity",
            "Developer ID Application: Example",
            "--notary-profile",
            "profile",
        ]
    ).release


def test_missing_signing_identity_fails_before_build_or_upload(monkeypatch):
    build = script("build_macos")
    monkeypatch.setattr(build.sys, "platform", "darwin")
    monkeypatch.setattr(
        build.sys,
        "argv",
        [
            "build_macos.py",
            "--release",
            "--identity",
            "Developer ID Application: Example",
            "--notary-profile",
            "example",
        ],
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="0 valid identities found")

    monkeypatch.setattr(build.subprocess, "run", run)
    with pytest.raises(SystemExit, match="identity is not installed"):
        build.main()
    assert len(commands) == 1 and commands[0][0] == "/usr/bin/security"


@pytest.mark.parametrize("change", ["empty", "wrong_image", "missing_ocr", "degraded"])
def test_inference_check_requires_correct_image_and_ocr(change, tmp_path):
    smoke = script("smoke")
    source = tmp_path / "invoice.png"
    result = {
        "paths": [str(source)],
        "text": "Invoice ABC-123 pizza",
        "status": "ready",
        "issues": [],
    }
    smoke.verify_search([result], source)
    if change == "wrong_image":
        result["paths"] = ["another.png"]
    elif change == "missing_ocr":
        result["text"] = ""
    elif change == "degraded":
        result["status"] = "degraded"
    with pytest.raises(ValueError):
        smoke.verify_search([] if change == "empty" else [result], source)


@pytest.mark.parametrize("host", ["darwin", "win32", "linux"])
def test_acceptance_failure_retains_checksum_and_failed_stage(tmp_path, monkeypatch, host):
    acceptance = script("acceptance")
    monkeypatch.setattr(acceptance.sys, "platform", host)
    monkeypatch.setattr(acceptance.platform, "platform", lambda: "test-host")
    monkeypatch.setattr(
        acceptance.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1)
    )
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"exact candidate bytes")
    output = tmp_path / "evidence"
    with pytest.raises(RuntimeError, match="package exited 1"):
        acceptance.validate(archive, output)
    report = json.loads((output / "acceptance.json").read_text(encoding="utf-8"))
    assert report["archive_sha256"] == acceptance.digest(archive)
    assert report["status"] == report["checks"][0]["status"] == "failed"
    assert report["inference"] == "not_run"


@pytest.mark.parametrize("host", ["darwin", "win32", "linux"])
def test_package_only_acceptance_does_not_claim_inference(tmp_path, monkeypatch, host):
    acceptance = script("acceptance")
    monkeypatch.setattr(acceptance.sys, "platform", host)
    monkeypatch.setattr(acceptance.platform, "platform", lambda: "test-host")
    monkeypatch.setattr(
        acceptance.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0)
    )
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"candidate")
    report = acceptance.validate(archive, tmp_path / "evidence")
    assert report["status"] == "passed" and report["inference"] == "not_run"
    assert report["notarization"] == ("not_applicable" if host != "darwin" else "not_checked")


@pytest.mark.parametrize("host", ["darwin", "win32", "linux"])
def test_acceptance_rejects_archive_mutation(tmp_path, monkeypatch, host):
    acceptance = script("acceptance")
    monkeypatch.setattr(acceptance.sys, "platform", host)
    monkeypatch.setattr(acceptance.platform, "platform", lambda: "test-host")
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"candidate")

    def mutate(*args, **kwargs):
        archive.write_bytes(b"different candidate")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(acceptance.subprocess, "run", mutate)
    with pytest.raises(ValueError, match="Archive changed"):
        acceptance.validate(archive, tmp_path / "evidence")
    assert (
        json.loads((tmp_path / "evidence/acceptance.json").read_text(encoding="utf-8"))["status"]
        == "failed"
    )


def test_release_acceptance_rejects_missing_stapled_ticket(tmp_path, monkeypatch):
    acceptance = script("acceptance")
    monkeypatch.setattr(acceptance.sys, "platform", "darwin")
    monkeypatch.setattr(acceptance.platform, "platform", lambda: "test-host")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if "stapler" in command else 0)

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"candidate")
    with pytest.raises(RuntimeError, match="stapled-ticket exited 1"):
        acceptance.validate(archive, tmp_path / "evidence", require_notarized=True)
    report = json.loads((tmp_path / "evidence/acceptance.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["notarization"] != "passed"
    assert not any("/usr/sbin/spctl" in command for command in commands)


@pytest.mark.parametrize("host", ["darwin", "win32", "linux"])
def test_missing_archive_is_recorded_without_claiming_validation(tmp_path, monkeypatch, host):
    acceptance = script("acceptance")
    monkeypatch.setattr(acceptance.sys, "platform", host)
    monkeypatch.setattr(acceptance.platform, "platform", lambda: "test-host")
    with pytest.raises(FileNotFoundError):
        acceptance.validate(tmp_path / "missing.zip", tmp_path / "evidence")
    report = json.loads((tmp_path / "evidence/acceptance.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["archive_sha256"] is None
    assert report["checks"] == [] and "FileNotFoundError" in report["error"]


def test_windows_rejects_mac_signing_checks_before_running_commands(tmp_path, monkeypatch):
    acceptance = script("acceptance")
    monkeypatch.setattr(acceptance.sys, "platform", "win32")
    with pytest.raises(ValueError, match="only to macOS"):
        acceptance.validate(tmp_path / "candidate.zip", tmp_path / "out", require_notarized=True)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("unsafe", [False, True])
def test_windows_model_acceptance_extracts_the_windows_layout(tmp_path, monkeypatch, unsafe):
    import zipfile

    acceptance = script("acceptance")
    monkeypatch.setattr(acceptance.sys, "platform", "win32")
    monkeypatch.setattr(acceptance.platform, "platform", lambda: "test-host")
    archive = tmp_path / "candidate.zip"
    output = tmp_path / "out"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(
            "../escape" if unsafe else "RealOpenLens/RealOpenLens.exe", b"fake executable"
        )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if Path(command[1]).name == "smoke.py":
            executable = Path(command[2])
            assert executable.name == "RealOpenLens.exe" and executable.is_file()
            assert executable.parent.name == "RealOpenLens"
            precision = command[-1]
            kwargs["stdout"].write(json.dumps({"siglip_precision": precision}) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    if unsafe:
        with pytest.raises(ValueError, match="escapes"):
            acceptance.validate(archive, output, models=tmp_path / "models")
        assert len(commands) == 1
    else:
        report = acceptance.validate(archive, output, models=tmp_path / "models")
        assert report["inference"] == "passed" and set(report["results"]) == {"int8", "fp32"}
        assert len(commands) == 3
    assert not any("/usr/bin/ditto" in command for command in commands)


@pytest.mark.parametrize("failure", ["exit", "wrong_precision"])
def test_model_acceptance_never_passes_when_second_profile_fails(tmp_path, monkeypatch, failure):
    import zipfile

    acceptance = script("acceptance")
    monkeypatch.setattr(acceptance.sys, "platform", "win32")
    monkeypatch.setattr(acceptance.platform, "platform", lambda: "test-host")
    archive = tmp_path / "candidate.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("RealOpenLens/RealOpenLens.exe", b"fake executable")

    def run(command, **kwargs):
        if Path(command[1]).name == "smoke.py":
            precision = command[-1]
            if precision == "fp32" and failure == "exit":
                return SimpleNamespace(returncode=1)
            kwargs["stdout"].write(json.dumps({"siglip_precision": "int8"}) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    output = tmp_path / "evidence"
    with pytest.raises((RuntimeError, ValueError)):
        acceptance.validate(archive, output, models=tmp_path / "models")
    report = json.loads((output / "acceptance.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["inference"] != "passed"
    assert set(report["results"]) == {"int8"}
    assert report["error"]
