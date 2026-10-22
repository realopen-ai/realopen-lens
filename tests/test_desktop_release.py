"""Release gates reject incomplete signing and unsuccessful acceptance evidence."""

import importlib.util
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
