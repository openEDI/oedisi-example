"""Tests for format-specific model readers in localfeeder.server."""

from __future__ import annotations

import zipfile
from pathlib import Path

import localfeeder.server as server

DATA_DIR = Path("tests/data")


def _extract(zip_path: Path, dst: Path) -> Path:
    """Extract archive into dst and return dst."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)
    return dst


def test_read_cim_uses_xml_from_extracted_archive(tmp_path, monkeypatch):
    """CIM reader should discover an XML file and pass it to CIMReader."""
    extracted = _extract(DATA_DIR / "cim_ieee_13.zip", tmp_path / "cim")

    observed: dict[str, object] = {"path": None, "read_called": False}
    sentinel_system = object()

    class FakeCIMReader:
        def __init__(self, cim_file):
            observed["path"] = str(cim_file)

        def read(self):
            observed["read_called"] = True

        def get_system(self):
            return sentinel_system

    monkeypatch.setattr(server, "CIMReader", FakeCIMReader)

    result = server._read_cim(str(extracted))

    assert result is sentinel_system
    assert observed["read_called"] is True
    assert observed["path"] is not None
    assert str(observed["path"]).lower().endswith(".xml")


def test_read_cyme_requires_network_equipment_and_passes_optional_load(tmp_path, monkeypatch):
    """CYME reader should pass Network/Equipment and optional Load files."""
    extracted = _extract(DATA_DIR / "cyme_ieee_123.zip", tmp_path / "cyme")

    observed: dict[str, tuple[str, ...] | None] = {"args": None}
    sentinel_system = object()

    class FakeCymeReader:
        def __init__(self, *args):
            observed["args"] = tuple(str(arg) for arg in args)
            self.system = sentinel_system

    monkeypatch.setattr(server, "CymeReader", FakeCymeReader)

    result = server._read_cyme(str(extracted))

    assert result is sentinel_system
    assert observed["args"] is not None
    basenames = {Path(arg).name.lower() for arg in observed["args"]}
    assert "network.txt" in basenames
    assert "equipment.txt" in basenames
    assert "load.txt" in basenames


def test_read_json_asserts_file_and_passes_path_to_from_json(tmp_path, monkeypatch):
    """JSON reader should discover a JSON file and pass its path to from_json."""
    model_dir = tmp_path / "json_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "sample.json"
    model_file.write_text("{}", encoding="utf-8")

    observed: dict[str, str | None] = {"path": None}
    sentinel_system = object()

    def fake_from_json(json_path):
        observed["path"] = str(json_path)
        return sentinel_system

    monkeypatch.setattr(server.DistributionSystem, "from_json", staticmethod(fake_from_json))

    result = server._read_json(str(model_dir))

    assert result is sentinel_system
    assert observed["path"] == str(model_file)
