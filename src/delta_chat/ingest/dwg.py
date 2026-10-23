"""DWG/DXF adapter — DWG->DXF convert (if needed) + ezdxf parse -> canonical Document.

Plugs into the same ``FormatAdapter`` seam as the native/scanned PDF adapters: the delta
engine never learns the source format. Scope is intentionally coarse — DXF-exposed geometry only
(TEXT/MTEXT/INSERT attribs/DIMENSION/LINE), no full P&ID connectivity-graph delta.

Honesty is load-bearing here: when the input is a real ``.dwg`` and no ODA/``dwg2dxf`` converter is
present on the host, this returns a **surfaced** stub (``raw_meta["unsupported"] = True``, zero items,
the CLI command that would have run) rather than fabricating content. ``.dxf`` inputs always parse in
full regardless of converter availability.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import ezdxf
import ezdxf.units as ezdxf_units
from ezdxf import bbox as ezdxf_bbox

from ..canonical.model import BBox, Document, Item, normalize_bbox
from ..config import Config, load_config
from ..observability import tracing
from . import tagging as _tagging
from .base import register_adapter

_CONVERTER_CANDIDATES = ("dwg2dxf", "ODAFileConverter")


class DwgParseError(RuntimeError):
    """Raised when conversion or DXF parsing fails -- always with a clear, traceable reason, never
    swallowed. A future ``tracing.stage('ingest')`` wrapper around adapter calls records it via its
    normal except-branch (record_exception + set_status(ERROR)); this module doesn't duplicate that,
    it just makes sure what propagates is diagnosable."""


def _is_dwg_binary(data: bytes) -> bool:
    return data[:4] == b"AC10"


def _locate_converter() -> str | None:
    """Locate an installed DWG->DXF converter on PATH. A separate, named function so tests can
    monkeypatch it to exercise the no-converter honest-stub path deterministically."""
    for name in _CONVERTER_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


def _would_run_command() -> str:
    return "dwg2dxf <input>.dwg <output>.dxf"


def _convert_dwg_to_dxf(data: bytes, converter_path: str, cfg: Config) -> bytes:
    timeout_s = float(cfg.get("ingest.dwg_convert_timeout_s", 120))
    oda_version = cfg.get("ingest.dwg_oda_target_version", "ACAD2018")
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "input.dwg"
        out_path = Path(td) / "input.dxf"
        in_path.write_bytes(data)
        if Path(converter_path).name.lower().startswith("dwg2dxf"):
            cmd = [converter_path, str(in_path), str(out_path)]
        else:  # ODA File Converter: <inDir> <outDir> <outVer> <outType> <recurse> <audit> [filter]
            cmd = [converter_path, td, td, oda_version, "DXF", "0", "1", "input.dwg"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout_s)
        except subprocess.CalledProcessError as exc:
            raise DwgParseError(
                f"DWG->DXF converter failed (exit {exc.returncode}): {' '.join(cmd)}\n"
                f"stderr: {exc.stderr.decode('utf-8', errors='replace')}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise DwgParseError(f"DWG->DXF converter timed out after {timeout_s}s: {' '.join(cmd)}") from exc
        if not out_path.exists():
            raise DwgParseError(f"DWG->DXF converter reported success but produced no output: {' '.join(cmd)}")
        return out_path.read_bytes()


def _entity_bbox(entity, page_size: tuple[float, float], insunits: float | None) -> BBox | None:
    extents = ezdxf_bbox.extents([entity], fast=True)
    if not extents.has_data:
        return None
    x0, y0 = extents.extmin.x, extents.extmin.y
    x1, y1 = extents.extmax.x, extents.extmax.y
    return normalize_bbox((x0, y0, x1, y1), "dxf_wcs", page_size, insunits=insunits)


def _build_document(dxf_doc, pid: str, source_rev: str, cfg: Config) -> Document:
    msp = dxf_doc.modelspace()
    all_entities = list(msp)
    whole = ezdxf_bbox.extents(all_entities, fast=True) if all_entities else None
    if whole is not None and whole.has_data and whole.extmax.x > 0 and whole.extmax.y > 0:
        page_size = (float(whole.extmax.x), float(whole.extmax.y))
    else:
        page_size = (1.0, 1.0)  # degenerate/empty drawing -- avoid a divide-by-zero in normalize_bbox

    insunits_code = dxf_doc.header.get("$INSUNITS", 0)
    unit = ezdxf_units.decode(insunits_code) or "unitless"

    items: list[Item] = []
    seq = 0

    def add_item(text: str, itype: str, parsed: dict, entity, layer: str | None) -> None:
        nonlocal seq
        bbox = _entity_bbox(entity, page_size, insunits_code)
        if bbox is None:
            return
        seq += 1
        items.append(
            Item(
                id=f"0:drawing:{seq}",
                source_rev=source_rev,  # type: ignore[arg-type]
                type=itype,  # type: ignore[arg-type]
                text=text,
                parsed_fields=parsed,
                bbox=bbox,
                page=0,
                sheet=None,
                region="drawing",
                rotation_deg=0.0,
                layer=layer,
                conf=1.0,
                source_engine="ezdxf",
            )
        )

    for entity in msp.query("TEXT MTEXT INSERT DIMENSION LINE"):
        dxftype = entity.dxftype()
        layer = entity.dxf.layer

        if dxftype == "TEXT":
            text = entity.dxf.text
            hits = _tagging.extract_tags(text, cfg)
            itype, parsed = (hits[0].type, hits[0].parsed_fields) if hits else ("text", {})
            add_item(text, itype, parsed, entity, layer)

        elif dxftype == "MTEXT":
            text = entity.plain_text()
            hits = _tagging.extract_tags(text, cfg)
            itype, parsed = (hits[0].type, hits[0].parsed_fields) if hits else ("text", {})
            add_item(text, itype, parsed, entity, layer)

        elif dxftype == "INSERT":
            block_name = entity.dxf.name
            add_item(block_name, "symbol", {"block": block_name}, entity, layer)
            for attrib in entity.attribs:
                text = attrib.dxf.text
                hits = _tagging.extract_tags(text, cfg)
                itype, parsed = (hits[0].type, hits[0].parsed_fields) if hits else ("text", {})
                add_item(text, itype, parsed, attrib, attrib.dxf.layer)

        elif dxftype == "DIMENSION":
            value = entity.get_measurement()
            add_item(str(value), "dimension", {"value": value, "unit": unit}, entity, layer)

        elif dxftype == "LINE":
            add_item("LINE", "symbol", {}, entity, layer)

    return Document(
        pid=pid,
        format="dwg",
        page_count=1,
        revision_label=None,
        sheet_size=page_size,
        items=items,
        regions=[],
        raw_meta={"insunits": insunits_code, "unit": unit},
    )


def parse_dwg(data: bytes, pid: str, source_rev: str = "A", cfg: Config | None = None) -> Document:
    """DWG->DXF convert (if needed) + ezdxf parse -> canonical Document. Honest stub if the input is
    ``.dwg`` and no converter is available."""
    cfg = cfg or load_config()

    if _is_dwg_binary(data):
        converter = _locate_converter()
        if converter is None:
            would_run = _would_run_command()
            tracing.mark_degraded("dwg_converter_unavailable", would_run=would_run)
            return Document(
                pid=pid,
                format="dwg",
                page_count=0,
                revision_label=None,
                sheet_size=(0.0, 0.0),
                items=[],
                regions=[],
                raw_meta={"unsupported": True, "would_run": would_run},
            )
        dxf_bytes = _convert_dwg_to_dxf(data, converter, cfg)
        text = dxf_bytes.decode("utf-8", errors="replace")
    else:
        text = data.decode("utf-8", errors="replace")

    try:
        dxf_doc = ezdxf.read(io.StringIO(text))
    except ezdxf.DXFError as exc:
        raise DwgParseError(f"malformed DXF content for pid={pid!r}: {exc}") from exc
    return _build_document(dxf_doc, pid, source_rev, cfg)


class DwgAdapter:
    format = "dwg"

    def detect(self, data: bytes, filename: str | None) -> bool:
        if filename and filename.lower().endswith((".dwg", ".dxf")):
            return True
        return data[:4] == b"AC10" or b"SECTION" in data[:64]

    def parse(self, data: bytes, pid: str, source_rev: str = "A") -> Document:
        return parse_dwg(data, pid, source_rev=source_rev)


register_adapter(DwgAdapter())
