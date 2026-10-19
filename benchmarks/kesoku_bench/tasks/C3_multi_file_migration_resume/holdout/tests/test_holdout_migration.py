"""Held-out F2P verification suite for C3_multi_file_migration_resume."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

from cfgkit.env import from_env
from cfgkit.export import dump, dump_dict, dump_key_value
from cfgkit.legacy import LEGACY_KEY_ORDER, OldConfig
from cfgkit.loader import load_config, load_json, validate_config
from cfgkit.merge import merge_chain, merge_configs
from cfgkit.v2 import ConfigV2

EXPECTED_ENV_SHA256 = "b33369f7b71dd178256e8162ae738e899ebbc641528db5070a7eb0041e23011c"
REPO_ROOT = Path(__file__).resolve().parents[1]
CFGKIT_DIR = REPO_ROOT / "cfgkit"


def _module_imported_names(py_path: Path) -> set[str]:
    """Return all symbol and module names imported or referenced in ``py_path``."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


def test_migrated_modules_import_v2_and_not_old_config() -> None:
    """loader.py, merge.py, and export.py must use ConfigV2 and never reference OldConfig."""
    for mod_name in ("loader.py", "merge.py", "export.py"):
        names = _module_imported_names(CFGKIT_DIR / mod_name)
        assert "ConfigV2" in names, f"{mod_name} does not import/use ConfigV2"
        assert "OldConfig" not in names, f"{mod_name} still imports or references OldConfig"


def test_v2_behavioral_loader_and_merge() -> None:
    """Loader and merge functions must return ConfigV2 instances with expected fields."""
    cfg = load_config({"service": "orders", "port": 9100, "tags": ["a"]}, strict=True)
    assert isinstance(cfg, ConfigV2)
    assert validate_config(cfg) == []

    from_js = load_json('{"service": "orders", "port": 9200, "debug": true}')
    assert isinstance(from_js, ConfigV2)
    assert from_js.port == 9200
    assert from_js.debug is True

    merged = merge_configs(cfg, {"port": 9300, "tags": ["b"], "limits": {"qps": 50}})
    assert isinstance(merged, ConfigV2)
    assert merged.port == 9300
    assert merged.tags == ["a", "b"]
    assert merged.limits == {"qps": 50}

    chained = merge_chain([cfg, {"version": "2.0"}, {"debug": True}])
    assert isinstance(chained, ConfigV2)
    assert chained.version == "2.0"
    assert chained.debug is True


def test_export_v2_key_order_matches_legacy() -> None:
    """export.dump() and dump_dict() on ConfigV2 must preserve OldConfig.to_dict() key order."""
    export_names = _module_imported_names(CFGKIT_DIR / "export.py")
    assert "ConfigV2" in export_names and "OldConfig" not in export_names

    v2_cfg = ConfigV2(
        service="SearchAPI",
        version="3.1",
        host="10.20.30.40",
        port=8088,
        debug=True,
        timeout_ms=1200,
        tags=["canary", "eu"],
        limits={"max_conn": 256},
    )
    # Sanity check: ConfigV2.to_dict() itself is alphabetical
    assert tuple(v2_cfg.to_dict().keys())[0] == "debug"

    exported_dict = dump_dict(v2_cfg)
    assert tuple(exported_dict.keys()) == LEGACY_KEY_ORDER

    exported_json = dump(v2_cfg)
    parsed_pairs = json.loads(exported_json, object_pairs_hook=list)
    json_keys = tuple(k for k, _ in parsed_pairs)
    assert json_keys == LEGACY_KEY_ORDER

    kv_lines = dump_key_value(v2_cfg).splitlines()
    kv_keys = tuple(line.split("=", 1)[0] for line in kv_lines)
    assert kv_keys == LEGACY_KEY_ORDER


def test_selective_migration_env_untouched_and_legacy_kept() -> None:
    """env.py must remain byte-identical (OldConfig) and legacy.py must still exist."""
    for mod_name in ("loader.py", "merge.py", "export.py"):
        names = _module_imported_names(CFGKIT_DIR / mod_name)
        assert "ConfigV2" in names and "OldConfig" not in names

    legacy_path = CFGKIT_DIR / "legacy.py"
    assert legacy_path.is_file(), "legacy.py was deleted despite env.py still needing it"

    env_path = CFGKIT_DIR / "env.py"
    assert env_path.is_file(), "env.py is missing"
    actual_sha = hashlib.sha256(env_path.read_bytes()).hexdigest()
    assert actual_sha == EXPECTED_ENV_SHA256, (
        "env.py was modified despite the Turn 3 plan change to skip env.py"
    )

    env_names = _module_imported_names(env_path)
    assert "OldConfig" in env_names
    env_cfg = from_env({"CFGKIT_SERVICE": "legacy-env", "CFGKIT_PORT": "7070"})
    assert isinstance(env_cfg, OldConfig)


def _find_checklist_text() -> str | None:
    """Locate migration_checklist.md in staging dir or fallback gold path."""
    ws_dir = REPO_ROOT.parent
    candidates: list[Path] = []
    sessions_dir = ws_dir / "sessions"
    if sessions_dir.is_dir():
        candidates.extend(sorted(sessions_dir.glob("**/migration_checklist.md")))
        candidates.extend(sorted(sessions_dir.glob("**/*checklist*.md")))
    candidates.extend(sorted(ws_dir.glob("*checklist*.md")))
    local_fallback = REPO_ROOT / ".migration_checklist.md"
    if local_fallback.is_file():
        candidates.append(local_fallback)
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def test_migration_checklist_completed_with_env_skipped() -> None:
    """Checklist must have loader, merge, export ticked and env marked skipped."""
    text = _find_checklist_text()
    assert text is not None, "migration_checklist.md not found in staging directory"

    lines = text.splitlines()
    ticked_re = re.compile(
        r"(\[[xX✓✔]\]|\b(done|completed|migrated|complete|ticked)\b|✓|✔|✅)",
        re.IGNORECASE,
    )
    skipped_re = re.compile(r"(skip|n/?a|excluded|deferred)", re.IGNORECASE)

    for mod in ("loader", "merge", "export"):
        mod_lines = [ln for ln in lines if mod in ln.lower()]
        assert mod_lines, f"No line mentioning {mod} in migration_checklist.md"
        assert any(ticked_re.search(ln) for ln in mod_lines), (
            f"Module {mod} is not marked done/ticked in migration_checklist.md: {mod_lines}"
        )

    env_lines = [ln for ln in lines if "env" in ln.lower()]
    assert env_lines, "No line mentioning env.py in migration_checklist.md"
    assert any(skipped_re.search(ln) for ln in env_lines), (
        f"env.py is not marked skipped in migration_checklist.md: {env_lines}"
    )
