"""Architecture checks that enforce the design's safeguards."""

import ast
from pathlib import Path

import yaml

from synthgen.reference_world.simulator import simulate

PKG = Path(__file__).resolve().parents[1] / "synthgen"
NETWORK_MODULES = {"requests", "httpx", "urllib3", "aiohttp", "openai", "anthropic", "ollama"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_only_gateway_uses_network_clients():
    offenders = []
    for path in PKG.rglob("*.py"):
        if "gateway" in path.parts:
            continue
        top = {name.split(".")[0] for name in _imports(path)}
        if top & NETWORK_MODULES:
            offenders.append(str(path.relative_to(PKG)))
    assert offenders == [], f"Only synthgen/gateway may use network clients: {offenders}"


def test_generators_never_import_evaluation():
    for path in (PKG / "generators").rglob("*.py"):
        assert not any(name.startswith("synthgen.evaluation") for name in _imports(path)), path


def test_reference_world_is_reproducible():
    config = yaml.safe_load((PKG.parent / "config" / "reference_world.yaml").read_text())
    config["n_cases"] = 500
    a, b = simulate(config), simulate(config)
    assert a.equals(b)
    assert a["case_number"].str.startswith("SYN").all()  # synthetic marker
