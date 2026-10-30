import ast
import enum
import inspect
import re
import sys
from pathlib import Path
from typing import Any, get_args, get_origin
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
import yaml

from retargetingkit.retargeting.retargeter import Retargeter
from retargetingkit.sources.source import Source
from retargetingkit.targets.target import Target


DOCUMENTS = (Path("README.md"), *sorted(Path("docs").rglob("*.md")))
PAGES = [
    page
    for group in yaml.safe_load(Path("docs/nav.yml").read_text())
    for run in [group.get("pages", []), *(fold["pages"] for fold in group.get("folds", []))]
    for page in run
]
# Each page's Python examples as the website copies them: shown code plus `# docs: hide` setup and checks.
EXAMPLES = {
    document.as_posix(): examples
    for document in DOCUMENTS
    if (
        examples := [
            "\n".join(l for l in match[1].splitlines() if l.strip() not in ("# docs: hide", "# docs: endhide")).strip(
                "\n"
            )
            for match in re.finditer(r"(?ms)^```python\n(.*?)^```$", document.read_text())
        ]
    )
}
# Device pages and the overview's loop step use autospecced devices: calls must match the code, but no hardware opens.
DEVICE_PAGES = ("docs/reference/sources/", "docs/reference/targets/", "docs/concepts/pipeline.md")


def device(cls: type) -> Any:
    """An autospecced `cls` instance whose tuple-returning methods unpack."""
    instance = create_autospec(cls, instance=True)
    for name, function in inspect.getmembers(cls, inspect.isfunction):
        returns = inspect.signature(function).return_annotation
        if get_origin(returns) is tuple:
            getattr(instance, name).return_value = tuple(MagicMock() for _ in get_args(returns))
    return instance


def run_page(document: str, examples: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run one page's examples in order and in one namespace, as a reader would."""
    devices = document.startswith(DEVICE_PAGES)
    namespace: dict[str, Any] = {}
    if devices:
        namespace.update(
            source=device(Source), retargeter=device(Retargeter), target=device(Target), target_qpos=np.zeros(20)
        )
    for number, example in enumerate(examples, 1):
        tree = ast.parse(example)
        imports: list[ast.stmt] = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(ast.Module(imports, []), f"{document} example {number}", "exec"), namespace)
        for name, value in list(namespace.items()) if devices else []:
            if (
                inspect.isclass(value)
                and value.__module__.startswith("retargetingkit.")
                and not issubclass(value, enum.Enum)
            ):
                if not name.endswith("Config"):
                    namespace[name] = create_autospec(value, return_value=device(value))
                elif "build" in vars(value):
                    built = inspect.signature(value.build).return_annotation
                    built = vars(sys.modules[value.__module__])[built] if isinstance(built, str) else built
                    monkeypatch.setattr(value, "build", lambda self, built=built: device(built))
        rest = [node for node in tree.body if node not in imports]
        exec(compile(ast.Module(rest, []), f"{document} example {number}", "exec"), namespace)


class TestDocumentation:
    def test_nav_lists_every_page_once(self) -> None:
        assert sorted(page["file"] for page in PAGES) == sorted(
            path.relative_to("docs").as_posix() for path in Path("docs").rglob("*.md")
        )
        assert len({page["id"] for page in PAGES}) == len(PAGES)

    @pytest.mark.parametrize("document", EXAMPLES)
    def test_python_examples_run(self, document: str, monkeypatch: pytest.MonkeyPatch) -> None:
        run_page(document, EXAMPLES[document], monkeypatch)

    @pytest.mark.parametrize("document", DOCUMENTS)
    def test_api_references_exist(self, document: Path) -> None:
        for name in re.findall(r"^::: ([\w.]+)$", document.read_text(), re.M):
            module, _, attr = name.rpartition(".")
            path = Path("src", *module.split("."))
            source = path.with_suffix(".py") if path.with_suffix(".py").exists() else path / "__init__.py"
            assert re.search(rf"^(class|def) {attr}\b", source.read_text(), re.M), f"{document}: {name}"

    @pytest.mark.parametrize("document", DOCUMENTS)
    def test_local_links_exist(self, document: Path) -> None:
        for raw in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            link = raw.strip("<>").split("#", 1)[0]
            if link and "://" not in link and not link.startswith("mailto:"):
                assert (document.parent / link).resolve().exists(), f"{document}: {raw}"

    @pytest.mark.parametrize("document", DOCUMENTS)
    def test_referenced_repo_paths_exist(self, document: Path) -> None:
        for raw in re.findall(r"(?:scripts|examples|src)/[A-Za-z0-9_./*-]+", document.read_text()):
            path = raw.rstrip("./")
            if "*" not in path:
                assert Path(path).exists(), f"{document}: {path}"
