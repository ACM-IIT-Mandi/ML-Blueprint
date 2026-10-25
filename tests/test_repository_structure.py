"""Tests that enforce the repository's own conventions."""

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "mlblueprint"

FAMILIES = [
    "linear",
    "tree",
    "cluster",
    "decomposition",
    "svm",
    "probabilistic",
    "optim",
    "neural",
]

# Cross-family imports that are deliberate. Everything else is a layering
# violation and blocks the family from ever graduating into its own library.
ALLOWED_CROSS_IMPORTS = {
    "tree": set(),
    "neural": {"optim"},
    "linear": set(),
    "cluster": set(),
    "decomposition": set(),
    "svm": set(),
    "probabilistic": set(),
    "optim": set(),
}

# Folders inside a family that are not algorithms.
NON_ALGORITHM_DIRS = {"tests", "docs", "__pycache__"}


def algorithm_folders() -> list[Path]:
    """Return every algorithm folder currently in the library."""
    folders = []
    for family in FAMILIES:
        family_path = PACKAGE_ROOT / family
        if not family_path.is_dir():
            continue
        folders.extend(
            child
            for child in family_path.iterdir()
            if child.is_dir()
            and child.name not in NON_ALGORITHM_DIRS
            and not child.name.startswith(".")
        )
    return folders


# Families


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_is_importable(family):
    module = importlib.import_module(f"mlblueprint.{family}")
    assert hasattr(module, "__all__"), f"{family} must define __all__"


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_has_a_readme(family):
    readme = PACKAGE_ROOT / family / "README.md"
    assert readme.is_file(), f"mlblueprint/{family}/README.md is missing"
    assert readme.read_text().strip(), f"mlblueprint/{family}/README.md is empty"


@pytest.mark.parametrize("family", FAMILIES)
def test_families_do_not_import_each_other(family):
    """A family may import core and third-party code, and nothing sideways.

    This is the rule that keeps extraction into a standalone library possible.
    See docs/ARCHITECTURE.md.
    """
    others = set(FAMILIES) - {family} - ALLOWED_CROSS_IMPORTS[family]
    violations = []

    for source in (PACKAGE_ROOT / family).rglob("*.py"):
        text = source.read_text()
        for other in others:
            if f"mlblueprint.{other}" in text:
                violations.append(f"{source.relative_to(REPO_ROOT)} imports {other}")

    assert not violations, (
        "Cross-family imports found:\n  "
        + "\n  ".join(violations)
        + "\n\nA family may import mlblueprint.core and third-party libraries only. "
        "If this dependency is deliberate, declare it in ALLOWED_CROSS_IMPORTS here "
        "and in the family README."
    )


# Algorithms


def test_every_algorithm_folder_follows_the_template():
    """Each algorithm folder needs at minimum a README, an __init__ and a test folder.

    The full contract is in docs/ALGORITHM_TEMPLATE.md; this checks the parts that
    can be checked mechanically.
    """
    problems = []

    for folder in algorithm_folders():
        relative = folder.relative_to(REPO_ROOT)
        if not (folder / "README.md").is_file():
            problems.append(f"{relative}: missing README.md")
        if not (folder / "__init__.py").is_file():
            problems.append(f"{relative}: missing __init__.py")
        if not (folder / "tests").is_dir():
            problems.append(f"{relative}: missing tests/")
        if not (folder / "docs").is_dir():
            problems.append(f"{relative}: missing docs/")

    assert not problems, (
        "Algorithm folders not following the template:\n  " + "\n  ".join(problems)
    )


def test_algorithm_folders_are_snake_case():
    bad = [
        f.relative_to(REPO_ROOT)
        for f in algorithm_folders()
        if f.name != f.name.lower().replace("-", "_") or "-" in f.name
    ]
    assert not bad, f"Algorithm folders must be snake_case: {bad}"


def test_algorithm_readmes_declare_a_maturity_level():
    """Every algorithm states how finished it is. See docs/ALGORITHM_TEMPLATE.md."""
    levels = ("`draft`", "`complete`", "`polished`")
    missing = []

    for folder in algorithm_folders():
        readme = folder / "README.md"
        if not readme.is_file():
            continue  # reported by the template test
        if not any(level in readme.read_text(encoding="utf-8") for level in levels):
            missing.append(str(folder.relative_to(REPO_ROOT)))

    assert not missing, (
        "These algorithms do not declare a maturity level in their README:\n  "
        + "\n  ".join(missing)
    )


# Repository


def test_no_notebooks_are_committed():
    """Notebooks produce unreviewable diffs. Examples are .py files.

    See CONTRIBUTING.md.
    """
    notebooks = [
        str(p.relative_to(REPO_ROOT))
        for p in REPO_ROOT.rglob("*.ipynb")
        if ".venv" not in p.parts and "node_modules" not in p.parts
    ]
    assert not notebooks, f"Jupyter notebooks must not be committed: {notebooks}"


@pytest.mark.parametrize(
    "document",
    [
        "README.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "AI_POLICY.md",
        "ROADMAP.md",
        "MAINTAINERS.md",
        "LICENSE",
        "docs/REPOSITORY_GUIDE.md",
        "docs/ARCHITECTURE.md",
        "docs/ALGORITHM_TEMPLATE.md",
        "docs/STYLE_GUIDE.md",
        "docs/TESTING.md",
        "docs/VISUALISATION.md",
        "docs/GETTING_STARTED.md",
        "docs/ALGORITHMS.md",
    ],
)
def test_required_documents_exist(document):
    """These are load-bearing, deleting one should fail loudly, not silently."""
    assert (REPO_ROOT / document).is_file(), f"{document} is missing"
