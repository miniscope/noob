"""
Drive towncrier across the packages in the monorepo.

Every package keeps its fragments and its rendered changelog together in
`changelog/<package>/`, and towncrier is pointed at one of them at a time with
`--dir`. `--config` has to be passed alongside `--dir` or towncrier searches
upwards for a config and uses *that* directory as its base instead, so every
invocation goes through here.
"""

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
CHANGELOG_DIR = REPO_ROOT / "changelog"
CONFIG = REPO_ROOT / "pyproject.toml"
PACKAGES_DIR = REPO_ROOT / "packages"
DOCS_DIR = REPO_ROOT / "docs"
# must match `start_string` in [tool.towncrier]
START_STRING = "<!-- towncrier release notes start -->"


def packages() -> list[str]:
    """Packages that have a changelog, i.e. every subdirectory of `changelog/`."""
    return sorted(p.name for p in CHANGELOG_DIR.iterdir() if p.is_dir())


def towncrier(command: str, package: str, *args: str, quiet: bool = False) -> int:
    """
    Run a towncrier subcommand against one package's changelog directory.

    `--config` has to be passed even though towncrier would find it on its own,
    because without it `--dir` is treated as a search path rather than the base
    directory, and every relative path in the config resolves against the repo
    root instead of the package.
    """
    target = ["--config", str(CONFIG), "--dir", f"changelog/{package}"]
    # towncrier inherits our streams, so flush first or our own output lands out of order
    sys.stdout.flush()
    return subprocess.run(
        [sys.executable, "-m", "towncrier", command, *target, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=quiet,
        text=quiet,
    ).returncode


def _today() -> str:
    """towncrier's default date is ISO, the changelogs are YY-MM-DD."""
    return date.today().strftime("%y-%m-%d")


def _require(package: str) -> Path:
    directory = CHANGELOG_DIR / package
    if not directory.is_dir():
        sys.exit(
            f"{package} has no changelog directory.\n"
            f"Start one with `pdm run changelog init {package}`, "
            f"or pick one of: {', '.join(packages())}"
        )
    return directory


def build(package: str, version: str) -> int:
    """Consume a package's fragments into its changelog."""
    _require(package)
    version = version.removeprefix("v")
    return towncrier("build", package, "--version", version, "--date", _today(), "--yes")


def draft(package: str | None) -> int:
    """Render pending fragments to stdout without touching anything."""
    for pkg in [package] if package else packages():
        if package:
            _require(pkg)
        elif not _fragments(pkg):
            continue
        print(f"\n{'=' * 70}\n{pkg}\n{'=' * 70}")
        towncrier("build", pkg, "--version", "Upcoming", "--date", _today(), "--draft")
    return 0


def _fragments(package: str) -> list[Path]:
    directory = CHANGELOG_DIR / package
    return [p for p in directory.glob("*.md") if p.name != "CHANGELOG.md"]


def changed_packages(compare_with: str) -> list[str]:
    """
    Packages whose source a branch touches, and which therefore owe an entry.

    Changes outside `packages/` - docs, CI, the JS frontend - belong to no
    package and so require nothing.
    """
    diff = subprocess.run(
        ["git", "diff", "--name-only", f"{compare_with}..."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    known = set(packages())
    touched = set()
    for path in diff:
        parts = Path(path).parts
        if len(parts) > 2 and parts[0] == "packages" and parts[1] in known:
            touched.add(parts[1])
    return sorted(touched)


def check(compare_with: str) -> int:
    """Require a fragment for every package the branch touches."""
    touched = changed_packages(compare_with)
    if not touched:
        print(f"No package source changed against {compare_with}, no fragment needed.")
        return 0

    print(f"Packages changed against {compare_with}: {', '.join(touched)}\n")
    failed = []
    for pkg in touched:
        # quietly - towncrier's check prints every file changed on the branch,
        # once per package, which buries the result
        found = not towncrier("check", pkg, "--compare-with", compare_with, quiet=True)
        print(f"  {'ok     ' if found else 'MISSING'}  changelog/{pkg}/")
        if not found:
            failed.append(pkg)

    if failed:
        sys.stdout.flush()
        print(
            "\nNo changelog entry for: " + ", ".join(failed) + "\n"
            "Add one file per change, named `{pr number}.{type}.md`:\n"
            + "\n".join(f"  changelog/{pkg}/1234.fixed.md" for pkg in failed)
            + "\nSee changelog/README.md, or label the PR `no changelog` to skip.",
            file=sys.stderr,
        )
        return 1
    return 0


def init(package: str) -> int:
    """Start a changelog for a package that doesn't have one yet."""
    if not (PACKAGES_DIR / package).is_dir():
        print(f"warning: no packages/{package} directory", file=sys.stderr)

    changelog = CHANGELOG_DIR / package / "CHANGELOG.md"
    if changelog.exists():
        print(f"{changelog.relative_to(REPO_ROOT)} already exists")
    else:
        changelog.parent.mkdir(parents=True, exist_ok=True)
        changelog.write_text(f"# {package}\n\n{START_STRING}\n")
        print(f"created {changelog.relative_to(REPO_ROOT)}")

    stub = DOCS_DIR / "changelog" / f"{package}.md"
    if not stub.exists():
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text(
            f"# {package}\n\n"
            f"```{{include}} ../../changelog/{package}/CHANGELOG.md\n"
            ":parser: myst\n"
            f":start-after: {START_STRING}\n"
            "```\n"
        )
        print(f"created {stub.relative_to(REPO_ROOT)}")

    print(f"\nNow add `changelog/{package}` to the toctree in docs/changelog.md")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="consume a package's fragments")
    p_build.add_argument("package", choices=packages())
    p_build.add_argument("version", help="release version, e.g. 1003.0.0")

    p_draft = sub.add_parser("draft", help="preview pending fragments")
    p_draft.add_argument("package", nargs="?", choices=packages())

    p_check = sub.add_parser("check", help="require an entry per changed package")
    p_check.add_argument("--compare-with", default="origin/main")

    p_init = sub.add_parser("init", help="start a changelog for a new package")
    p_init.add_argument("package")

    args = parser.parse_args()
    if args.command == "build":
        return build(args.package, args.version)
    if args.command == "draft":
        return draft(args.package)
    if args.command == "check":
        return check(args.compare_with)
    return init(args.package)


if __name__ == "__main__":
    sys.exit(main())
