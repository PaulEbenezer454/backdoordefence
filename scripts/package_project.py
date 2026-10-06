"""Build a portable source-and-results archive of TCLAD-FL."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".venv", "venv", "__pycache__", ".pytest_cache", ".git", "data",
    "work", "outputs", "logs", "node_modules",
}


def package_project(output: Path) -> Path:
    """Archive tracked project content and results, excluding local runtime/data."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {output}")
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.resolve() == output:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append((relative, path))
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, path in sorted(files, key=lambda item: item[0].as_posix().lower()):
            archive.write(path, Path("TCLAD-FL") / relative)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "TCLAD-FL-project.zip")
    args = parser.parse_args()
    archive = package_project(args.output)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"Archive: {archive}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
