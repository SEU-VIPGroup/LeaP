"""Build a source ZIP from an explicit allowlist, excluding local artifacts."""
import argparse
import hashlib
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = [root / name for name in (
        "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "MANIFEST.in", ".gitignore",
        "setup.py", "requirements.txt", "constraints-robotwin.txt",
        "third_party/README.md", "third_party/robotwin.lock.json")]
    for name in ("leap", "configs", "scripts", "tests", "docs", "licenses"):
        files.extend(path for path in (root / name).rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"))
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked release source: {path}")
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(Path("LeaP_release") / path.relative_to(root)))
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"{args.output}: {len(files)} files, {args.output.stat().st_size} bytes, sha256={digest}")


if __name__ == "__main__":
    main()
