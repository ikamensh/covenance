#!/usr/bin/env python3
import argparse
import glob
import subprocess
import sys


def dist_files() -> list[str]:
    files = sorted(glob.glob("dist/*"))
    if not files:
        print("No dist files found in dist/. Run the build step first.")
        sys.exit(1)
    return files


def run(cmd: list[str], label: str) -> None:
    print(label)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run release checks, build artifacts, verify them, and upload to PyPI."
        )
    )
    parser.add_argument(
        "-t",
        "--test-pypi",
        action="store_true",
        help="Upload to TestPyPI instead of the real PyPI index.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()

    run(["python", "-m", "ruff", "check", "."], "Running ruff...")
    run(["python", "-m", "pytest"], "Running pytest...")
    run(["python", "-m", "build"], "Building artifacts...")
    run(
        ["python", "-m", "twine", "check", *dist_files()],
        "Checking artifacts...",
    )

    if args.test_pypi:
        run(
            ["python", "-m", "twine", "upload", "--repository", "testpypi", *dist_files()],
            "Uploading to TestPyPI...",
        )
        return 0

    if not args.yes:
        confirm = input(
            "Upload to PyPI (https://pypi.org)? Type 'yes' to continue: "
        ).strip()
        if confirm != "yes":
            print("Aborted.")
            return 1

    run(
        ["python", "-m", "twine", "upload", *dist_files()],
        "Uploading to PyPI...",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
