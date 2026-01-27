import shutil
import subprocess
from pathlib import Path

# --- Configuration ---
# You can change these variables to customize the build
DIST_DIR = Path("dist")
CLEAN_DIST = True  # Set to False if you want to keep old versions in dist/
# ---------------------


def build_wheel():
    """
    Builds a wheel for the current package.
    """
    if CLEAN_DIST and DIST_DIR.exists():
        print(f"Removing existing {DIST_DIR} directory...")
        shutil.rmtree(DIST_DIR)

    print("Running: python -m build --wheel")
    try:
        subprocess.run(["python", "-m", "build", "--wheel"], check=True)

        print("\nBuild successful! Wheels available in 'dist/':")
        wheels = list(DIST_DIR.glob("*.whl"))
        for wheel in wheels:
            print(f"  - {wheel.name}")

        if not wheels:
            print("  No wheel files found in dist/.")

    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(build_wheel())
