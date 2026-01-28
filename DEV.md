# Development

## Setup
- Install dev dependencies:
  - `python -m pip install -e ".[dev]"`

## Release (PyPI)
- Versioning uses git tags via `hatch-vcs`; use `vX.Y.Z` tags.
- Build artifacts:
  - `python -m build`
- Verify artifacts:
  - `python -m twine check dist/*`
- Upload to PyPI:
  - `python -m twine upload dist/*`
 
### Release script
- Full release (real PyPI): `scripts/release.sh`
- TestPyPI upload: `scripts/release.sh --test-pypi`

### Suggested release flow
1. Run checks: `ruff check .` and `pytest`
2. Tag the release: `git tag -a vX.Y.Z -m "vX.Y.Z"`
3. Push tag: `git push origin vX.Y.Z`
4. Build and upload as above

