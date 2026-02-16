
## Step 9: LICENSE

Use the BSD-3-Clause license (copy from AstroPy's LICENSE file for consistency).

## Step 10: Build and Publish

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Test on TestPyPI first
twine upload --repository testpypi dist/*

# If all looks good, publish to PyPI
twine upload dist/*
