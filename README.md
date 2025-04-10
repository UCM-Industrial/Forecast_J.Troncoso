# TS Energy Patterns 🔋⏳

Advanced time series decomposition module to uncover hidden patterns in energy demand dynamics across multiple sources.

## Setup

### Prerequisites

- Python 3.12+
- [UV](https://github.com/astral-sh/uv) (recommended)
- [Ruff](https://github.com/astral-sh/ruff) (pre-installed in dev dependencies)

### Installation Options

#### Using UV (Recommended)

```bash
# Install uv (globally if not already installed)
pip install uv

# Create and activate the virtual environment
uv venv

# Install core dependencies
uv pip install --requirements uv.lock
```

#### Using Pip

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies from pyproject.toml
pip install -e .

# For development dependencies
pip install -e ".[dev]"
```

#### Using Conda

```bash
# Create a conda environment with Python 3.13
conda create -n project-name python=3.13 # or 3.12

# Activate the environment
conda activate project-name

# Install pip inside conda environment
conda install pip

# Install dependencies from pyproject.toml
pip install -e .

# For development dependencies
pip install -e ".[dev]"
```
