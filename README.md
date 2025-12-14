<p align="center">
  <h1 align="center">Virtual Cell</h1>
  <p align="center">
    <strong>A Python library for simulating gene expression dynamics</strong>
  </p>
  <p align="center">
    <a href="https://www.gnu.org/licenses/gpl-3.0"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License: GPL v3"></a>
    <a href="https://www.python.org/downloads/release/python-312/"><img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12"></a>
  </p>
</p>

---

## Overview

**VirtualCell** is a comprehensive Python library for simulating gene expression dynamics at both the molecular and spatial levels. It supports deterministic (ODE) and stochastic (Gillespie/SSA) simulations, with full 2D and 3D spatial modeling capabilities.

### Features

- **Deterministic Simulation** — ODE-based continuous dynamics using `gillespy2`
- **Stochastic Simulation** — Gillespie algorithm (SSA) with Tau-Leaping for discrete molecule counts
- **Spatial Modeling** — 2D and 3D particle tracking with diffusion and compartmentalization
- **Nuclear Transport** — Models RNA transport through the nuclear envelope
- **Drug Perturbations** — Simulate parameter changes at specified time points
- **Synthetic Microscopy** — Generate realistic 2D microscopy images from simulations
- **Particle Detection** — TrackPy integration for spot detection and analysis

### Biological Model

The simulation models a 3-stage gene expression process:

```text
Transcription          RNA Transport           Translation
    G_on → R_n    ────▶    R_n → R_c    ────▶    R_c → P
   (in nucleus)        (through NPC)          (in cytosol)
```

**Species:**

- `R_n` — Nuclear RNA (mRNA in nucleus)
- `R_c` — Cytoplasmic RNA (mRNA in cytosol)
- `P` — Protein

---

## Project Structure

```text
virtual_cell/
├── src/
│   └── imports.py              # Core simulation module
├── notebooks/
│   ├── constitutive.ipynb      # Constitutive gene expression examples
│   ├── simulated_data.ipynb    # Synthetic microscopy data generation
│   ├── fspExample.ipynb        # Finite State Projection examples
│   └── Figure_4_*.ipynb        # Publication figures
├── spatio_temporal_models/
│   └── simulated_cell_single.ipynb  # 2D/3D spatial simulation examples
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

> Requires [Anaconda](https://www.anaconda.com) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Quick Start

```bash
git clone https://github.com/luisub/virtual_cell.git
cd virtual_cell
conda env create -f vc.yml
conda activate vc
```

### Manual Setup

```bash
conda create -n vc python=3.12 -y
conda activate vc
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|:--------|:--------|
| `gillespy2` | Stochastic simulation (Gillespie algorithm) |
| `trackpy` | Particle detection and tracking |
| `scikit-image` | Image processing and segmentation |
| `numpy`, `scipy` | Numerical computing |
| `matplotlib`, `seaborn` | Visualization |
| `cellpose` | Deep learning-based cell segmentation |

---

## Usage

### Stochastic Simulation

```python
from src.imports import *

parameter_values = {
    'k_r': 1.0,      # Transcription rate
    'k_t': 0.5,      # Transport rate
    'k_p': 0.1,      # Translation rate
    'gamma_r': 0.1,  # RNA decay rate
    'gamma_p': 0.01  # Protein decay rate
}

initial_conditions = {'R_n': 0, 'R_c': 0, 'P': 0}

time, trajectories = simulate_model(
    parameter_values, 
    initial_conditions,
    total_simulation_time=100,
    simulation_type='discrete',
    number_of_trajectories=100
)

plotting_stochastic(time, trajectories, species_colors)
```

### Spatial Simulation

```python
params = {
    'simulation_type': '3D',
    'simulation_volume_size': [100, 100, 100],
    'nucleus_size': [30, 30, 30],
    'cytosol_size': [80, 80, 80],
    'position_TS': 'center',
    'k_r': 1.0,
    'k_on': 0.1,
    'k_off': 0.1,
    # ... other parameters
}

simulator = GeneExpressionSimulator(params)
results = simulator.run()

plot_particle_positions(
    results,
    simulation_volume_size=params['simulation_volume_size'],
    masks_nucleus=results['nucleus_mask'],
    masks_cytosol=results['cytosol_mask'],
    simulation_type='3D'
)
```

---

## Notebooks

| Notebook | Description |
|:---------|:------------|
| `constitutive.ipynb` | Basic constitutive gene expression models |
| `simulated_data.ipynb` | Generate synthetic microscopy images |
| `fspExample.ipynb` | Finite State Projection method examples |
| `simulated_cell_single.ipynb` | Full spatial simulation workflow |

---

## License

This project is licensed under the [GNU General Public License v3](LICENSE).

---

## Citation

If you use VirtualCell in your research, please cite:

> Luis U. Aguilera. *VirtualCell: A library to simulate gene expression.* GitHub, 2025.
> <https://github.com/luisub/virtual_cell>

```bibtex
@misc{Aguilera2025VirtualCell,
  author       = {Luis U. Aguilera},
  title        = {VirtualCell: A Python library for gene expression simulation},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/luisub/virtual_cell.git}},
  note         = {Licensed under the GPL v3 License},
  keywords     = {gene expression, stochastic simulation, spatial modeling}
}
```

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

---

## Author

**Luis U. Aguilera**

For questions or issues, please [open an issue](https://github.com/luisub/virtual_cell/issues).
