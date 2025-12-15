# Microscopy Image Simulator - Complete Action Plan

**Version:** 1.0  
**Date:** 2025-12-14  
**Author:** Senior Scientific Software Engineer  
**Target File:** `spatio_temporal_models/microscopy_simulation.py`

---

## Executive Summary

This document provides a detailed implementation plan for expanding the existing biological simulation (`minimal_simulation.py`) into a microscopy image simulator that generates synthetic fluorescence microscopy data as 5D NumPy arrays `[T, Z, Y, X, C]`.

---

## Phase 0: Analysis of Existing Simulation State Variables

### Summary of Available State from `minimal_simulation.py`

| Category | Variable | Description | Availability |
|----------|----------|-------------|--------------|
| **Transcription Site** | `TS_trajectory` | List of dicts with `position`, `state` (ON/OFF), `time` | ✅ Per frame |
| **Nascent RNAs** | `RNA_trajectories[id]` where `state='nascent'` | RNAs at TS, position fixed at TS | ✅ Countable per frame |
| **Mature RNAs** | `RNA_trajectories[id]` where `state='mature'` | Diffusing RNAs with `[x,y,z]` positions | ✅ Per frame |
| **Nascent Proteins** | `Protein_trajectories[id]` where `state='nascent'` | Attached to parent RNA, has `parent_rna_id` | ✅ Per frame |
| **Mature Proteins** | `Protein_trajectories[id]` where `state='mature'` | Freely diffusing proteins | ✅ Countable per frame |
| **Geometry** | `nucleus_mask`, `cytosol_mask` | 3D boolean arrays | ✅ Static |
| **TS Position** | `transcription_site` | Fixed `[x, y, z]` np.array | ✅ Static |

### Key Insights for Channel Mapping

| Channel | C Index | Data Source | Rendering Type |
|---------|---------|-------------|----------------|
| TS Signal | 0 | Count of nascent RNAs at TS position | Gaussian PSF (dynamic amplitude + size) |
| Mature RNA | 1 | Positions of mature RNAs | Diffraction-limited Gaussian spots |
| Nascent Protein | 2 | Positions of translating RNAs with nascent proteins | Gaussian spots (amplitude ∝ nascent count) |
| Mature Protein | 3 | Total count of mature proteins | Diffuse cytosolic haze |

---

## Phase 1: Coordinate System & Conventions

### 1.1 Coordinate Definitions

```
Simulation Coordinates (from minimal_simulation.py):
- x, y, z: Continuous floating-point pixel coordinates
- Origin: Center of simulation volume (approximately)
- Range: [0, simulation_volume_size[i]] for each axis

Microscopy Image Coordinates:
- Shape: [T, Z, Y, X, C]
- Y, X: Image pixels (0-511 for 512×512)
- Z: Slice index (0-11 for 12 slices)
- Ordering: NumPy (row-major), where Y is rows, X is columns
```

### 1.2 Coordinate Mapping Functions

```python
# Mapping simulation -> image coordinates
def sim_to_image_xy(sim_x, sim_y, sim_vol_size, img_size=(512, 512)):
    """Map simulation XY to image YX (note axis swap for row/col)."""
    # Scale from simulation volume to image pixels
    scale_x = img_size[1] / sim_vol_size[0]  # X -> X pixels
    scale_y = img_size[0] / sim_vol_size[1]  # Y -> Y pixels
    return sim_y * scale_y, sim_x * scale_x  # Return (row, col) = (Y, X)

def map_z_to_slice_index(z_continuous, z_range, num_slices=12):
    """Map continuous z to nearest slice index."""
    # z_range: (z_min, z_max) of simulation volume
    # Returns: int in [0, num_slices-1]
    normalized_z = (z_continuous - z_range[0]) / (z_range[1] - z_range[0])
    slice_idx = int(round(normalized_z * (num_slices - 1)))
    return max(0, min(num_slices - 1, slice_idx))
```

### 1.3 Physical Units (Configurable)

| Parameter | Default Value | Unit | Notes |
|-----------|---------------|------|-------|
| `pixel_size_xy` | 0.1 | µm/pixel | Typical for high-NA microscopy |
| `z_step` | 0.5 | µm/slice | Between z-slices |
| `sigma_xy_psf` | 1.5 | pixels | ~150 nm lateral PSF |
| `sigma_z_psf` | 3.0 | pixels | ~300 nm axial PSF |

---

## Phase 2: Implementation Architecture

### 2.1 Module Structure

```
microscopy_simulation.py
├── CONSTANTS & DEFAULT PARAMETERS
├── MicroscopyParameters dataclass
├── Helper Functions:
│   ├── make_cell_and_nucleus_masks()
│   ├── create_baseline_intensity_map()
│   ├── render_gaussian_spot_2d()
│   ├── render_gaussian_spot_3d()
│   ├── map_z_to_slice_index()
│   ├── map_z_to_slice_weights()  # Optional: linear interpolation
│   ├── add_noise()
│   ├── blur_channel()
│   └── to_uint16()
├── Channel Rendering Functions:
│   ├── render_ts_channel()       # C=0
│   ├── render_mature_rna_channel()  # C=1
│   ├── render_nascent_protein_channel()  # C=2
│   └── render_mature_protein_channel()  # C=3
├── Main Function:
│   └── simulate_microscopy_stack()
└── if __name__ == "__main__":
    └── Example/test usage
```

### 2.2 Dependencies

```python
import numpy as np
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import tifffile  # Optional, for saving

# Import the biology simulator
from minimal_simulation import GeneExpressionSimulator, SimulationParameters
```

---

## Phase 3: Detailed Implementation Steps

### Step 3.1: Cell Masks & Baseline Intensities

**Function:** `make_cell_and_nucleus_masks()`

```python
def make_cell_and_nucleus_masks(
    img_shape: Tuple[int, int, int] = (12, 512, 512),  # Z, Y, X
    cytosol_radii: Tuple[float, float, float] = (175, 175, 40),  # half-axes
    nucleus_radii: Tuple[float, float, float] = (60, 50, 30),
    nucleus_offset: Tuple[float, float] = (0, 0),
    center: Optional[Tuple[float, float, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3D boolean masks for cytosol and nucleus.
    
    Returns:
        cytosol_mask: Shape (Z, Y, X), True inside cytosol (excluding nucleus)
        nucleus_mask: Shape (Z, Y, X), True inside nucleus
    """
    pass
```

**Function:** `create_baseline_intensity_map()`

```python
def create_baseline_intensity_map(
    cytosol_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    intensity_outside: float = 100.0,      # Camera offset/dark current
    intensity_cytosol: float = 200.0,      # Autofluorescence
    intensity_nucleus: float = 150.0       # Different from cytosol
) -> np.ndarray:
    """
    Create baseline intensity for each channel.
    
    Returns:
        baseline: Shape (Z, Y, X) float32 array
    """
    pass
```

### Step 3.2: PSF Rendering Primitives

**Function:** `render_gaussian_spot_3d()`

```python
def render_gaussian_spot_3d(
    image: np.ndarray,          # Shape (Z, Y, X), modified in-place
    position: Tuple[float, float, float],  # (z, y, x) in image coordinates
    amplitude: float,
    sigma_xy: float,
    sigma_z: float,
    truncate: float = 4.0       # Truncate at N sigmas
) -> np.ndarray:
    """
    Render a 3D Gaussian PSF onto the image.
    
    Implementation notes:
    - Create local patch around position
    - Compute Gaussian values analytically
    - Add to image (additive blending)
    """
    pass
```

**Function:** `render_gaussian_spot_2d()`

```python
def render_gaussian_spot_2d(
    image: np.ndarray,          # Shape (Y, X), modified in-place
    position: Tuple[float, float],  # (y, x) in image coordinates
    amplitude: float,
    sigma: float,
    truncate: float = 4.0
) -> np.ndarray:
    """
    Render a 2D Gaussian PSF onto a single z-slice.
    """
    pass
```

### Step 3.3: Z-Slice Mapping

**Function:** `map_z_to_slice_index()`

```python
def map_z_to_slice_index(
    z_continuous: float,
    z_min: float,
    z_max: float,
    num_slices: int = 12
) -> int:
    """
    Map continuous z coordinate to nearest slice index.
    
    Args:
        z_continuous: Z position from simulation
        z_min, z_max: Bounds of simulation z-range
        num_slices: Number of z-slices in output
        
    Returns:
        Slice index in [0, num_slices-1]
    """
    pass
```

**Function (Optional):** `map_z_to_slice_weights()`

```python
def map_z_to_slice_weights(
    z_continuous: float,
    z_min: float,
    z_max: float,
    num_slices: int = 12
) -> List[Tuple[int, float]]:
    """
    Map z to weighted contributions to adjacent slices.
    
    Returns:
        List of (slice_index, weight) tuples for linear interpolation.
        E.g., [(5, 0.7), (6, 0.3)] means 70% on slice 5, 30% on slice 6.
    """
    pass
```

### Step 3.4: Channel-Specific Rendering

#### Channel 0: Transcription Site (TS)

```python
def render_ts_channel(
    image_3d: np.ndarray,       # Shape (Z, Y, X)
    ts_position: Tuple[float, float, float],  # Simulation coordinates
    nascent_rna_count: int,
    params: MicroscopyParameters
) -> np.ndarray:
    """
    Render TS as Gaussian with amplitude AND size proportional to nascent RNAs.
    
    Amplitude model:
        A_TS = A_TS_per_RNA * N_nascent_RNA
        
    Size model (configurable):
        sigma_TS = sigma_base + sigma_per_RNA * f(N_nascent_RNA)
        where f(N) = sqrt(N) or log(1 + N)
    """
    pass
```

#### Channel 1: Mature RNA

```python
def render_mature_rna_channel(
    image_3d: np.ndarray,       # Shape (Z, Y, X)
    rna_positions: List[Tuple[float, float, float]],  # List of [x,y,z]
    params: MicroscopyParameters
) -> np.ndarray:
    """
    Render each mature RNA as a diffraction-limited Gaussian spot.
    
    Each spot has:
    - Fixed amplitude (A_mature_rna)
    - Fixed sigma (microscope PSF parameters)
    """
    pass
```

#### Channel 2: Nascent Proteins

```python
def render_nascent_protein_channel(
    image_3d: np.ndarray,       # Shape (Z, Y, X)
    translating_rnas: List[Dict],  # RNA positions with nascent protein counts
    params: MicroscopyParameters
) -> np.ndarray:
    """
    Render nascent protein signal at each translating RNA position.
    
    For each RNA with nascent proteins:
    - Position: Same as parent RNA
    - Amplitude: A_per_nascent_prot * N_nascent_on_this_RNA
    """
    pass
```

#### Channel 3: Mature Proteins

```python
def render_mature_protein_channel(
    image_3d: np.ndarray,       # Shape (Z, Y, X)
    mature_protein_count: int,
    cytosol_mask: np.ndarray,
    params: MicroscopyParameters
) -> np.ndarray:
    """
    Render mature proteins as diffuse cytosolic haze.
    
    Implementation:
    1. Create uniform intensity within cytosol_mask
    2. Scale intensity by protein count:
       I_cytosol += protein_scale * mature_protein_count
    3. Apply Gaussian blur to simulate diffusion/out-of-focus blur
    """
    pass
```

### Step 3.5: Noise Model

**Function:** `add_noise()`

```python
def add_noise(
    image: np.ndarray,          # Shape (Z, Y, X) or (T, Z, Y, X, C)
    cytosol_mask: np.ndarray,   # For cellular noise
    nucleus_mask: np.ndarray,
    read_noise_std: float = 5.0,
    shot_noise: bool = True,
    cellular_noise_amplitude: float = 10.0,
    cellular_noise_sigma: float = 5.0,  # Spatial correlation
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add realistic microscopy noise to an image.
    
    Noise model:
        output = signal + read_noise + shot_noise + cellular_texture
        
    Where:
    - read_noise: N(0, read_noise_std) - Gaussian camera/read noise
    - shot_noise: Poisson(signal) - Photon shot noise (optional)
    - cellular_texture: Spatially correlated noise within cell masks
    """
    pass
```

### Step 3.6: Post-Processing

**Function:** `blur_channel()`

```python
def blur_channel(
    image: np.ndarray,
    sigma: float,
    mask: Optional[np.ndarray] = None  # Only blur within mask
) -> np.ndarray:
    """Apply Gaussian blur to simulate diffusion/out-of-focus effects."""
    pass
```

**Function:** `to_uint16()`

```python
def to_uint16(
    image: np.ndarray,
    clip_percentile: float = 99.9,
    scale_to: int = 65535
) -> np.ndarray:
    """Convert float32 image to uint16 with proper scaling."""
    pass
```

---

## Phase 4: Main Simulation Function

```python
def simulate_microscopy_stack(
    sim_params: Optional[SimulationParameters] = None,
    micro_params: Optional[MicroscopyParameters] = None,
    simulation_results: Optional[Dict] = None,  # Pre-computed results
    save_path: Optional[str] = None,
    return_uint16: bool = False,
    verbose: bool = True
) -> np.ndarray:
    """
    Generate synthetic microscopy data from biology simulation.
    
    Args:
        sim_params: SimulationParameters for the biology simulation.
                   If None, uses defaults.
        micro_params: MicroscopyParameters for image rendering.
                     If None, uses defaults.
        simulation_results: Optional pre-run simulation results.
                           If None, runs the simulation internally.
        save_path: Optional path to save output TIFF.
        return_uint16: If True, return uint16 instead of float32.
        verbose: Print progress information.
        
    Returns:
        img5d: np.ndarray of shape (T, 12, 512, 512, 4)
               dtype: float32 or uint16
               
    Output Array Layout:
        img5d[t, z, y, x, c] where:
        - t: time frame index [0, T-1]
        - z: z-slice index [0, 11]
        - y: row/Y pixel [0, 511]
        - x: column/X pixel [0, 511]
        - c: channel [0=TS, 1=mature_RNA, 2=nascent_prot, 3=mature_prot]
    """
    
    # ALGORITHM:
    # 1. Run or use biology simulation
    # 2. Create geometry masks (2D projected to each Z slice or full 3D)
    # 3. For each time frame t:
    #    a. Extract state at time t from trajectories
    #    b. Initialize 4-channel 3D volume with baselines
    #    c. Channel 0: Render TS (count nascent RNAs)
    #    d. Channel 1: Render all mature RNA spots
    #    e. Channel 2: Render nascent protein at translating RNAs
    #    f. Channel 3: Render diffuse mature protein haze
    #    g. Add noise to all channels
    # 4. Stack all frames into [T, Z, Y, X, C]
    # 5. Optionally convert to uint16 and save
    
    pass
```

---

## Phase 5: Parameters Dataclass

```python
@dataclass
class MicroscopyParameters:
    """
    Parameters for microscopy image generation.
    
    Organized into logical groups:
    - Image dimensions
    - PSF parameters
    - Channel-specific amplitudes
    - Noise parameters
    - Baseline intensities
    """
    
    # Image Dimensions
    image_size_xy: Tuple[int, int] = (512, 512)  # Y, X
    num_z_slices: int = 12
    
    # PSF Parameters (in pixels)
    psf_sigma_xy: float = 1.5          # Lateral PSF width
    psf_sigma_z: float = 3.0           # Axial PSF width
    
    # Channel 0: Transcription Site
    ts_amplitude_per_nascent_rna: float = 500.0
    ts_sigma_base: float = 2.0
    ts_sigma_per_nascent_rna: float = 0.5
    ts_size_function: str = 'sqrt'     # 'sqrt' or 'log'
    
    # Channel 1: Mature RNA
    mature_rna_amplitude: float = 300.0
    
    # Channel 2: Nascent Protein
    nascent_protein_amplitude_per_chain: float = 200.0
    
    # Channel 3: Mature Protein
    mature_protein_scale: float = 0.5   # Intensity per protein
    mature_protein_blur_sigma: float = 10.0
    
    # Baseline Intensities
    intensity_outside_cell: float = 100.0
    intensity_cytosol: float = 200.0
    intensity_nucleus: float = 150.0
    
    # Noise Parameters
    read_noise_std: float = 5.0
    shot_noise_enabled: bool = True
    cellular_noise_amplitude: float = 10.0
    cellular_noise_sigma: float = 5.0
    
    # Random Seed
    random_seed: Optional[int] = None
    
    # Output Options
    output_dtype: str = 'float32'      # 'float32' or 'uint16'
    uint16_clip_percentile: float = 99.9
    
    def validate(self):
        """Validate all parameters."""
        errors = []
        if self.num_z_slices < 1:
            errors.append("num_z_slices must be >= 1")
        if self.psf_sigma_xy <= 0:
            errors.append("psf_sigma_xy must be > 0")
        # ... additional validation
        if errors:
            raise ValueError("Invalid microscopy parameters:\n" + "\n".join(errors))
```

---

## Phase 6: Testing & Validation

### 6.1 Sanity Checks (in `if __name__ == "__main__"`)

```python
if __name__ == "__main__":
    # 1. Generate short simulation
    print("Running biology simulation...")
    sim_params = SimulationParameters(
        total_time=50.0,
        frame_rate=1.0,
        burnin_time=100.0,
        random_seed=42
    )
    
    # 2. Generate microscopy stack
    print("Generating microscopy images...")
    micro_params = MicroscopyParameters(random_seed=42)
    
    img5d = simulate_microscopy_stack(
        sim_params=sim_params,
        micro_params=micro_params,
        save_path="test_microscopy.tiff",
        verbose=True
    )
    
    # 3. Validate output
    print(f"\n=== Output Validation ===")
    print(f"Shape: {img5d.shape}")
    print(f"Expected: (50, 12, 512, 512, 4)")
    print(f"Dtype: {img5d.dtype}")
    
    for c in range(4):
        channel_names = ['TS', 'Mature RNA', 'Nascent Prot', 'Mature Prot']
        ch_data = img5d[:, :, :, :, c]
        print(f"Channel {c} ({channel_names[c]}): "
              f"min={ch_data.min():.2f}, max={ch_data.max():.2f}, "
              f"mean={ch_data.mean():.2f}")
    
    # 4. Visual inspection (optional)
    # Save max projections or montages for quick visual check
```

### 6.2 Acceptance Criteria Checklist

| Criterion | Test Method | Pass Condition |
|-----------|-------------|----------------|
| Shape | `assert img5d.shape == (T, 12, 512, 512, 4)` | Exact match |
| Dtype | `assert img5d.dtype == np.float32` | or uint16 if requested |
| TS in C=0 only | Visual inspection of max projection | Bright spot at TS location |
| RNA spots in C=1 | Check spot count matches trajectory | ~N_mature_rnas spots visible |
| Nascent prot in C=2 | Spots at translation sites | Colocalizes with cytoplasmic RNAs |
| Diffuse haze in C=3 | No sharp spots in channel 3 | Smooth gradient inside cell |
| Cell boundary visible | Compare inside/outside intensity | Clear cell outline |
| Noise present | Check std deviation in flat regions | > 0 with seed reproducibility |
| Reproducibility | Run twice with same seed | Identical outputs |

---

## Phase 7: Implementation Timeline

| Step | Description | Estimated Time | Dependencies |
|------|-------------|----------------|--------------|
| 1 | Create file structure, imports, dataclass | 30 min | - |
| 2 | Implement `make_cell_and_nucleus_masks()` | 45 min | - |
| 3 | Implement `create_baseline_intensity_map()` | 20 min | Step 2 |
| 4 | Implement `render_gaussian_spot_2d/3d()` | 60 min | - |
| 5 | Implement `map_z_to_slice_index()` | 15 min | - |
| 6 | Implement `render_ts_channel()` (C=0) | 45 min | Steps 3, 4, 5 |
| 7 | Implement `render_mature_rna_channel()` (C=1) | 30 min | Steps 4, 5 |
| 8 | Implement `render_nascent_protein_channel()` (C=2) | 45 min | Steps 4, 5 |
| 9 | Implement `render_mature_protein_channel()` (C=3) | 45 min | Step 2 |
| 10 | Implement `add_noise()` | 45 min | - |
| 11 | Implement `blur_channel()`, `to_uint16()` | 20 min | - |
| 12 | Implement `simulate_microscopy_stack()` | 90 min | All above |
| 13 | Testing & validation | 60 min | Step 12 |

**Total Estimated Time:** ~8-10 hours

---

## Phase 8: Future Extensions (Out of Scope for v1.0)

1. **Chromatic aberration**: Slight offsets between channels
2. **Channel crosstalk**: Bleed-through between channels
3. **Bleaching**: Intensity decay over time  
4. **Drift**: Slow XY drift during acquisition
5. **Camera effects**: Hot pixels, vignetting
6. **Non-Gaussian PSFs**: Airy disk, astigmatism for z-encoding
7. **OME-TIFF metadata**: Full bioformats-compatible output
8. **GPU acceleration**: CuPy/JAX for faster rendering

---

## Phase 9: Ground Truth DataFrame Generation (MicroLive/Trackpy Compatible)

### 9.1 Purpose and Design Philosophy

The microscopy simulation serves as **ground truth** for testing and validating the [MicroLive](https://github.com/ningzhaoAnschutz/microlive) image analysis pipeline. To enable proper validation, we must generate DataFrames that:

1. **Match the trackpy/MicroLive format** exactly (column names, data types)
2. **Provide perfect ground truth** for spot positions at sub-pixel accuracy
3. **Include particle IDs** that persist across frames for trajectory validation
4. **Only include visible molecules** (nascent RNA visible as TS "bulb", mature RNA as spots, nascent proteins as spots)

### 9.2 Visible vs. Invisible Molecules

| Molecule Type | Visible? | Channel | DataFrame Output | Notes |
|--------------|----------|---------|------------------|-------|
| **Transcription Site (TS)** | ✅ Yes | C=0 | `df_ts` | Bulb intensity ∝ nascent RNA count |
| **Nascent RNA** | ❌ No (part of TS) | C=0 | N/A | Contributes to TS size/intensity, not individual spots |
| **Mature RNA** | ✅ Yes | C=1 | `df_mature_rna` | Individual diffraction-limited spots |
| **Nascent Protein** | ✅ Yes | C=2 | `df_nascent_protein` | Spots at translating RNA positions |
| **Mature Protein** | ❌ No (diffuse) | C=3 | N/A | Diffuse haze, not trackable |

### 9.3 MicroLive/ParticleTracking DataFrame Format

Based on the actual DataFrame structure from `temp_microscopy.py` (MicroLive library), the standard DataFrame format includes:

#### 9.3.1 Core Columns (Required for Tracking)

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `frame` | int64 | Frame number (0-indexed) | Time step index |
| `x` | float64 | X position in **image coordinates** (columns) | Converted from sim |
| `y` | float64 | Y position in **image coordinates** (rows) | Converted from sim |
| `z` | float64 | Z position (continuous or slice index) | From sim z-coordinate |
| `particle` | int64 | Unique particle ID persistent across frames | From simulation IDs |
| `image_id` | int64 | Image/dataset identifier | Default: 0 |
| `cell_id` | int64 | Cell identifier (from mask) | From simulation cell ID |
| `spot_id` | int64 | Spot identifier within cell | Incremental |
| `spot_type` | int64 | Channel/molecule type (0=TS, 1=RNA, 2=protein) | From channel |
| `cluster_size` | int64 | Number of molecules in cluster | 1 for single spots |

#### 9.3.2 Spatial Metadata Columns (MicroLive-compatible)

| Column | Type | Description | How to Compute |
|--------|------|-------------|----------------|
| `nuc_loc_y` | float64 | Nucleus centroid Y | From simulation nucleus center |
| `nuc_loc_x` | float64 | Nucleus centroid X | From simulation nucleus center |
| `cyto_loc_y` | float64 | Cytosol centroid Y | From simulation cell center |
| `cyto_loc_x` | float64 | Cytosol centroid X | From simulation cell center |
| `nuc_area_px` | float64 | Nucleus area in pixels | From simulation geometry |
| `cyto_area_px` | float64 | Cytosol area in pixels | From simulation geometry |
| `cell_area_px` | float64 | Total cell area in pixels | nuc + cyto |
| `is_nuc` | bool | Whether spot is in nucleus | From simulation location |
| `is_cluster` | bool | Whether this is a cluster | False for single spots |
| `is_cell_fragmented` | int64 | Cell fragmentation flag | 0 for simulated |

#### 9.3.3 Intensity Columns (per channel)

For each channel `c` in `range(num_channels)`:

| Column Pattern | Type | Description |
|----------------|------|-------------|
| `nuc_int_ch_{c}` | float64 | Mean nuclear intensity for channel c |
| `cyto_int_ch_{c}` | float64 | Mean cytosol intensity for channel c |
| `complete_cell_int_ch_{c}` | float64 | Mean total cell intensity for channel c |
| `spot_int_ch_{c}` | float64 | Spot intensity for channel c |
| `psf_amplitude_ch_{c}` | float64 | PSF amplitude (peak intensity) |
| `psf_sigma_ch_{c}` | float64 | PSF sigma (size) |
| `snr_ch_{c}` | float64 | Signal-to-noise ratio |
| `total_spot_int_ch_{c}` | float64 | Integrated spot intensity |

#### 9.3.4 Simulation-Specific Columns (Ground Truth)

| Column | Type | Description |
|--------|------|-------------|
| `time` | float64 | Actual simulation time (seconds) |
| `molecule_type` | str | 'TS', 'mature_RNA', or 'nascent_protein' |
| `channel` | int64 | Channel index (0, 1, or 2) |
| `sim_x` | float64 | Original simulation X coordinate |
| `sim_y` | float64 | Original simulation Y coordinate |
| `sim_z` | float64 | Original simulation Z coordinate |
| `in_cytosol` | bool | Whether particle is in cytosol |
| `nascent_count` | int64 | For TS: number of nascent RNAs |
| `parent_rna_id` | int64 | For nascent proteins: parent RNA ID |
| `ground_truth` | bool | Always True (to distinguish from detected spots) |

### 9.4 DataFrame Generation Functions

#### 9.4.1 Module Structure Update

Add these functions to the module structure:

```
microscopy_simulation.py
├── ... (existing structure) ...
├── Ground Truth DataFrame Functions:
│   ├── generate_ts_dataframe()           # TS positions + nascent count
│   ├── generate_mature_rna_dataframe()   # Mature RNA spot positions
│   ├── generate_nascent_protein_dataframe()  # Nascent protein spot positions
│   ├── generate_all_ground_truth()       # Combined function returning dict of DataFrames
│   └── save_ground_truth_csv()           # Save DataFrames to CSV files
└── ... (main function, etc.) ...
```

#### 9.4.2 Function Specifications

```python
def generate_ts_dataframe(
    simulation_results: Dict,
    ts_position: np.ndarray,
    micro_params: MicroscopyParameters,
    sim_vol_size: Tuple[int, int, int]
) -> pd.DataFrame:
    """
    Generate ground truth DataFrame for the Transcription Site.
    
    The TS is a single "spot" per frame whose intensity and size depend on 
    the number of nascent RNAs at that time.
    
    Args:
        simulation_results: Results dict from GeneExpressionSimulator.run()
        ts_position: Fixed TS position from simulation [x, y, z]
        micro_params: Microscopy parameters for coordinate conversion
        sim_vol_size: Simulation volume size [X, Y, Z]
        
    Returns:
        DataFrame with columns:
            frame, x, y, z, particle, mass, size, signal, time,
            molecule_type, channel, nascent_count, sim_x, sim_y, sim_z
    """
    pass


def generate_mature_rna_dataframe(
    simulation_results: Dict,
    micro_params: MicroscopyParameters,
    sim_vol_size: Tuple[int, int, int]
) -> pd.DataFrame:
    """
    Generate ground truth DataFrame for mature RNA spots.
    
    Each mature RNA is an individual trackable particle that moves via diffusion.
    
    Args:
        simulation_results: Results dict from GeneExpressionSimulator.run()
        micro_params: Microscopy parameters for coordinate conversion
        sim_vol_size: Simulation volume size [X, Y, Z]
        
    Returns:
        DataFrame with columns:
            frame, x, y, z, particle, mass, size, signal, time,
            molecule_type, channel, sim_x, sim_y, sim_z, in_cytosol
            
    Notes:
        - `particle` column uses the original RNA ID from simulation
        - Only includes RNAs with state='mature'
        - Tracks particles across frames (linking is already done by simulation)
    """
    pass


def generate_nascent_protein_dataframe(
    simulation_results: Dict,
    micro_params: MicroscopyParameters,
    sim_vol_size: Tuple[int, int, int]
) -> pd.DataFrame:
    """
    Generate ground truth DataFrame for nascent protein spots.
    
    Nascent proteins are attached to their parent RNA during translation.
    Their position matches the parent RNA position.
    
    Args:
        simulation_results: Results dict from GeneExpressionSimulator.run()
        micro_params: Microscopy parameters for coordinate conversion
        sim_vol_size: Simulation volume size [X, Y, Z]
        
    Returns:
        DataFrame with columns:
            frame, x, y, z, particle, mass, size, signal, time,
            molecule_type, channel, sim_x, sim_y, sim_z, parent_rna_id
            
    Notes:
        - `particle` column uses the original protein ID from simulation
        - Only includes proteins with state='nascent'
        - Multiple nascent proteins on the same RNA appear as intensity increase
    """
    pass


def generate_all_ground_truth(
    simulation_results: Dict,
    ts_position: np.ndarray,
    micro_params: MicroscopyParameters,
    sim_vol_size: Tuple[int, int, int]
) -> Dict[str, pd.DataFrame]:
    """
    Generate all ground truth DataFrames.
    
    Returns:
        Dictionary with keys:
            'ts': Transcription site DataFrame
            'mature_rna': Mature RNA DataFrame
            'nascent_protein': Nascent protein DataFrame
            'combined': All particles concatenated (for multi-channel analysis)
    """
    df_ts = generate_ts_dataframe(simulation_results, ts_position, 
                                   micro_params, sim_vol_size)
    df_rna = generate_mature_rna_dataframe(simulation_results, 
                                            micro_params, sim_vol_size)
    df_prot = generate_nascent_protein_dataframe(simulation_results, 
                                                   micro_params, sim_vol_size)
    
    # Create combined DataFrame with unique particle IDs
    # Offset particle IDs to avoid collisions between molecule types
    df_combined = pd.concat([df_ts, df_rna, df_prot], ignore_index=True)
    
    return {
        'ts': df_ts,
        'mature_rna': df_rna,
        'nascent_protein': df_prot,
        'combined': df_combined
    }


def save_ground_truth_csv(
    ground_truth: Dict[str, pd.DataFrame],
    output_dir: str,
    prefix: str = 'ground_truth'
) -> Dict[str, str]:
    """
    Save ground truth DataFrames to CSV files.
    
    Files created:
        {output_dir}/{prefix}_ts.csv
        {output_dir}/{prefix}_mature_rna.csv
        {output_dir}/{prefix}_nascent_protein.csv
        {output_dir}/{prefix}_combined.csv
        
    Returns:
        Dict mapping DataFrame name to file path
    """
    pass
```

### 9.5 Coordinate Conversion for DataFrames

```python
def sim_to_image_coords_for_df(
    sim_x: float, 
    sim_y: float, 
    sim_z: float,
    sim_vol_size: Tuple[int, int, int],
    img_size: Tuple[int, int] = (512, 512),
    num_z_slices: int = 12
) -> Tuple[float, float, float]:
    """
    Convert simulation coordinates to image coordinates for DataFrame.
    
    IMPORTANT: The DataFrame uses standard trackpy convention:
        - x: column position (horizontal)
        - y: row position (vertical)  
        - z: slice position (continuous, not integer)
    
    Args:
        sim_x, sim_y, sim_z: Simulation coordinates
        sim_vol_size: [X, Y, Z] simulation volume
        img_size: (Y, X) image size in pixels (rows, cols)
        num_z_slices: Number of z-slices
        
    Returns:
        (img_x, img_y, img_z) in image coordinates
        
    Note:
        img_x corresponds to image COLUMNS (horizontal position)
        img_y corresponds to image ROWS (vertical position)
    """
    # Scale XY from simulation to image
    scale_x = img_size[1] / sim_vol_size[0]  # X -> cols
    scale_y = img_size[0] / sim_vol_size[1]  # Y -> rows
    
    img_x = sim_x * scale_x
    img_y = sim_y * scale_y
    
    # Scale Z from simulation to slice range
    z_max = sim_vol_size[2]
    img_z = (sim_z / z_max) * (num_z_slices - 1)
    
    return img_x, img_y, img_z
```

### 9.6 Example DataFrame Output

#### Example: `ground_truth_ts.csv`

| frame | x | y | z | particle | mass | size | signal | time | molecule_type | channel | nascent_count | sim_x | sim_y | sim_z |
|-------|-----|-----|-----|----------|------|------|--------|------|---------------|---------|---------------|-------|-------|-------|
| 0 | 256.0 | 256.0 | 5.5 | 0 | 4712.4 | 2.5 | 1500.0 | 0.0 | TS | 0 | 3 | 256.0 | 256.0 | 45.8 |
| 1 | 256.0 | 256.0 | 5.5 | 0 | 6283.2 | 3.0 | 2000.0 | 1.0 | TS | 0 | 4 | 256.0 | 256.0 | 45.8 |
| 2 | 256.0 | 256.0 | 5.5 | 0 | 3141.6 | 2.2 | 1000.0 | 2.0 | TS | 0 | 2 | 256.0 | 256.0 | 45.8 |

#### Example: `ground_truth_mature_rna.csv`

| frame | x | y | z | particle | mass | size | signal | time | molecule_type | channel | sim_x | sim_y | sim_z | in_cytosol |
|-------|-------|-------|-----|----------|------|------|--------|------|---------------|---------|-------|-------|-------|------------|
| 5 | 280.3 | 310.5 | 4.2 | 7 | 2827.4 | 1.5 | 300.0 | 5.0 | mature_RNA | 1 | 280.3 | 310.5 | 35.0 | True |
| 5 | 195.1 | 288.9 | 6.8 | 12 | 2827.4 | 1.5 | 300.0 | 5.0 | mature_RNA | 1 | 195.1 | 288.9 | 56.7 | True |
| 6 | 281.1 | 311.2 | 4.3 | 7 | 2827.4 | 1.5 | 300.0 | 6.0 | mature_RNA | 1 | 281.1 | 311.2 | 35.8 | True |

#### Example: `ground_truth_nascent_protein.csv`

| frame | x | y | z | particle | mass | size | signal | time | molecule_type | channel | sim_x | sim_y | sim_z | parent_rna_id |
|-------|-------|-------|-----|----------|------|------|--------|------|---------------|---------|-------|-------|-------|---------------|
| 10 | 280.3 | 310.5 | 4.2 | 25 | 1256.6 | 1.5 | 200.0 | 10.0 | nascent_protein | 2 | 280.3 | 310.5 | 35.0 | 7 |
| 10 | 195.1 | 288.9 | 6.8 | 28 | 1256.6 | 1.5 | 200.0 | 10.0 | nascent_protein | 2 | 195.1 | 288.9 | 56.7 | 12 |

### 9.7 Updated Main Function Signature

```python
def simulate_microscopy_stack(
    sim_params: Optional[SimulationParameters] = None,
    micro_params: Optional[MicroscopyParameters] = None,
    simulation_results: Optional[Dict] = None,
    save_path: Optional[str] = None,
    save_ground_truth: bool = True,          # NEW
    ground_truth_dir: Optional[str] = None,  # NEW
    return_uint16: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, pd.DataFrame]]:  # UPDATED return type
    """
    Generate synthetic microscopy data and ground truth DataFrames.
    
    Args:
        ... (existing args) ...
        save_ground_truth: If True, save ground truth CSVs
        ground_truth_dir: Directory for ground truth CSVs (default: same as image)
        
    Returns:
        Tuple of:
            img5d: np.ndarray of shape (T, 12, 512, 512, 4)
            ground_truth: Dict of DataFrames with keys:
                'ts', 'mature_rna', 'nascent_protein', 'combined'
    """
    pass
```

### 9.8 Validation Against MicroLive

#### 9.8.1 Validation Workflow

1. **Generate synthetic data** → `img5d`, `ground_truth`
2. **Load into MicroLive** → Open TIFF, run tracking
3. **Export MicroLive results** → Get tracked particle DataFrames
4. **Compare against ground truth**:
   - Position error (RMSE in x, y, z)
   - Particle ID consistency (tracking accuracy)
   - Particle count per frame
   - Trajectory length distribution

#### 9.8.2 Validation Metrics

```python
def validate_tracking_against_ground_truth(
    tracked_df: pd.DataFrame,      # From MicroLive
    ground_truth_df: pd.DataFrame, # From simulation
    max_distance: float = 3.0,     # Max matching distance in pixels
    channel: Optional[int] = None  # Filter by channel
) -> Dict[str, float]:
    """
    Compute validation metrics comparing MicroLive output to ground truth.
    
    Returns:
        Dict with metrics:
            'detection_rate': Fraction of ground truth spots detected
            'false_positive_rate': Fraction of detections not matching GT
            'position_rmse_xy': RMSE of XY position error (pixels)
            'position_rmse_z': RMSE of Z position error (slices)
            'tracking_accuracy': Fraction of correct particle ID assignments
            'n_ground_truth': Total ground truth spots
            'n_detected': Total detected spots
    """
    pass
```

### 9.9 Implementation Timeline Update

Add to Phase 7 timeline:

| Step | Description | Estimated Time | Dependencies |
|------|-------------|----------------|--------------|
| 14 | Implement `generate_ts_dataframe()` | 30 min | Step 12 |
| 15 | Implement `generate_mature_rna_dataframe()` | 45 min | Step 12 |
| 16 | Implement `generate_nascent_protein_dataframe()` | 45 min | Step 12 |
| 17 | Implement `generate_all_ground_truth()` | 20 min | Steps 14-16 |
| 18 | Implement `save_ground_truth_csv()` | 15 min | Step 17 |
| 19 | Update main function + testing | 45 min | Steps 12-18 |
| 20 | Validation utilities | 60 min | Step 19 |

**Additional Time:** ~4-5 hours  
**New Total Estimated Time:** ~12-15 hours

---

## Appendix A: Reference Code Snippets

### Gaussian Spot Rendering (Efficient Implementation)

```python
def render_gaussian_spot_2d(image, center_yx, amplitude, sigma, truncate=4.0):
    """Efficiently render a 2D Gaussian onto image."""
    y_c, x_c = center_yx
    radius = int(np.ceil(truncate * sigma))
    
    # Compute patch bounds
    y_min = max(0, int(y_c) - radius)
    y_max = min(image.shape[0], int(y_c) + radius + 1)
    x_min = max(0, int(x_c) - radius)
    x_max = min(image.shape[1], int(x_c) + radius + 1)
    
    # Create coordinate grids for the patch
    y_coords = np.arange(y_min, y_max) - y_c
    x_coords = np.arange(x_min, x_max) - x_c
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Compute Gaussian
    gauss = amplitude * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Add to image
    image[y_min:y_max, x_min:x_max] += gauss
    return image
```

### Extracting State at Time t from Trajectories

```python
def get_state_at_time(results, time_idx):
    """Extract simulation state at a specific time index."""
    time_val = results['time_steps'][time_idx]
    
    # TS state
    ts_info = results['TS_trajectory'][time_idx]
    
    # Collect RNAs at this time
    nascent_rnas = []
    mature_rnas = []
    for rna_id, trajectory in results['RNA_trajectories'].items():
        # Find snapshot at this time
        for snap in trajectory:
            if abs(snap['time'] - time_val) < 1e-6:
                if snap.get('state', 'mature') == 'nascent':
                    nascent_rnas.append(snap)
                else:
                    mature_rnas.append(snap)
                break
    
    # Similar for proteins...
    return {
        'ts': ts_info,
        'nascent_rnas': nascent_rnas,
        'mature_rnas': mature_rnas,
        # ...
    }
```

### DataFrame Generation Implementation

```python
import pandas as pd

def generate_mature_rna_dataframe(
    simulation_results: Dict,
    micro_params: MicroscopyParameters,
    sim_vol_size: Tuple[int, int, int]
) -> pd.DataFrame:
    """Generate ground truth DataFrame for mature RNA spots."""
    
    rows = []
    time_steps = simulation_results['time_steps']
    
    for frame_idx, time_val in enumerate(time_steps):
        # Find all mature RNAs at this time
        for rna_id, trajectory in simulation_results['RNA_trajectories'].items():
            for snap in trajectory:
                if abs(snap['time'] - time_val) < 1e-6:
                    # Only include mature RNAs
                    if snap.get('state', 'mature') != 'mature':
                        continue
                    
                    # Get simulation position
                    sim_pos = snap['position']
                    sim_x, sim_y, sim_z = sim_pos[0], sim_pos[1], sim_pos[2]
                    
                    # Convert to image coordinates
                    img_x, img_y, img_z = sim_to_image_coords_for_df(
                        sim_x, sim_y, sim_z, sim_vol_size,
                        micro_params.image_size_xy, micro_params.num_z_slices
                    )
                    
                    # Compute properties
                    sigma = micro_params.psf_sigma_xy
                    amplitude = micro_params.mature_rna_amplitude
                    mass = amplitude * 2 * np.pi * sigma**2  # Integrated Gaussian
                    
                    rows.append({
                        'frame': frame_idx,
                        'x': img_x,
                        'y': img_y,
                        'z': img_z,
                        'particle': rna_id,  # Use original ID for tracking
                        'mass': mass,
                        'size': sigma,
                        'signal': amplitude,
                        'raw_mass': mass,
                        'ecc': 0.0,
                        'ep': 0.0,
                        'time': time_val,
                        'molecule_type': 'mature_RNA',
                        'channel': 1,
                        'sim_x': sim_x,
                        'sim_y': sim_y,
                        'sim_z': sim_z,
                        'in_cytosol': snap.get('in_cytosol', True)
                    })
                    break  # Found this RNA at this time
    
    return pd.DataFrame(rows)
```

---

## Appendix B: Channel Visualization Guide

For debugging and validation, generate max-intensity projections:

```python
def save_channel_preview(img5d, output_dir, frame_idx=0):
    """Save max projections for visual inspection."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    channel_names = ['C0: TS', 'C1: Mature RNA', 
                     'C2: Nascent Protein', 'C3: Mature Protein']
    
    for c, (ax, name) in enumerate(zip(axes.flat, channel_names)):
        # Max projection over Z
        max_proj = img5d[frame_idx, :, :, :, c].max(axis=0)
        im = ax.imshow(max_proj, cmap='gray')
        ax.set_title(name)
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/channel_preview_t{frame_idx:04d}.png")
    plt.close()
```

---

## Appendix C: Ground Truth Overlay Visualization

```python
def overlay_ground_truth_on_image(
    img5d: np.ndarray,
    ground_truth: Dict[str, pd.DataFrame],
    frame_idx: int = 0,
    output_path: str = 'ground_truth_overlay.png'
):
    """
    Create visualization with ground truth spots overlaid on image.
    
    Useful for verifying coordinate conversion is correct.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    channel_names = ['C0: TS', 'C1: Mature RNA', 'C2: Nascent Protein']
    df_keys = ['ts', 'mature_rna', 'nascent_protein']
    colors = ['red', 'green', 'blue']
    
    for i, (ax, name, df_key, color) in enumerate(zip(axes, channel_names, df_keys, colors)):
        # Max projection
        max_proj = img5d[frame_idx, :, :, :, i].max(axis=0)
        ax.imshow(max_proj, cmap='gray')
        ax.set_title(name)
        
        # Overlay ground truth positions
        df = ground_truth[df_key]
        frame_df = df[df['frame'] == frame_idx]
        
        for _, row in frame_df.iterrows():
            circle = Circle((row['x'], row['y']), radius=5, 
                           fill=False, color=color, linewidth=2)
            ax.add_patch(circle)
        
        ax.set_xlim(0, max_proj.shape[1])
        ax.set_ylim(max_proj.shape[0], 0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved ground truth overlay to {output_path}")
```

---

**End of Action Plan**
