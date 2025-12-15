#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microscopy Image Simulator
==========================

Generates synthetic fluorescence microscopy data from biological simulation.

This module takes the ground-truth particle states from minimal_simulation.py
and renders them as realistic microscopy images with:
- 5D output: [T, Z, Y, X, C] where C=4 channels
- PSF-convolved spots for TS, mature RNA, nascent proteins
- Diffuse signal for mature proteins
- Realistic noise model (read noise + shot noise + cellular texture)
- MicroLive-compatible ground truth DataFrames

Channel Definitions:
    C=0: Transcription Site (TS) - bulb intensity proportional to nascent RNAs
    C=1: Mature RNA - diffraction-limited spots
    C=2: Nascent Proteins - spots at translating RNA positions
    C=3: Mature Proteins - diffuse cytosolic haze

Authors: Luis U. Aguilera
Date: 2025-12-14
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Import the biology simulator
from minimal_simulation import GeneExpressionSimulator, SimulationParameters

# Optional imports for saving
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


# =============================================================================
# PARAMETERS DATACLASS
# =============================================================================

@dataclass
class MicroscopyParameters:
    """
    Parameters for microscopy image generation.
    
    Attributes are organized into logical groups:
    - Image dimensions
    - PSF parameters
    - Channel-specific amplitudes
    - Noise parameters
    - Baseline intensities
    
    Load from config.yaml using: MicroscopyParameters.from_yaml('config.yaml')
    """
    
    # Image Dimensions
    image_size_yx: Tuple[int, int] = (512, 512)  # (Y, X) = (rows, cols)
    num_z_slices: int = 12
    
    # Physical Parameters (for TIFF metadata)
    voxel_size_yx_nm: float = 100.0   # Pixel size XY in nanometers
    voxel_size_z_nm: float = 300.0    # Z-step in nanometers
    time_interval_s: float = 1.0      # Time between frames in seconds
    
    # PSF Parameters (in pixels)
    psf_sigma_xy: float = 1.5          # Lateral PSF width
    psf_sigma_z: float = 3.0           # Axial PSF width (in z-slice units)
    
    # Channel 0: Transcription Site
    ts_amplitude_per_nascent_rna: float = 500.0
    ts_sigma_base: float = 2.0
    ts_sigma_per_nascent_rna: float = 0.3
    ts_size_function: str = 'sqrt'     # 'sqrt' or 'log'
    ts_min_amplitude: float = 100.0    # Minimum TS amplitude when active
    
    # Channel 1: Mature RNA (INCREASED for visibility)
    mature_rna_amplitude: float = 800.0
    
    # Channel 2: Nascent Protein (INCREASED for visibility)
    nascent_protein_amplitude_per_chain: float = 400.0
    
    # Channel 3: Mature Protein
    mature_protein_scale: float = 0.2   # Intensity per protein (INCREASED)
    mature_protein_blur_sigma: float = 15.0  # Blur sigma to make diffuse
    
    # Baseline Intensities (camera offset / autofluorescence)
    intensity_outside_cell: float = 100.0
    intensity_cytosol: float = 120.0
    intensity_nucleus: float = 110.0
    
    # Noise Parameters
    read_noise_std: float = 5.0
    shot_noise_enabled: bool = True
    shot_noise_scale: float = 0.1      # Scale factor for Poisson noise
    cellular_noise_amplitude: float = 10.0
    cellular_noise_sigma: float = 8.0  # Spatial correlation in pixels
    
    # Photobleaching decay rates (s⁻¹) per channel
    # 0.0 = no photobleaching (default)
    # Typical: 0.001 s⁻¹ (from rSNAPed)
    photobleaching_rates: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0]
    )
    
    # Channel Names (for TIFF metadata)
    channel_names: List[str] = field(default_factory=lambda: ['TS', 'Mature_RNA', 'Nascent_Protein', 'Mature_Protein'])
    
    # Random Seed
    random_seed: Optional[int] = None
    
    # Output Options
    output_dtype: str = 'float32'      # 'float32' or 'uint16'
    uint16_clip_percentile: float = 99.9
    
    def validate(self) -> None:
        """Validate all parameters."""
        errors = []
        if self.num_z_slices < 1:
            errors.append("num_z_slices must be >= 1")
        if self.psf_sigma_xy <= 0:
            errors.append("psf_sigma_xy must be > 0")
        if self.psf_sigma_z <= 0:
            errors.append("psf_sigma_z must be > 0")
        if self.ts_size_function not in ('sqrt', 'log'):
            errors.append("ts_size_function must be 'sqrt' or 'log'")
        if self.image_size_yx[0] < 1 or self.image_size_yx[1] < 1:
            errors.append("image_size_yx dimensions must be >= 1")
        if errors:
            raise ValueError("Invalid microscopy parameters:\n  - " + "\n  - ".join(errors))
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'MicroscopyParameters':
        """
        Load microscopy parameters from a YAML config file.
        
        Expects a 'microscopy' section in the YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            MicroscopyParameters instance
        """
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        micro = config.get('microscopy', {})
        if not micro:
            print(f"Warning: No 'microscopy' section in {config_path}, using defaults")
            return cls()
        
        # Build kwargs from config
        kwargs = {}
        
        # Image dimensions
        if 'image_size_yx' in micro:
            kwargs['image_size_yx'] = tuple(micro['image_size_yx'])
        if 'num_z_slices' in micro:
            kwargs['num_z_slices'] = micro['num_z_slices']
        
        # Physical parameters
        if 'voxel_size_yx_nm' in micro:
            kwargs['voxel_size_yx_nm'] = micro['voxel_size_yx_nm']
        if 'voxel_size_z_nm' in micro:
            kwargs['voxel_size_z_nm'] = micro['voxel_size_z_nm']
        
        # Time interval: use simulation/frame_rate (not microscopy/time_interval_s)
        simulation = config.get('simulation', {})
        if 'frame_rate' in simulation:
            kwargs['time_interval_s'] = float(simulation['frame_rate'])
        elif 'time_interval_s' in micro:
            # Fallback to microscopy section if defined
            kwargs['time_interval_s'] = micro['time_interval_s']
        
        # PSF parameters
        psf = micro.get('psf', {})
        if 'sigma_xy' in psf:
            kwargs['psf_sigma_xy'] = psf['sigma_xy']
        if 'sigma_z' in psf:
            kwargs['psf_sigma_z'] = psf['sigma_z']
        
        # Amplitudes
        amp = micro.get('amplitudes', {})
        if 'ts_per_nascent_rna' in amp:
            kwargs['ts_amplitude_per_nascent_rna'] = amp['ts_per_nascent_rna']
        if 'ts_min' in amp:
            kwargs['ts_min_amplitude'] = amp['ts_min']
        if 'ts_sigma_base' in amp:
            kwargs['ts_sigma_base'] = amp['ts_sigma_base']
        if 'ts_sigma_per_rna' in amp:
            kwargs['ts_sigma_per_nascent_rna'] = amp['ts_sigma_per_rna']
        if 'ts_size_function' in amp:
            kwargs['ts_size_function'] = amp['ts_size_function']
        if 'mature_rna' in amp:
            kwargs['mature_rna_amplitude'] = amp['mature_rna']
        if 'nascent_protein_per_chain' in amp:
            kwargs['nascent_protein_amplitude_per_chain'] = amp['nascent_protein_per_chain']
        if 'mature_protein_scale' in amp:
            kwargs['mature_protein_scale'] = amp['mature_protein_scale']
        if 'mature_protein_blur_sigma' in amp:
            kwargs['mature_protein_blur_sigma'] = amp['mature_protein_blur_sigma']
        
        # Baseline intensities
        baseline = micro.get('baseline', {})
        if 'outside_cell' in baseline:
            kwargs['intensity_outside_cell'] = baseline['outside_cell']
        if 'cytosol' in baseline:
            kwargs['intensity_cytosol'] = baseline['cytosol']
        if 'nucleus' in baseline:
            kwargs['intensity_nucleus'] = baseline['nucleus']
        
        # Noise parameters
        noise = micro.get('noise', {})
        if 'read_noise_std' in noise:
            kwargs['read_noise_std'] = noise['read_noise_std']
        if 'shot_noise_enabled' in noise:
            kwargs['shot_noise_enabled'] = noise['shot_noise_enabled']
        if 'shot_noise_scale' in noise:
            kwargs['shot_noise_scale'] = noise['shot_noise_scale']
        if 'cellular_noise_amplitude' in noise:
            kwargs['cellular_noise_amplitude'] = noise['cellular_noise_amplitude']
        if 'cellular_noise_sigma' in noise:
            kwargs['cellular_noise_sigma'] = noise['cellular_noise_sigma']
        
        # Channel names
        if 'channel_names' in micro:
            kwargs['channel_names'] = micro['channel_names']
        
        # Photobleaching decay rates
        pb = micro.get('photobleaching', {})
        if pb:
            rates = [
                pb.get('channel_0_decay_rate', 0.0),
                pb.get('channel_1_decay_rate', 0.0),
                pb.get('channel_2_decay_rate', 0.0),
                pb.get('channel_3_decay_rate', 0.0),
            ]
            kwargs['photobleaching_rates'] = rates
        
        # Random seed
        if 'random_seed' in micro:
            kwargs['random_seed'] = micro['random_seed']
        
        params = cls(**kwargs)
        params.validate()
        return params


# =============================================================================
# COORDINATE CONVERSION FUNCTIONS
# =============================================================================

def sim_to_image_coords(
    sim_x: float, 
    sim_y: float, 
    sim_z: float,
    sim_vol_size: Tuple[int, int, int],
    img_size_yx: Tuple[int, int] = (512, 512),
    num_z_slices: int = 12
) -> Tuple[float, float, float]:
    """
    Convert simulation coordinates to image coordinates.
    
    Args:
        sim_x, sim_y, sim_z: Simulation coordinates (continuous)
        sim_vol_size: [X, Y, Z] simulation volume dimensions
        img_size_yx: (Y, X) image size in pixels (rows, cols)
        num_z_slices: Number of z-slices
        
    Returns:
        (img_y, img_x, img_z) in image coordinates where:
        - img_y: row position (0 to img_size_yx[0]-1)
        - img_x: column position (0 to img_size_yx[1]-1)
        - img_z: z-slice position (continuous, 0 to num_z_slices-1)
    """
    # Scale XY from simulation to image
    scale_x = img_size_yx[1] / sim_vol_size[0]  # X -> cols
    scale_y = img_size_yx[0] / sim_vol_size[1]  # Y -> rows
    
    img_x = sim_x * scale_x
    img_y = sim_y * scale_y
    
    # Scale Z from simulation to slice range
    # Z in simulation is typically [0, sim_vol_size[2]]
    z_max = sim_vol_size[2]
    if z_max > 0:
        img_z = (sim_z / z_max) * (num_z_slices - 1)
    else:
        img_z = 0.0
    
    return img_y, img_x, img_z


def map_z_to_slice_index(
    z_continuous: float,
    num_slices: int = 12
) -> int:
    """
    Map continuous z coordinate to nearest slice index.
    
    Args:
        z_continuous: Z position in image coordinates (0 to num_slices-1)
        num_slices: Number of z-slices in output
        
    Returns:
        Slice index in [0, num_slices-1]
    """
    slice_idx = int(round(z_continuous))
    return max(0, min(num_slices - 1, slice_idx))


# =============================================================================
# GEOMETRY: MASKS & BASELINE INTENSITIES
# =============================================================================

def make_cell_and_nucleus_masks(
    img_shape_zyx: Tuple[int, int, int],
    cytosol_radii_xyz: Tuple[float, float, float],
    nucleus_radii_xyz: Tuple[float, float, float],
    center_yx: Optional[Tuple[float, float]] = None,
    nucleus_offset_yx: Tuple[float, float] = (0, 0),
    half_ellipsoid: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3D boolean masks for cytosol and nucleus.
    
    Args:
        img_shape_zyx: (Z, Y, X) shape of output masks
        cytosol_radii_xyz: Half-axes of cytosol ellipsoid (X, Y, Z) in pixels
        nucleus_radii_xyz: Half-axes of nucleus ellipsoid (X, Y, Z) in pixels
        center_yx: Center of cell in (Y, X). If None, uses image center.
        nucleus_offset_yx: Offset of nucleus from cell center (Y, X)
        half_ellipsoid: If True, create half-ellipsoid (dome at z=0)
        
    Returns:
        cytosol_mask: Shape (Z, Y, X), True inside cytosol (excluding nucleus)
        nucleus_mask: Shape (Z, Y, X), True inside nucleus
    """
    Z, Y, X = img_shape_zyx
    
    if center_yx is None:
        center_y, center_x = Y / 2, X / 2
    else:
        center_y, center_x = center_yx
    
    # Create coordinate grids
    z_coords = np.arange(Z)
    y_coords = np.arange(Y)
    x_coords = np.arange(X)
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    
    # Cytosol: half-ellipsoid centered at z=0
    rx_cyto, ry_cyto, rz_cyto = cytosol_radii_xyz
    
    # Distance from center in normalized ellipsoid coordinates
    dx_cyto = (xx - center_x) / rx_cyto
    dy_cyto = (yy - center_y) / ry_cyto
    
    if half_ellipsoid:
        # For half-ellipsoid, z goes from 0 upward, center at z=0
        dz_cyto = zz / rz_cyto  # z=0 is base
        cytosol_mask = (dx_cyto**2 + dy_cyto**2 + dz_cyto**2 <= 1) & (zz >= 0)
    else:
        center_z = Z / 2
        dz_cyto = (zz - center_z) / rz_cyto
        cytosol_mask = (dx_cyto**2 + dy_cyto**2 + dz_cyto**2 <= 1)
    
    # Nucleus: smaller half-ellipsoid, possibly offset
    rx_nuc, ry_nuc, rz_nuc = nucleus_radii_xyz
    nuc_center_y = center_y + nucleus_offset_yx[0]
    nuc_center_x = center_x + nucleus_offset_yx[1]
    
    dx_nuc = (xx - nuc_center_x) / rx_nuc
    dy_nuc = (yy - nuc_center_y) / ry_nuc
    
    if half_ellipsoid:
        dz_nuc = zz / rz_nuc
        nucleus_mask = (dx_nuc**2 + dy_nuc**2 + dz_nuc**2 <= 1) & (zz >= 0)
    else:
        center_z = Z / 2
        dz_nuc = (zz - center_z) / rz_nuc
        nucleus_mask = (dx_nuc**2 + dy_nuc**2 + dz_nuc**2 <= 1)
    
    # Cytosol excludes nucleus
    cytosol_mask = cytosol_mask & ~nucleus_mask
    
    return cytosol_mask, nucleus_mask


def create_baseline_intensity_map(
    cytosol_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    intensity_outside: float = 100.0,
    intensity_cytosol: float = 150.0,
    intensity_nucleus: float = 120.0
) -> np.ndarray:
    """
    Create baseline intensity map for the image.
    
    Args:
        cytosol_mask: Boolean mask for cytosol
        nucleus_mask: Boolean mask for nucleus
        intensity_outside: Intensity outside cell
        intensity_cytosol: Intensity in cytosol
        intensity_nucleus: Intensity in nucleus
        
    Returns:
        baseline: Shape (Z, Y, X) float32 array
    """
    baseline = np.full(cytosol_mask.shape, intensity_outside, dtype=np.float32)
    baseline[cytosol_mask] = intensity_cytosol
    baseline[nucleus_mask] = intensity_nucleus
    return baseline


# =============================================================================
# PSF RENDERING PRIMITIVES
# =============================================================================

def render_gaussian_spot_3d(
    image: np.ndarray,
    position_zyx: Tuple[float, float, float],
    amplitude: float,
    sigma_xy: float,
    sigma_z: float,
    truncate: float = 4.0
) -> np.ndarray:
    """
    Render a 3D Gaussian PSF onto a volume (in-place).
    
    Args:
        image: 3D array (Z, Y, X) to render onto
        position_zyx: (z, y, x) center position
        amplitude: Peak intensity
        sigma_xy: Gaussian sigma in XY
        sigma_z: Gaussian sigma in Z
        truncate: Truncate Gaussian at this many sigmas
        
    Returns:
        The modified image array
    """
    z_c, y_c, x_c = position_zyx
    
    radius_z = int(np.ceil(truncate * sigma_z))
    radius_xy = int(np.ceil(truncate * sigma_xy))
    
    # Compute patch bounds
    z_min = max(0, int(z_c) - radius_z)
    z_max = min(image.shape[0], int(z_c) + radius_z + 1)
    y_min = max(0, int(y_c) - radius_xy)
    y_max = min(image.shape[1], int(y_c) + radius_xy + 1)
    x_min = max(0, int(x_c) - radius_xy)
    x_max = min(image.shape[2], int(x_c) + radius_xy + 1)
    
    if z_max <= z_min or y_max <= y_min or x_max <= x_min:
        return image
    
    # Create coordinate grids for the patch
    z_coords = np.arange(z_min, z_max) - z_c
    y_coords = np.arange(y_min, y_max) - y_c
    x_coords = np.arange(x_min, x_max) - x_c
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    
    # Compute 3D Gaussian (anisotropic: different sigma in Z)
    gauss = amplitude * np.exp(
        -(xx**2 + yy**2) / (2 * sigma_xy**2) 
        - zz**2 / (2 * sigma_z**2)
    )
    
    # Add to image
    image[z_min:z_max, y_min:y_max, x_min:x_max] += gauss
    return image


# =============================================================================
# CHANNEL-SPECIFIC RENDERING
# =============================================================================

def render_ts_channel(
    image_3d: np.ndarray,
    ts_position_zyx: Tuple[float, float, float],
    nascent_rna_count: int,
    params: MicroscopyParameters
) -> np.ndarray:
    """
    Render Transcription Site as Gaussian with dynamic amplitude AND size.
    
    The TS appears as a bright "bulb" whose intensity and size scale with
    the number of nascent RNAs being transcribed.
    
    Args:
        image_3d: 3D array (Z, Y, X) to render onto
        ts_position_zyx: TS position in image coordinates (z, y, x)
        nascent_rna_count: Number of nascent RNAs at the TS
        params: MicroscopyParameters
        
    Returns:
        The modified image array
    """
    if nascent_rna_count <= 0:
        return image_3d
    
    # Amplitude scales linearly with nascent RNA count
    amplitude = max(
        params.ts_min_amplitude,
        params.ts_amplitude_per_nascent_rna * nascent_rna_count
    )
    
    # Size scales with f(N) where f is sqrt or log
    if params.ts_size_function == 'sqrt':
        size_factor = np.sqrt(nascent_rna_count)
    else:  # 'log'
        size_factor = np.log1p(nascent_rna_count)
    
    sigma_xy = params.ts_sigma_base + params.ts_sigma_per_nascent_rna * size_factor
    sigma_z = sigma_xy * (params.psf_sigma_z / params.psf_sigma_xy)  # Maintain aspect ratio
    
    render_gaussian_spot_3d(
        image_3d, ts_position_zyx, amplitude, sigma_xy, sigma_z
    )
    
    return image_3d


def render_mature_rna_channel(
    image_3d: np.ndarray,
    rna_positions_zyx: List[Tuple[float, float, float]],
    params: MicroscopyParameters
) -> np.ndarray:
    """
    Render each mature RNA as a diffraction-limited Gaussian spot.
    
    Args:
        image_3d: 3D array (Z, Y, X) to render onto
        rna_positions_zyx: List of (z, y, x) positions in image coordinates
        params: MicroscopyParameters
        
    Returns:
        The modified image array
    """
    for pos in rna_positions_zyx:
        render_gaussian_spot_3d(
            image_3d, pos,
            amplitude=params.mature_rna_amplitude,
            sigma_xy=params.psf_sigma_xy,
            sigma_z=params.psf_sigma_z
        )
    
    return image_3d


def render_nascent_protein_channel(
    image_3d: np.ndarray,
    translation_sites: List[Dict],
    params: MicroscopyParameters
) -> np.ndarray:
    """
    Render nascent protein signal at each translating RNA position.
    
    Args:
        image_3d: 3D array (Z, Y, X) to render onto
        translation_sites: List of dicts with 'position_zyx' and 'nascent_count'
        params: MicroscopyParameters
        
    Returns:
        The modified image array
    """
    for site in translation_sites:
        pos = site['position_zyx']
        n_nascent = site['nascent_count']
        
        if n_nascent <= 0:
            continue
        
        amplitude = params.nascent_protein_amplitude_per_chain * n_nascent
        
        render_gaussian_spot_3d(
            image_3d, pos,
            amplitude=amplitude,
            sigma_xy=params.psf_sigma_xy,
            sigma_z=params.psf_sigma_z
        )
    
    return image_3d


def render_mature_protein_channel(
    image_3d: np.ndarray,
    mature_protein_count: int,
    cytosol_mask: np.ndarray,
    params: MicroscopyParameters
) -> np.ndarray:
    """
    Render mature proteins as diffuse cytosolic haze.
    
    The signal is spread throughout the cytosol and blurred to appear
    as out-of-focus background, not as distinct spots.
    
    Args:
        image_3d: 3D array (Z, Y, X) to render onto
        mature_protein_count: Total number of mature proteins
        cytosol_mask: Boolean mask for cytosol region (Z, Y, X)
        params: MicroscopyParameters
        
    Returns:
        The modified image array
    """
    if mature_protein_count <= 0:
        return image_3d
    
    # Add intensity proportional to protein count within cytosol
    intensity_boost = params.mature_protein_scale * mature_protein_count
    
    # Create a temporary layer for protein signal
    protein_layer = np.zeros_like(image_3d)
    protein_layer[cytosol_mask] = intensity_boost
    
    # Apply strong Gaussian blur to make it diffuse
    # Blur each z-slice independently for speed
    for z in range(protein_layer.shape[0]):
        protein_layer[z] = gaussian_filter(
            protein_layer[z], 
            sigma=params.mature_protein_blur_sigma
        )
    
    image_3d += protein_layer
    return image_3d


# =============================================================================
# NOISE MODEL
# =============================================================================

def add_noise(
    image: np.ndarray,
    cell_mask: np.ndarray,
    params: MicroscopyParameters,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Add realistic microscopy noise to an image.
    
    Noise model:
        output = signal + read_noise + shot_noise + cellular_texture
        
    Args:
        image: 3D array (Z, Y, X) to add noise to
        cell_mask: Boolean mask for cell region (for cellular noise)
        params: MicroscopyParameters
        rng: Optional random generator for reproducibility
        
    Returns:
        The modified image array (modified in-place)
    """
    if rng is None:
        rng = np.random.default_rng(params.random_seed)
    
    # 1. Gaussian read noise (camera noise)
    if params.read_noise_std > 0:
        read_noise = rng.normal(0, params.read_noise_std, image.shape)
        image += read_noise.astype(np.float32)
    
    # 2. Poisson shot noise (photon noise)
    if params.shot_noise_enabled and params.shot_noise_scale > 0:
        # Shot noise is proportional to sqrt(signal)
        signal_positive = np.maximum(image, 0)
        shot_noise = rng.poisson(signal_positive * params.shot_noise_scale)
        shot_noise = (shot_noise - signal_positive * params.shot_noise_scale) / np.sqrt(
            np.maximum(signal_positive * params.shot_noise_scale, 1)
        )
        image += shot_noise.astype(np.float32) * np.sqrt(signal_positive)
    
    # 3. Cellular texture noise (spatially correlated noise inside cell)
    if params.cellular_noise_amplitude > 0 and cell_mask.any():
        # Create white noise
        white_noise = rng.normal(0, 1, image.shape)
        
        # Apply low-pass filter for spatial correlation
        for z in range(image.shape[0]):
            white_noise[z] = gaussian_filter(
                white_noise[z], 
                sigma=params.cellular_noise_sigma
            )
        
        # Scale and apply only inside cell
        cellular_noise = white_noise * params.cellular_noise_amplitude
        image[cell_mask] += cellular_noise[cell_mask].astype(np.float32)
    
    return image


# =============================================================================
# POST-PROCESSING
# =============================================================================

def apply_photobleaching(
    img5d: np.ndarray,
    time_interval_s: float,
    decay_rates: List[float],
    verbose: bool = False
) -> np.ndarray:
    """
    Apply channel-dependent photobleaching (exponential decay).
    
    Model: I(t) = I(t) * exp(-k * t)
    
    Applied uniformly to all z-slices at each time point.
    
    Args:
        img5d: 5D array (T, Z, Y, X, C)
        time_interval_s: Time between frames (seconds)
        decay_rates: List of decay constants k (s⁻¹) per channel
        verbose: Print progress
        
    Returns:
        Modified img5d array (in-place)
    """
    T, Z, Y, X, C = img5d.shape
    
    for c in range(min(C, len(decay_rates))):
        k = decay_rates[c]
        if k > 0:
            if verbose:
                print(f"    Channel {c}: k={k:.4f} s⁻¹")
            for t in range(T):
                time_s = t * time_interval_s
                decay_factor = np.exp(-k * time_s)
                # Apply decay to all z-slices uniformly
                img5d[t, :, :, :, c] *= decay_factor
    
    return img5d


def to_uint16(
    image: np.ndarray,
    clip_percentile: float = 99.9,
    scale_to: int = 65535
) -> np.ndarray:
    """
    Convert float32 image to uint16 with proper scaling.
    
    Args:
        image: Float image
        clip_percentile: Percentile for clipping bright pixels
        scale_to: Maximum value after scaling
        
    Returns:
        uint16 image
    """
    # Clip to percentile to avoid outliers dominating scaling
    vmax = np.percentile(image, clip_percentile)
    vmin = image.min()
    
    if vmax > vmin:
        scaled = (image - vmin) / (vmax - vmin) * scale_to
    else:
        scaled = np.zeros_like(image)
    
    return np.clip(scaled, 0, scale_to).astype(np.uint16)


def save_tiff_for_microlive(
    img5d: np.ndarray,
    output_path: str,
    voxel_size_yx_nm: float = 100.0,
    voxel_size_z_nm: float = 300.0,
    time_interval_s: float = 1.0,
    channel_names: Optional[List[str]] = None,
    clip_percentile: float = 99.9,
    verbose: bool = True
) -> str:
    """
    Save 5D image as TIFF with MicroLive-compatible metadata.
    
    MicroLive expects:
    - TIFF with axes 'TCZYX' (reordered from our TZYXC)
    - PhysicalSizeX, PhysicalSizeZ in µm (stored as JSON in ImageDescription)
    - TimeIncrement in seconds
    - Channel names
    
    Args:
        img5d: 5D array with shape (T, Z, Y, X, C)
        output_path: Path for output TIFF file
        voxel_size_yx_nm: Pixel size in XY (nanometers)
        voxel_size_z_nm: Pixel size in Z (nanometers)
        time_interval_s: Time between frames (seconds)
        channel_names: Names for each channel
        clip_percentile: Percentile for intensity clipping
        verbose: Print progress
        
    Returns:
        Path to saved file
    """
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required for TIFF export. Install with: pip install tifffile")
    
    if channel_names is None:
        channel_names = ['TS', 'Mature_RNA', 'Nascent_Protein', 'Mature_Protein']
    
    T, Z, Y, X, C = img5d.shape
    
    if verbose:
        print(f"Saving TIFF for MicroLive: {output_path}")
        print(f"  Shape: {img5d.shape} -> reordering to TCZYX")
    
    # Convert to uint16 with proper scaling
    img_uint16 = np.zeros((T, Z, Y, X, C), dtype=np.uint16)
    for c in range(C):
        img_uint16[:, :, :, :, c] = to_uint16(
            img5d[:, :, :, :, c], 
            clip_percentile=clip_percentile
        )
    
    # Reorder from TZYXC to TCZYX (MicroLive format)
    img_tczyx = np.moveaxis(img_uint16, 4, 1)  # C from axis 4 to axis 1
    
    # Prepare metadata (MicroLive reads this from ImageDescription as JSON)
    metadata = {
        'axes': 'TCZYX',
        'PhysicalSizeX': voxel_size_yx_nm / 1000.0,  # Convert nm to µm
        'PhysicalSizeY': voxel_size_yx_nm / 1000.0,
        'PhysicalSizeZ': voxel_size_z_nm / 1000.0,
        'TimeIncrement': time_interval_s,
        'TimeIncrementUnit': 's',
        'SignificantBits': 16,
        'Channel': {
            'Name': channel_names[:C]
        }
    }
    
    if verbose:
        print(f"  Metadata: PhysicalSizeX={metadata['PhysicalSizeX']:.3f}µm, "
              f"PhysicalSizeZ={metadata['PhysicalSizeZ']:.3f}µm, "
              f"TimeIncrement={metadata['TimeIncrement']}s")
        print(f"  Channels: {channel_names[:C]}")
    
    # Save with tifffile (with ZLIB compression to reduce file size)
    tifffile.imwrite(
        output_path,
        img_tczyx,
        shape=img_tczyx.shape,
        dtype='uint16',
        compression='zlib',  # ZLIB compression (compatible with ImageJ/FIJI)
        imagej=False,
        metadata=metadata
    )
    
    if verbose:
        print(f"  Saved: {output_path}")
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.1f} MB")
    
    return output_path


def save_masks_as_tiff(
    cytosol_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    output_dir: str,
    cell_id: int = 1,
    voxel_size_yx_nm: float = 100.0,
    voxel_size_z_nm: float = 300.0,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Save cytosol and nucleus masks as TIFF files.
    
    Masks are saved with labeled integers:
    - Background = 0
    - Cell/nucleus = cell_id (default 1)
    
    This supports future multi-cell simulations where each cell has a unique ID.
    
    Args:
        cytosol_mask: 3D boolean array (Z, Y, X) for cytosol
        nucleus_mask: 3D boolean array (Z, Y, X) for nucleus
        output_dir: Directory to save mask TIFFs
        cell_id: Label ID for the cell (default 1)
        voxel_size_yx_nm: Pixel size in XY (nanometers)
        voxel_size_z_nm: Z-step size (nanometers)
        verbose: Print progress
        
    Returns:
        Dict with paths: {'cytosol': path, 'nucleus': path}
    """
    if not HAS_TIFFFILE:
        raise ImportError("tifffile is required for TIFF export. Install with: pip install tifffile")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Convert boolean masks to labeled uint16
    cytosol_labeled = (cytosol_mask.astype(np.uint16)) * cell_id
    nucleus_labeled = (nucleus_mask.astype(np.uint16)) * cell_id
    
    # Metadata for ImageJ/MicroLive compatibility
    metadata = {
        'axes': 'ZYX',
        'PhysicalSizeX': voxel_size_yx_nm / 1000.0,  # Convert nm to µm
        'PhysicalSizeY': voxel_size_yx_nm / 1000.0,
        'PhysicalSizeZ': voxel_size_z_nm / 1000.0,
        'Unit': 'um'
    }
    
    paths = {}
    
    # Save cytosol mask
    cyto_path = str(output_path / "mask_cytosol.tif")
    tifffile.imwrite(
        cyto_path,
        cytosol_labeled,
        dtype='uint16',
        metadata=metadata
    )
    paths['cytosol'] = cyto_path
    
    # Save nucleus mask
    nuc_path = str(output_path / "mask_nucleus.tif")
    tifffile.imwrite(
        nuc_path,
        nucleus_labeled,
        dtype='uint16',
        metadata=metadata
    )
    paths['nucleus'] = nuc_path
    
    if verbose:
        print(f"Saved masks to {output_dir}:")
        print(f"  - mask_cytosol.tif (shape={cytosol_labeled.shape}, label={cell_id})")
        print(f"  - mask_nucleus.tif (shape={nucleus_labeled.shape}, label={cell_id})")
    
    return paths


def save_simulation_metadata(
    output_dir: str,
    sim_params: 'SimulationParameters',
    micro_params: 'MicroscopyParameters',
    image_shape: Tuple[int, ...],
    ground_truth_stats: Optional[Dict] = None,
    verbose: bool = True
) -> str:
    """
    Save simulation metadata in MicroLive-compatible format.
    
    Creates a human-readable text file with sections for:
    - Image Properties
    - Biology Simulation Parameters
    - Microscopy Parameters
    - Photobleaching Settings
    - Ground Truth Statistics
    
    Format is compatible with MicroLive's Metadata class.
    
    Args:
        output_dir: Directory to save metadata file
        sim_params: SimulationParameters used for simulation
        micro_params: MicroscopyParameters used for rendering
        image_shape: Shape of output image (T, Z, Y, X, C)
        ground_truth_stats: Optional dict with ground truth counts
        verbose: Print progress
        
    Returns:
        Path to saved metadata file
    """
    from datetime import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    metadata_file = str(output_path / "simulation_metadata.txt")
    
    line_width = 70
    separator = '=' * line_width
    sub_separator = '-' * line_width
    
    with open(metadata_file, 'w') as fd:
        # Helper functions (matching MicroLive format)
        def write_section(title):
            fd.write(f'\n{separator}\n')
            fd.write(f'{title.upper()}\n')
            fd.write(f'{separator}\n')
        
        def write_subsection(title):
            fd.write(f'\n{sub_separator}\n')
            fd.write(f'{title}\n')
            fd.write(f'{sub_separator}\n')
        
        def write_value(label, value, indent=4):
            fd.write(f'{" " * indent}{label:.<40} {value}\n')
        
        # Header
        fd.write(f'{separator}\n')
        fd.write('VIRTUAL CELL SIMULATION METADATA\n')
        fd.write(f'{separator}\n')
        fd.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        fd.write(f'Format: MicroLive-compatible\n')
        
        # Image Properties Section
        write_section('Image Properties')
        T, Z, Y, X, C = image_shape
        write_value('Dimensions (T, Z, Y, X, C)', f'({T}, {Z}, {Y}, {X}, {C})')
        write_value('Total Frames', T)
        write_value('Z Slices', Z)
        write_value('Image Size (Y, X)', f'({Y}, {X})')
        write_value('Channels', C)
        write_value('Pixel Size XY (nm)', micro_params.voxel_size_yx_nm)
        write_value('Pixel Size Z (nm)', micro_params.voxel_size_z_nm)
        write_value('Time Interval (s)', micro_params.time_interval_s)
        write_value('Channel Names', ', '.join(micro_params.channel_names[:C]))
        
        # Biology Simulation Section
        write_section('Biology Simulation')
        write_subsection('Kinetic Parameters')
        write_value('k_on (TS activation)', f'{sim_params.k_on} s⁻¹')
        write_value('k_off (TS deactivation)', f'{sim_params.k_off} s⁻¹')
        write_value('k_r (RNA initiation)', f'{sim_params.k_r} s⁻¹')
        write_value('k_p (Protein translation)', f'{sim_params.k_p} s⁻¹')
        write_value('RNA degradation (gamma_r)', f'{sim_params.gamma_r} s⁻¹')
        write_value('Protein degradation (gamma_p)', f'{sim_params.gamma_p} s⁻¹')
        
        write_subsection('Diffusion & Transport')
        write_value('RNA diffusion coeff', f'{sim_params.k_diff_r} px²/s')
        write_value('Protein diffusion coeff', f'{sim_params.k_diff_p} px²/s')
        write_value('Transport rate', f'{sim_params.transport_rate} s⁻¹')
        
        write_subsection('Timing')
        write_value('Total time', f'{sim_params.total_time} s')
        write_value('Frame rate', f'{sim_params.frame_rate} s')
        write_value('Burn-in time', f'{sim_params.burnin_time} s')
        write_value('Random seed', sim_params.random_seed or 'None (random)')
        
        # Microscopy Parameters Section
        write_section('Microscopy Parameters')
        write_subsection('PSF Settings')
        write_value('Sigma XY (pixels)', micro_params.psf_sigma_xy)
        write_value('Sigma Z (z-slices)', micro_params.psf_sigma_z)
        
        write_subsection('Channel Amplitudes')
        write_value('TS amplitude/nascent chain', micro_params.ts_amplitude_per_nascent_rna)
        write_value('Mature RNA amplitude', micro_params.mature_rna_amplitude)
        write_value('Nascent protein amplitude', micro_params.nascent_protein_amplitude_per_chain)
        write_value('Mature protein scale', micro_params.mature_protein_scale)
        
        write_subsection('Baseline Intensities')
        write_value('Outside cell', micro_params.intensity_outside_cell)
        write_value('Cytosol', micro_params.intensity_cytosol)
        write_value('Nucleus', micro_params.intensity_nucleus)
        
        write_subsection('Noise Parameters')
        write_value('Read noise std', micro_params.read_noise_std)
        write_value('Shot noise enabled', 'Yes' if micro_params.shot_noise_enabled else 'No')
        write_value('Shot noise scale', micro_params.shot_noise_scale)
        write_value('Cellular noise amplitude', micro_params.cellular_noise_amplitude)
        write_value('Cellular noise sigma', micro_params.cellular_noise_sigma)
        
        # Photobleaching Section
        write_section('Photobleaching')
        rates = micro_params.photobleaching_rates
        has_photobleaching = any(k > 0 for k in rates)
        write_value('Enabled', 'Yes' if has_photobleaching else 'No')
        if has_photobleaching:
            for i, rate in enumerate(rates[:C]):
                ch_name = micro_params.channel_names[i] if i < len(micro_params.channel_names) else f'Ch{i}'
                write_value(f'Channel {i} ({ch_name})', f'{rate} s⁻¹')
        
        # Ground Truth Section
        write_section('Ground Truth')
        if ground_truth_stats:
            for key, value in ground_truth_stats.items():
                write_value(key.replace('_', ' ').title(), value)
        else:
            write_value('Available', 'Yes (see CSV files)')
        
        # Files Section
        write_section('Output Files')
        write_value('Main TIFF', 'simulated_microscopy.tif')
        write_value('Cytosol mask', 'mask_cytosol.tif')
        write_value('Nucleus mask', 'mask_nucleus.tif')
        write_value('Ground truth (TS)', 'ground_truth_ts.csv')
        write_value('Ground truth (RNA)', 'ground_truth_mature_rna.csv')
        write_value('Ground truth (Protein)', 'ground_truth_nascent_protein.csv')
        write_value('Ground truth (Combined)', 'ground_truth_combined.csv')
        write_value('Metadata', 'simulation_metadata.txt')
        
        fd.write(f'\n{separator}\n')
    
    if verbose:
        print(f"Saved metadata to {metadata_file}")
    
    return metadata_file


# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_channel_projections(
    img5d: np.ndarray,
    frame_idx: int = 0,
    output_path: Optional[str] = None,
    channel_names: Optional[List[str]] = None,
    colormaps: Optional[List[str]] = None,
    min_percentile: float = 1.0,
    max_percentile: float = 99.5,
    figsize: Tuple[int, int] = (16, 4),
    show_colorbar: bool = True,
    show_plot: bool = True
) -> None:
    """
    Generate maximum projection visualization for each channel.
    
    Similar to MicroLive/rsnaped visualization style.
    
    Args:
        img5d: 5D array (T, Z, Y, X, C)
        frame_idx: Time frame to visualize
        output_path: Optional path to save PNG
        channel_names: Names for each channel
        colormaps: Matplotlib colormaps for each channel
        min_percentile: Percentile for intensity minimum
        max_percentile: Percentile for intensity maximum
        figsize: Figure size
        show_colorbar: Whether to show colorbars
        show_plot: Whether to display the plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Default channel names
    if channel_names is None:
        channel_names = ['TS (C0)', 'Mature RNA (C1)', 'Nascent Protein (C2)', 'Mature Protein (C3)']
    
    # Create custom colormaps similar to MicroLive
    def make_colormap(color_name):
        """Create black-to-color colormap."""
        colors = {
            'green': ((0, 0, 0), (0, 1, 0)),
            'magenta': ((0, 0, 0), (1, 0, 1)),
            'cyan': ((0, 0, 0), (0, 1, 1)),
            'yellow': ((0, 0, 0), (1, 1, 0)),
            'red': ((0, 0, 0), (1, 0, 0)),
        }
        if color_name in colors:
            return LinearSegmentedColormap.from_list(
                f'black_{color_name}', colors[color_name], N=256
            )
        return color_name  # Return as-is if it's a matplotlib colormap name
    
    if colormaps is None:
        colormaps = ['green', 'magenta', 'cyan', 'yellow']
    
    num_channels = img5d.shape[4]
    
    fig, axes = plt.subplots(1, num_channels, figsize=figsize)
    if num_channels == 1:
        axes = [axes]
    
    for c in range(num_channels):
        ax = axes[c]
        
        # Max projection over Z
        max_proj = img5d[frame_idx, :, :, :, c].max(axis=0)
        
        # Percentile-based normalization
        vmin = np.percentile(max_proj, min_percentile)
        vmax = np.percentile(max_proj, max_percentile)
        
        # Apply normalization
        normalized = np.clip((max_proj - vmin) / (vmax - vmin + 1e-8), 0, 1)
        
        # Get colormap
        cmap = make_colormap(colormaps[c % len(colormaps)])
        
        # Display
        im = ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(channel_names[c] if c < len(channel_names) else f'Channel {c}', 
                     fontsize=12, fontweight='bold')
        ax.axis('off')
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Intensity (a.u.)', fontsize=8)
    
    plt.suptitle(f'Max Projections - Frame {frame_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved visualization to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_ground_truth_overlay(
    img5d: np.ndarray,
    ground_truth: Dict[str, pd.DataFrame],
    frame_idx: int = 0,
    output_path: Optional[str] = None,
    spot_radius: int = 8,
    show_plot: bool = True
) -> None:
    """
    Create visualization with ground truth spots overlaid on max projections.
    
    Args:
        img5d: 5D array (T, Z, Y, X, C)
        ground_truth: Dict of DataFrames with 'ts', 'mature_rna', 'nascent_protein'
        frame_idx: Time frame to visualize
        output_path: Optional path to save PNG
        spot_radius: Circle radius for spot markers
        show_plot: Whether to display the plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.colors import LinearSegmentedColormap
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    channel_names = ['TS (C0)', 'Mature RNA (C1)', 'Nascent Protein (C2)']
    df_keys = ['ts', 'mature_rna', 'nascent_protein']
    colors = ['lime', 'magenta', 'cyan']
    
    # Custom colormaps
    cmaps = []
    for color in ['green', 'magenta', 'cyan']:
        cdict = {
            'green': ((0, 0, 0), (0, 1, 0)),
            'magenta': ((0, 0, 0), (1, 0, 1)),
            'cyan': ((0, 0, 0), (0, 1, 1)),
        }
        cmaps.append(LinearSegmentedColormap.from_list(f'black_{color}', cdict[color], N=256))
    
    for i, (ax, name, df_key, color, cmap) in enumerate(zip(axes, channel_names, df_keys, colors, cmaps)):
        # Max projection
        max_proj = img5d[frame_idx, :, :, :, i].max(axis=0)
        
        # Normalize
        vmin = np.percentile(max_proj, 1)
        vmax = np.percentile(max_proj, 99.5)
        normalized = np.clip((max_proj - vmin) / (vmax - vmin + 1e-8), 0, 1)
        
        ax.imshow(normalized, cmap=cmap)
        ax.set_title(name, fontsize=12, fontweight='bold')
        
        # Overlay ground truth positions
        if df_key in ground_truth:
            df = ground_truth[df_key]
            frame_df = df[df['frame'] == frame_idx]
            
            for _, row in frame_df.iterrows():
                circle = Circle((row['x'], row['y']), radius=spot_radius, 
                               fill=False, color='white', linewidth=1.5)
                ax.add_patch(circle)
            
            ax.text(10, 30, f'n={len(frame_df)}', color='white', fontsize=10, 
                   fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        ax.axis('off')
    
    plt.suptitle(f'Ground Truth Overlay - Frame {frame_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved ground truth overlay to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_composite_image(
    img5d: np.ndarray,
    frame_idx: int = 0,
    channels: List[int] = [0, 1, 2],
    colors: List[Tuple[float, float, float]] = None,
    output_path: Optional[str] = None,
    max_percentile: float = 99.5,
    show_plot: bool = True
) -> np.ndarray:
    """
    Create RGB composite from multiple channels.
    
    Args:
        img5d: 5D array (T, Z, Y, X, C)
        frame_idx: Time frame
        channels: Which channels to include (max 3 for RGB)
        colors: RGB colors for each channel, e.g., [(0,1,0), (1,0,1)]
        output_path: Optional save path
        max_percentile: Percentile for max normalization
        show_plot: Whether to display
        
    Returns:
        RGB composite image as uint8 array
    """
    import matplotlib.pyplot as plt
    
    if colors is None:
        colors = [(0, 1, 0), (1, 0, 1), (0, 1, 1)]  # Green, Magenta, Cyan
    
    Y, X = img5d.shape[2], img5d.shape[3]
    composite = np.zeros((Y, X, 3), dtype=np.float32)
    
    for i, ch in enumerate(channels[:3]):
        if ch >= img5d.shape[4]:
            continue
            
        # Max projection
        max_proj = img5d[frame_idx, :, :, :, ch].max(axis=0)
        
        # Normalize
        vmin = np.percentile(max_proj, 1)
        vmax = np.percentile(max_proj, max_percentile)
        normalized = np.clip((max_proj - vmin) / (vmax - vmin + 1e-8), 0, 1)
        
        # Add to composite with color
        color = colors[i % len(colors)]
        for c in range(3):
            composite[:, :, c] += normalized * color[c]
    
    # Clip to [0, 1]
    composite = np.clip(composite, 0, 1)
    
    if show_plot or output_path:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(composite)
        ax.axis('off')
        ax.set_title(f'Composite (Frame {frame_idx})', fontsize=12, fontweight='bold')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
            print(f"Saved composite to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    return (composite * 255).astype(np.uint8)

def get_state_at_time(
    results: Dict,
    time_idx: int
) -> Dict:
    """
    Extract simulation state at a specific time index.
    
    Args:
        results: Simulation results from GeneExpressionSimulator.run()
        time_idx: Time frame index
        
    Returns:
        Dict with keys: 'ts', 'nascent_rnas', 'mature_rnas', 
                       'nascent_proteins', 'mature_proteins', 'time'
    """
    time_val = results['time_steps'][time_idx]
    
    # TS state
    ts_info = results['TS_trajectory'][time_idx]
    
    # Collect RNAs at this time
    nascent_rnas = []
    mature_rnas = []
    
    for rna_id, trajectory in results['RNA_trajectories'].items():
        for snap in trajectory:
            if abs(snap['time'] - time_val) < 1e-6:
                if snap.get('state', 'mature') == 'nascent':
                    snap_copy = snap.copy()
                    snap_copy['rna_id'] = rna_id
                    nascent_rnas.append(snap_copy)
                else:
                    snap_copy = snap.copy()
                    snap_copy['rna_id'] = rna_id
                    mature_rnas.append(snap_copy)
                break
    
    # Collect Proteins at this time
    nascent_proteins = []
    mature_proteins = []
    
    for prot_id, trajectory in results['Protein_trajectories'].items():
        for snap in trajectory:
            if abs(snap['time'] - time_val) < 1e-6:
                if snap.get('state', 'mature') == 'nascent':
                    snap_copy = snap.copy()
                    snap_copy['protein_id'] = prot_id
                    nascent_proteins.append(snap_copy)
                else:
                    snap_copy = snap.copy()
                    snap_copy['protein_id'] = prot_id
                    mature_proteins.append(snap_copy)
                break
    
    return {
        'ts': ts_info,
        'nascent_rnas': nascent_rnas,
        'mature_rnas': mature_rnas,
        'nascent_proteins': nascent_proteins,
        'mature_proteins': mature_proteins,
        'time': time_val
    }


# =============================================================================
# GROUND TRUTH DATAFRAME GENERATION
# =============================================================================

def generate_ts_dataframe(
    results: Dict,
    ts_position: np.ndarray,
    micro_params: MicroscopyParameters,
    sim_vol_size: Tuple[int, int, int],
    cell_areas: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate ground truth DataFrame for the Transcription Site.
    
    Args:
        results: Simulation results
        ts_position: TS position in simulation coordinates [x, y, z]
        micro_params: Microscopy parameters
        sim_vol_size: Simulation volume [X, Y, Z]
        cell_areas: Optional dict with 'nuc_area', 'cyto_area', 'nuc_loc', 'cyto_loc'
        
    Returns:
        DataFrame with MicroLive-compatible columns
    """
    rows = []
    time_steps = results['time_steps']
    
    # Convert TS position to image coordinates once
    ts_y, ts_x, ts_z = sim_to_image_coords(
        ts_position[0], ts_position[1], ts_position[2],
        sim_vol_size, micro_params.image_size_yx, micro_params.num_z_slices
    )
    
    for frame_idx, time_val in enumerate(time_steps):
        # Count nascent RNAs at this time
        state = get_state_at_time(results, frame_idx)
        nascent_count = len(state['nascent_rnas'])
        
        # TS is only "visible" if there are nascent RNAs
        if nascent_count > 0:
            # Compute intensity/size based on nascent count
            if micro_params.ts_size_function == 'sqrt':
                size = micro_params.ts_sigma_base + micro_params.ts_sigma_per_nascent_rna * np.sqrt(nascent_count)
            else:
                size = micro_params.ts_sigma_base + micro_params.ts_sigma_per_nascent_rna * np.log1p(nascent_count)
            
            amplitude = micro_params.ts_amplitude_per_nascent_rna * nascent_count
            
            row = {
                'frame': frame_idx,
                'x': ts_x,
                'y': ts_y,
                'z': ts_z,
                'particle': 0,  # TS is always particle 0
                'image_id': 0,
                'cell_id': 0,
                'spot_id': 0,
                'spot_type': 0,  # TS channel
                'cluster_size': nascent_count,
                'is_nuc': True,
                'is_cluster': nascent_count > 1,
                'is_cell_fragmented': 0,
                'time': time_val,
                'molecule_type': 'TS',
                'channel': 0,
                'sim_x': ts_position[0],
                'sim_y': ts_position[1],
                'sim_z': ts_position[2],
                'nascent_count': nascent_count,
                'ground_truth': True,
                'psf_amplitude_ch_0': amplitude,
                'psf_sigma_ch_0': size,
            }
            
            # Add cell area info if available
            if cell_areas:
                row.update({
                    'nuc_loc_y': cell_areas.get('nuc_loc_y', np.nan),
                    'nuc_loc_x': cell_areas.get('nuc_loc_x', np.nan),
                    'cyto_loc_y': cell_areas.get('cyto_loc_y', np.nan),
                    'cyto_loc_x': cell_areas.get('cyto_loc_x', np.nan),
                    'nuc_area_px': cell_areas.get('nuc_area_px', np.nan),
                    'cyto_area_px': cell_areas.get('cyto_area_px', np.nan),
                    'cell_area_px': cell_areas.get('cell_area_px', np.nan),
                })
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def generate_mature_rna_dataframe(
    results: Dict,
    micro_params: MicroscopyParameters,
    sim_vol_size: Tuple[int, int, int],
    cell_areas: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate ground truth DataFrame for mature RNA spots.
    
    Args:
        results: Simulation results
        micro_params: Microscopy parameters
        sim_vol_size: Simulation volume [X, Y, Z]
        cell_areas: Optional dict with cell geometry info
        
    Returns:
        DataFrame with MicroLive-compatible columns
    """
    rows = []
    time_steps = results['time_steps']
    
    for frame_idx, time_val in enumerate(time_steps):
        state = get_state_at_time(results, frame_idx)
        
        for rna in state['mature_rnas']:
            pos = rna['position']
            sim_x, sim_y, sim_z = pos[0], pos[1], pos[2]
            
            # Convert to image coordinates
            img_y, img_x, img_z = sim_to_image_coords(
                sim_x, sim_y, sim_z,
                sim_vol_size, micro_params.image_size_yx, micro_params.num_z_slices
            )
            
            row = {
                'frame': frame_idx,
                'x': img_x,
                'y': img_y,
                'z': img_z,
                'particle': rna['rna_id'],
                'image_id': 0,
                'cell_id': 0,
                'spot_id': rna['rna_id'],
                'spot_type': 1,  # Mature RNA channel
                'cluster_size': 1,
                'is_nuc': not rna.get('in_cytosol', True),
                'is_cluster': False,
                'is_cell_fragmented': 0,
                'time': time_val,
                'molecule_type': 'mature_RNA',
                'channel': 1,
                'sim_x': sim_x,
                'sim_y': sim_y,
                'sim_z': sim_z,
                'in_cytosol': rna.get('in_cytosol', True),
                'ground_truth': True,
                'psf_amplitude_ch_1': micro_params.mature_rna_amplitude,
                'psf_sigma_ch_1': micro_params.psf_sigma_xy,
            }
            
            if cell_areas:
                row.update({
                    'nuc_loc_y': cell_areas.get('nuc_loc_y', np.nan),
                    'nuc_loc_x': cell_areas.get('nuc_loc_x', np.nan),
                    'cyto_loc_y': cell_areas.get('cyto_loc_y', np.nan),
                    'cyto_loc_x': cell_areas.get('cyto_loc_x', np.nan),
                    'nuc_area_px': cell_areas.get('nuc_area_px', np.nan),
                    'cyto_area_px': cell_areas.get('cyto_area_px', np.nan),
                    'cell_area_px': cell_areas.get('cell_area_px', np.nan),
                })
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def generate_nascent_protein_dataframe(
    results: Dict,
    micro_params: MicroscopyParameters,
    sim_vol_size: Tuple[int, int, int],
    cell_areas: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate ground truth DataFrame for nascent protein spots.
    
    Args:
        results: Simulation results
        micro_params: Microscopy parameters
        sim_vol_size: Simulation volume [X, Y, Z]
        cell_areas: Optional dict with cell geometry info
        
    Returns:
        DataFrame with MicroLive-compatible columns
    """
    rows = []
    time_steps = results['time_steps']
    
    for frame_idx, time_val in enumerate(time_steps):
        state = get_state_at_time(results, frame_idx)
        
        for prot in state['nascent_proteins']:
            pos = prot['position']
            sim_x, sim_y, sim_z = pos[0], pos[1], pos[2]
            
            # Convert to image coordinates
            img_y, img_x, img_z = sim_to_image_coords(
                sim_x, sim_y, sim_z,
                sim_vol_size, micro_params.image_size_yx, micro_params.num_z_slices
            )
            
            row = {
                'frame': frame_idx,
                'x': img_x,
                'y': img_y,
                'z': img_z,
                'particle': prot['protein_id'],
                'image_id': 0,
                'cell_id': 0,
                'spot_id': prot['protein_id'],
                'spot_type': 2,  # Nascent protein channel
                'cluster_size': 1,
                'is_nuc': False,  # Nascent proteins are on cytoplasmic RNAs
                'is_cluster': False,
                'is_cell_fragmented': 0,
                'time': time_val,
                'molecule_type': 'nascent_protein',
                'channel': 2,
                'sim_x': sim_x,
                'sim_y': sim_y,
                'sim_z': sim_z,
                'parent_rna_id': prot.get('parent_rna_id', -1),
                'ground_truth': True,
                'psf_amplitude_ch_2': micro_params.nascent_protein_amplitude_per_chain,
                'psf_sigma_ch_2': micro_params.psf_sigma_xy,
            }
            
            if cell_areas:
                row.update({
                    'nuc_loc_y': cell_areas.get('nuc_loc_y', np.nan),
                    'nuc_loc_x': cell_areas.get('nuc_loc_x', np.nan),
                    'cyto_loc_y': cell_areas.get('cyto_loc_y', np.nan),
                    'cyto_loc_x': cell_areas.get('cyto_loc_x', np.nan),
                    'nuc_area_px': cell_areas.get('nuc_area_px', np.nan),
                    'cyto_area_px': cell_areas.get('cyto_area_px', np.nan),
                    'cell_area_px': cell_areas.get('cell_area_px', np.nan),
                })
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def generate_all_ground_truth(
    results: Dict,
    ts_position: np.ndarray,
    micro_params: MicroscopyParameters,
    sim_vol_size: Tuple[int, int, int],
    cell_areas: Optional[Dict] = None
) -> Dict[str, pd.DataFrame]:
    """
    Generate all ground truth DataFrames.
    
    Returns:
        Dictionary with keys: 'ts', 'mature_rna', 'nascent_protein', 'combined'
    """
    df_ts = generate_ts_dataframe(results, ts_position, micro_params, sim_vol_size, cell_areas)
    df_rna = generate_mature_rna_dataframe(results, micro_params, sim_vol_size, cell_areas)
    df_prot = generate_nascent_protein_dataframe(results, micro_params, sim_vol_size, cell_areas)
    
    # Create combined DataFrame
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
    
    Returns:
        Dict mapping DataFrame name to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    for name, df in ground_truth.items():
        filepath = output_path / f"{prefix}_{name}.csv"
        df.to_csv(filepath, index=False)
        paths[name] = str(filepath)
    
    return paths


# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================

def simulate_microscopy_stack(
    sim_params: Optional[SimulationParameters] = None,
    micro_params: Optional[MicroscopyParameters] = None,
    simulation_results: Optional[Dict] = None,
    ts_position: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    save_ground_truth: bool = True,
    save_masks: bool = True,
    ground_truth_dir: Optional[str] = None,
    return_uint16: bool = False,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, pd.DataFrame]]:
    """
    Generate synthetic microscopy data and ground truth DataFrames.
    
    This is the main entry point for the microscopy simulator.
    
    Args:
        sim_params: SimulationParameters for the biology simulation.
                   If None, uses defaults.
        micro_params: MicroscopyParameters for image rendering.
                     If None, uses defaults.
        simulation_results: Optional pre-run simulation results.
                           If None, runs the simulation internally.
        ts_position: TS position in simulation coordinates. Required if
                    using pre-computed simulation_results.
        save_path: Optional path to save output TIFF.
        save_ground_truth: If True, save ground truth CSVs.
        save_masks: If True, save cytosol/nucleus masks as TIFFs.
        ground_truth_dir: Directory for ground truth CSVs and masks.
        return_uint16: If True, return uint16 instead of float32.
        verbose: Print progress information.
        
    Returns:
        Tuple of:
            img5d: np.ndarray of shape (T, Z, Y, X, C) where C=4
            ground_truth: Dict of DataFrames with keys:
                'ts', 'mature_rna', 'nascent_protein', 'combined'
    """
    # Initialize parameters
    if sim_params is None:
        sim_params = SimulationParameters()
    if micro_params is None:
        micro_params = MicroscopyParameters()
    
    micro_params.validate()
    
    # Set random seed
    if micro_params.random_seed is not None:
        rng = np.random.default_rng(micro_params.random_seed)
    else:
        rng = np.random.default_rng()
    
    # Run biology simulation if needed
    if simulation_results is None:
        if verbose:
            print("Running biology simulation...")
        
        params_dict = sim_params.to_dict()
        simulator = GeneExpressionSimulator(params_dict)
        simulation_results = simulator.run()
        ts_position = simulator.transcription_site
        sim_vol_size = tuple(params_dict['simulation_volume_size'])
        
        if verbose:
            print(f"  Simulation complete: {len(simulation_results['time_steps'])} frames")
    else:
        if ts_position is None:
            raise ValueError("ts_position is required when using pre-computed simulation_results")
        sim_vol_size = tuple(sim_params.simulation_volume_size)
    
    # Get dimensions
    T = len(simulation_results['time_steps'])
    Z = micro_params.num_z_slices
    Y, X = micro_params.image_size_yx
    C = 4  # Always 4 channels
    
    if verbose:
        print(f"Generating microscopy stack: shape ({T}, {Z}, {Y}, {X}, {C})")
    
    # Create geometry masks for image space
    # Convert simulation geometry to image space
    cytosol_size = sim_params.cytosol_size if hasattr(sim_params, 'cytosol_size') else [350, 350, 80]
    nucleus_size = sim_params.nucleus_size if hasattr(sim_params, 'nucleus_size') else [120, 100, 60]
    
    # Scale radii from simulation to image coordinates
    scale_x = X / sim_vol_size[0]
    scale_y = Y / sim_vol_size[1]
    scale_z = Z / sim_vol_size[2] if sim_vol_size[2] > 0 else 1
    
    cytosol_radii_img = (
        cytosol_size[0] / 2 * scale_x,
        cytosol_size[1] / 2 * scale_y,
        cytosol_size[2] / 2 * scale_z
    )
    nucleus_radii_img = (
        nucleus_size[0] / 2 * scale_x,
        nucleus_size[1] / 2 * scale_y,
        nucleus_size[2] / 2 * scale_z
    )
    
    if verbose:
        print("  Creating cell geometry masks...")
    
    cytosol_mask, nucleus_mask = make_cell_and_nucleus_masks(
        img_shape_zyx=(Z, Y, X),
        cytosol_radii_xyz=cytosol_radii_img,
        nucleus_radii_xyz=nucleus_radii_img,
        half_ellipsoid=True
    )
    cell_mask = cytosol_mask | nucleus_mask
    
    # Compute cell areas for ground truth
    cell_areas = {
        'nuc_area_px': np.sum(nucleus_mask),
        'cyto_area_px': np.sum(cytosol_mask),
        'cell_area_px': np.sum(cell_mask),
        'nuc_loc_y': Y / 2,
        'nuc_loc_x': X / 2,
        'cyto_loc_y': Y / 2,
        'cyto_loc_x': X / 2,
    }
    
    # Create baseline intensity map
    baseline = create_baseline_intensity_map(
        cytosol_mask, nucleus_mask,
        micro_params.intensity_outside_cell,
        micro_params.intensity_cytosol,
        micro_params.intensity_nucleus
    )
    
    # Allocate output array
    img5d = np.zeros((T, Z, Y, X, C), dtype=np.float32)
    
    if verbose:
        print("  Rendering frames...")
    
    # Render each frame
    for t in range(T):
        if verbose and t % 10 == 0:
            print(f"    Frame {t}/{T}")
        
        # Get state at this time
        state = get_state_at_time(simulation_results, t)
        
        # Initialize channels with baseline
        for c in range(C):
            img5d[t, :, :, :, c] = baseline.copy()
        
        # Channel 0: Transcription Site
        ts_y, ts_x, ts_z = sim_to_image_coords(
            ts_position[0], ts_position[1], ts_position[2],
            sim_vol_size, (Y, X), Z
        )
        nascent_rna_count = len(state['nascent_rnas'])
        render_ts_channel(
            img5d[t, :, :, :, 0],
            (ts_z, ts_y, ts_x),
            nascent_rna_count,
            micro_params
        )
        
        # Channel 1: Mature RNA
        rna_positions = []
        for rna in state['mature_rnas']:
            pos = rna['position']
            y, x, z = sim_to_image_coords(pos[0], pos[1], pos[2], sim_vol_size, (Y, X), Z)
            rna_positions.append((z, y, x))
        
        render_mature_rna_channel(img5d[t, :, :, :, 1], rna_positions, micro_params)
        
        # Channel 2: Nascent Proteins
        # Group nascent proteins by their parent RNA
        translation_sites = {}
        for prot in state['nascent_proteins']:
            parent_id = prot.get('parent_rna_id', -1)
            pos = prot['position']
            y, x, z = sim_to_image_coords(pos[0], pos[1], pos[2], sim_vol_size, (Y, X), Z)
            
            if parent_id not in translation_sites:
                translation_sites[parent_id] = {
                    'position_zyx': (z, y, x),
                    'nascent_count': 0
                }
            translation_sites[parent_id]['nascent_count'] += 1
        
        render_nascent_protein_channel(
            img5d[t, :, :, :, 2],
            list(translation_sites.values()),
            micro_params
        )
        
        # Channel 3: Mature Proteins
        mature_prot_count = len(state['mature_proteins'])
        render_mature_protein_channel(
            img5d[t, :, :, :, 3],
            mature_prot_count,
            cytosol_mask,
            micro_params
        )
        
        # Add noise to all channels
        for c in range(C):
            add_noise(img5d[t, :, :, :, c], cell_mask, micro_params, rng)
    
    # Apply photobleaching (exponential decay) if any rates > 0
    if any(k > 0 for k in micro_params.photobleaching_rates):
        if verbose:
            print("  Applying photobleaching...")
        apply_photobleaching(
            img5d,
            micro_params.time_interval_s,
            micro_params.photobleaching_rates,
            verbose=verbose
        )
    
    if verbose:
        print("  Generating ground truth DataFrames...")
    
    # Generate ground truth
    ground_truth = generate_all_ground_truth(
        simulation_results, ts_position, micro_params, sim_vol_size, cell_areas
    )
    
    # Save ground truth if requested
    if save_ground_truth and ground_truth_dir is not None:
        if verbose:
            print(f"  Saving ground truth to {ground_truth_dir}")
        save_ground_truth_csv(ground_truth, ground_truth_dir)
    
    # Save masks if requested
    if save_masks and ground_truth_dir is not None:
        if verbose:
            print(f"  Saving masks to {ground_truth_dir}")
        save_masks_as_tiff(
            cytosol_mask, nucleus_mask,
            ground_truth_dir,
            cell_id=1,
            voxel_size_yx_nm=micro_params.voxel_size_yx_nm,
            voxel_size_z_nm=micro_params.voxel_size_z_nm,
            verbose=verbose
        )
    
    # Save simulation metadata
    if ground_truth_dir is not None:
        gt_stats = {
            'ts_entries': len(ground_truth.get('ts', [])),
            'mature_rna_spots': len(ground_truth.get('mature_rna', [])),
            'nascent_protein_spots': len(ground_truth.get('nascent_protein', [])),
            'combined_entries': len(ground_truth.get('combined', [])),
        }
        save_simulation_metadata(
            ground_truth_dir,
            sim_params,
            micro_params,
            img5d.shape,
            ground_truth_stats=gt_stats,
            verbose=verbose
        )
    
    # Convert to uint16 if requested
    if return_uint16:
        if verbose:
            print("  Converting to uint16...")
        for c in range(C):
            img5d[:, :, :, :, c] = to_uint16(
                img5d[:, :, :, :, c], 
                micro_params.uint16_clip_percentile
            )
        img5d = img5d.astype(np.uint16)
    
    # Save TIFF if requested
    if save_path is not None and HAS_TIFFFILE:
        if verbose:
            print(f"  Saving to {save_path}")
        
        # Ensure output is uint16 for TIFF
        if img5d.dtype != np.uint16:
            img_to_save = np.zeros_like(img5d, dtype=np.uint16)
            for c in range(C):
                img_to_save[:, :, :, :, c] = to_uint16(img5d[:, :, :, :, c])
        else:
            img_to_save = img5d
        
        tifffile.imwrite(save_path, img_to_save, imagej=True)
    
    if verbose:
        print("Done!")
        print(f"  Output shape: {img5d.shape}")
        print(f"  Output dtype: {img5d.dtype}")
        for c in range(C):
            ch_data = img5d[:, :, :, :, c]
            print(f"  Channel {c}: min={ch_data.min():.1f}, max={ch_data.max():.1f}, mean={ch_data.mean():.1f}")
    
    return img5d, ground_truth


# =============================================================================
# EXAMPLE / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Microscopy Image Simulator - Test Run")
    print("=" * 60)
    
    # Load parameters from shared config file
    config_path = Path(__file__).parent / "config.yaml"
    
    if config_path.exists():
        print(f"Loading parameters from: {config_path}")
        sim_params = SimulationParameters.from_yaml(str(config_path))
        micro_params = MicroscopyParameters.from_yaml(str(config_path))
        print(f"  Simulation: total_time={sim_params.total_time}s, frame_rate={sim_params.frame_rate}s, burnin={sim_params.burnin_time}s")
    else:
        print("Config file not found, using defaults")
        sim_params = SimulationParameters(
            total_time=50.0,
            frame_rate=1.0,
            burnin_time=100.0,
            random_seed=42
        )
        micro_params = MicroscopyParameters(random_seed=42)
    
    print(f"  Microscopy amplitudes: mature_rna={micro_params.mature_rna_amplitude}, "
          f"nascent_protein={micro_params.nascent_protein_amplitude_per_chain}")
    
    # Create output directory
    output_dir = Path("./results_simulation")
    output_dir.mkdir(exist_ok=True)
    
    # Run simulation
    img5d, ground_truth = simulate_microscopy_stack(
        sim_params=sim_params,
        micro_params=micro_params,
        save_ground_truth=True,
        ground_truth_dir=str(output_dir),
        verbose=True
    )
    
    # Print validation results
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    
    expected_shape = (50, 12, 512, 512, 4)
    actual_shape = img5d.shape
    
    print(f"Shape: {actual_shape}")
    print(f"Expected: {expected_shape}")
    print(f"Shape OK: {actual_shape == expected_shape}")
    
    print(f"\nGround Truth DataFrames:")
    for name, df in ground_truth.items():
        print(f"  {name}: {len(df)} rows, {len(df.columns)} columns")
    
    print("\nChannel Statistics:")
    channel_names = ['TS', 'Mature RNA', 'Nascent Prot', 'Mature Prot']
    for c, name in enumerate(channel_names):
        ch = img5d[:, :, :, :, c]
        print(f"  C{c} ({name}): min={ch.min():.1f}, max={ch.max():.1f}, mean={ch.mean():.1f}")
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # Visualize multiple frames
    for frame in [0, 10, 25, 40]:
        if frame < img5d.shape[0]:
            # Channel projections
            visualize_channel_projections(
                img5d, 
                frame_idx=frame,
                output_path=str(output_dir / f"channels_frame_{frame:03d}.png"),
                show_plot=False
            )
            
            # Ground truth overlay
            plot_ground_truth_overlay(
                img5d,
                ground_truth,
                frame_idx=frame,
                output_path=str(output_dir / f"overlay_frame_{frame:03d}.png"),
                show_plot=False
            )
    
    # Create composite for frame 25
    create_composite_image(
        img5d,
        frame_idx=25,
        channels=[0, 1, 2],
        output_path=str(output_dir / "composite_frame_025.png"),
        show_plot=False
    )
    
    # Save TIFF with MicroLive-compatible metadata
    print("\n" + "=" * 60)
    print("Saving TIFF for MicroLive")
    print("=" * 60)
    
    tiff_path = str(output_dir / "simulated_microscopy.tif")
    save_tiff_for_microlive(
        img5d,
        output_path=tiff_path,
        voxel_size_yx_nm=micro_params.voxel_size_yx_nm,
        voxel_size_z_nm=micro_params.voxel_size_z_nm,
        time_interval_s=micro_params.time_interval_s,
        channel_names=micro_params.channel_names,
        verbose=True
    )
    
    print(f"\nVisualizations and TIFF saved to: {output_dir.absolute()}")
    print("\nFiles created:")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")


