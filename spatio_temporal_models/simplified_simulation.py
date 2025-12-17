"""
Simplified Gene Expression Simulator (Tau-Leap Style)
======================================================

A faster alternative to the full Gillespie SSA in full_simulation.py.
Uses fixed time steps and batch diffusion for dramatically improved performance.

Key differences:
- Fixed time step (dt) instead of exact SSA exponential waiting times
- Batch diffusion: all molecules move every dt (not stochastic reactions)
- Poisson-sampled reaction counts per step
- Same output format as the full model for compatibility

Usage:
    python simplified_simulation.py --config config.yaml
"""

import numpy as np
import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path
# Import plotting functions from separate module
from plotting import plot_molecule_concentrations, plot_all_projections, generate_temporal_gif


@dataclass
class SimplifiedSimulationParameters:
    """
    Parameters for simplified 3D gene expression simulation.
    
    Inherits most parameters from the full model but adds:
    - dt: Fixed time step for tau-leap simulation
    """
    
    # Kinetic rates
    k_on: float = 0.2
    k_off: float = 0.1
    k_r: float = 3.0
    gamma_r: float = 0.05
    k_p: float = 0.9
    gamma_p: float = 0.65
    k_diff_r: float = 10.0
    k_diff_p: float = 10.0
    transport_rate: float = 2.0
    
    # Elongation times (deterministic)
    RNA_elongation_time: float = 60.0
    Protein_elongation_time: float = 30.0
    
    # Geometry
    simulation_volume_size: List[int] = field(default_factory=lambda: [512, 512, 100])
    cytosol_size: List[int] = field(default_factory=lambda: [350, 350, 80])
    nucleus_size: List[int] = field(default_factory=lambda: [120, 100, 60])
    nucleus_xy_offset: List[int] = field(default_factory=lambda: [0, 0])
    
    # Simulation settings
    total_time: float = 100.0
    frame_rate: float = 1.0
    burnin_time: float = 0.0
    random_seed: int = None
    
    # Simplified model specific
    dt: float = 0.1  # Fixed time step (seconds)
    
    # Options
    position_TS: str = 'random'
    movement_protein_into_nucleus: bool = False
    
    # Drug perturbation
    apply_drug: bool = False
    drug_application_time: float = 120.0
    inhibited_parameters: Dict[str, float] = field(default_factory=lambda: {'transport_rate': 0.1})
    
    # Output options
    generate_gif: bool = False
    gif_fps: int = 5
    gif_skip_frames: int = 1
    gif_dpi: int = 80
    gif_show_surfaces: bool = True
    gif_surface_decimation: int = 4
    show_nascent_separately: bool = True
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate()
    
    def _validate(self):
        """Check critical parameters are valid."""
        errors = []
        
        if self.dt <= 0:
            errors.append(f"dt must be positive, got {self.dt}")
        if self.total_time <= 0:
            errors.append(f"total_time must be positive, got {self.total_time}")
        if self.frame_rate <= 0:
            errors.append(f"frame_rate must be positive, got {self.frame_rate}")
            
        # Check geometry
        for i, (n, c) in enumerate(zip(self.nucleus_size, self.cytosol_size)):
            if n > c:
                axis = ['X', 'Y', 'Z'][i]
                errors.append(f"nucleus_size {axis}={n} exceeds cytosol_size {axis}={c}")
        
        if errors:
            raise ValueError("Invalid simulation parameters:\n  - " + "\n  - ".join(errors))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            'k_on': self.k_on, 'k_off': self.k_off, 'k_r': self.k_r,
            'gamma_r': self.gamma_r, 'k_p': self.k_p, 'gamma_p': self.gamma_p,
            'k_diff_r': self.k_diff_r, 'k_diff_p': self.k_diff_p,
            'transport_rate': self.transport_rate,
            'RNA_elongation_time': self.RNA_elongation_time,
            'Protein_elongation_time': self.Protein_elongation_time,
            'simulation_volume_size': list(self.simulation_volume_size),
            'cytosol_size': list(self.cytosol_size),
            'nucleus_size': list(self.nucleus_size),
            'nucleus_xy_offset': list(self.nucleus_xy_offset),
            'total_time': self.total_time, 'frame_rate': self.frame_rate,
            'burnin_time': self.burnin_time, 'random_seed': self.random_seed,
            'dt': self.dt,
            'position_TS': self.position_TS,
            'movement_protein_into_nucleus': self.movement_protein_into_nucleus,
            'apply_drug': self.apply_drug,
            'drug_application_time': self.drug_application_time,
            'inhibited_parameters': dict(self.inhibited_parameters),
            'generate_gif': self.generate_gif, 'gif_fps': self.gif_fps,
            'gif_skip_frames': self.gif_skip_frames, 'gif_dpi': self.gif_dpi,
            'gif_show_surfaces': self.gif_show_surfaces,
            'gif_surface_decimation': self.gif_surface_decimation,
            'show_nascent_separately': self.show_nascent_separately,
        }
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'SimplifiedSimulationParameters':
        """Load parameters from a YAML configuration file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        params = {}
        
        # Kinetics
        if 'kinetics' in config:
            params.update(config['kinetics'])
        
        # Elongation times
        if 'elongation' in config:
            params.update(config['elongation'])
        
        # Geometry
        if 'geometry' in config:
            params.update(config['geometry'])
        
        # Simulation
        if 'simulation' in config:
            params.update(config['simulation'])
            # Get simplified model dt
            if 'simplified_dt' in config['simulation']:
                params['dt'] = config['simulation']['simplified_dt']
        
        # Transcription site
        if 'transcription_site' in config:
            if 'position' in config['transcription_site']:
                params['position_TS'] = config['transcription_site']['position']
        
        # Movement
        if 'movement' in config:
            if 'protein_into_nucleus' in config['movement']:
                params['movement_protein_into_nucleus'] = config['movement']['protein_into_nucleus']
        
        # Drug
        if 'drug' in config:
            if 'apply' in config['drug']:
                params['apply_drug'] = config['drug']['apply']
            if 'application_time' in config['drug']:
                params['drug_application_time'] = config['drug']['application_time']
            if 'inhibited_parameters' in config['drug']:
                params['inhibited_parameters'] = config['drug']['inhibited_parameters']
        
        # Output options
        if 'output' in config:
            output = config['output']
            for key in ['generate_gif', 'gif_fps', 'gif_skip_frames', 'gif_dpi',
                        'gif_show_surfaces', 'gif_surface_decimation', 'show_nascent_separately']:
                if key in output:
                    params[key] = output[key]
        
        # Filter to only known fields (ignore selector keys like use_simplified_model)
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_params = {k: v for k, v in params.items() if k in known_fields}
        
        return cls(**filtered_params)


class SimplifiedGeneExpressionSimulator:
    """
    Tau-leap style gene expression simulator for fast testing.
    
    Key optimizations over the full Gillespie model:
    1. Fixed time step (dt) instead of exponential SSA waiting times
    2. Batch diffusion: all molecules move every dt (vectorized)
    3. Poisson-sampled reaction counts per step
    
    Produces the same output format as GeneExpressionSimulator for compatibility.
    """
    
    def __init__(self, params: Dict):
        # Set random seed for reproducibility
        random_seed = params.get('random_seed', None)
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.dt = params.get('dt', 0.1)  # Fixed time step
        self.TS_state = False
        self.RNAs = {}
        self.Proteins = {}
        self.next_rna_id = 1
        self.next_protein_id = 1
        
        # Geometry setup (same as full model)
        simulation_volume_size = params['simulation_volume_size'][:3]
        center_of_box = np.array(simulation_volume_size) / 2
        self.center_of_box = center_of_box
        self.simulation_volume_size = simulation_volume_size
        
        nucleus_xy_offset = params.get('nucleus_xy_offset', [0, 0])
        self.nucleus_mask = np.zeros(simulation_volume_size, dtype=bool)
        self.cytosol_mask = np.zeros(simulation_volume_size, dtype=bool)
        self.nucleus_size = params['nucleus_size']
        self.rates = params.copy()
        
        # Half-ellipsoid geometry
        cytosol_radii = np.array(params['cytosol_size']) / 2
        nucleus_radii = np.array(params['nucleus_size']) / 2
        
        self._cytosol_radii = cytosol_radii
        self._cytosol_center = np.array([center_of_box[0], center_of_box[1], 0])
        self._nucleus_radii = nucleus_radii
        self._nucleus_center = np.array([
            center_of_box[0] + nucleus_xy_offset[0],
            center_of_box[1] + nucleus_xy_offset[1],
            0
        ])
        self.nucleus_center = self._nucleus_center
        
        # Transport zone
        self.transport_zone_threshold = 2
        shrunk_size = np.array(params['nucleus_size']) - 2 * self.transport_zone_threshold
        shrunk_size = np.clip(shrunk_size, a_min=self.transport_zone_threshold, a_max=None)
        self._shrunk_radii = shrunk_size / 2
        
        # Place transcription site
        if params['position_TS'] == 'center':
            nucleus_radius_z = params['nucleus_size'][2] / 2
            self.transcription_site = np.array([
                self.nucleus_center[0], self.nucleus_center[1], nucleus_radius_z / 2
            ])
        else:
            self.transcription_site = self._find_random_TS_position()
        
        # Tracking
        self.RNA_trajectories = {}
        self.Protein_trajectories = {}
        self.frame_rate = params['frame_rate']
        self.time_steps = []
        self.TS_trajectory = []
        self.transport_rate = params['transport_rate']
        self.small_distance_outside_nucleus = 1
        self.movement_protein_into_nucleus = params['movement_protein_into_nucleus']
        self.burnin_time = params.get('burnin_time', 0)
        self.next_save_time = self.burnin_time
        self.total_time = params['total_time'] + self.burnin_time
        self.drug_application_time = params['drug_application_time'] + self.burnin_time
        self.inhibited_parameters = params['inhibited_parameters']
        self.apply_drug = params['apply_drug']
        self.parameters_updated = False
        
        # Elongation times
        self.RNA_elongation_time = params.get('RNA_elongation_time', 60.0)
        self.Protein_elongation_time = params.get('Protein_elongation_time', 30.0)
    
    def _find_random_TS_position(self):
        """Find random position inside nucleus."""
        nucleus_radii = np.array(self.rates['nucleus_size']) / 2
        for _ in range(10000):
            random_offset = np.array([
                np.random.uniform(-nucleus_radii[0], nucleus_radii[0]),
                np.random.uniform(-nucleus_radii[1], nucleus_radii[1]),
                np.random.uniform(0, nucleus_radii[2])
            ])
            random_pos = self.nucleus_center + random_offset
            if self._is_within_nucleus(random_pos):
                return random_pos
        return self.nucleus_center.copy()
    
    def _is_within_nucleus(self, pos):
        """Check if position is within nucleus (half-ellipsoid)."""
        dx = pos[0] - self._nucleus_center[0]
        dy = pos[1] - self._nucleus_center[1]
        dz = pos[2] - self._nucleus_center[2]
        rx, ry, rz = self._nucleus_radii
        return ((dx/rx)**2 + (dy/ry)**2 + (dz/rz)**2 <= 1) and (pos[2] >= 0)
    
    def _is_within_cytosol(self, pos):
        """Check if position is within cytosol."""
        dx = pos[0] - self._cytosol_center[0]
        dy = pos[1] - self._cytosol_center[1]
        dz = pos[2] - self._cytosol_center[2]
        rx, ry, rz = self._cytosol_radii
        in_cell = ((dx/rx)**2 + (dy/ry)**2 + (dz/rz)**2 <= 1) and (pos[2] >= 0)
        return in_cell and not self._is_within_nucleus(pos)
    
    def _is_near_envelope(self, pos):
        """Check if position is in nuclear envelope transport zone."""
        dx = pos[0] - self._nucleus_center[0]
        dy = pos[1] - self._nucleus_center[1]
        dz = pos[2] - self._nucleus_center[2]
        outer_dist = (dx/self._nucleus_radii[0])**2 + (dy/self._nucleus_radii[1])**2 + (dz/self._nucleus_radii[2])**2
        inner_dist = (dx/self._shrunk_radii[0])**2 + (dy/self._shrunk_radii[1])**2 + (dz/self._shrunk_radii[2])**2
        return (outer_dist <= 1) and (inner_dist > 1)
    
    def _sample_poisson(self, rate):
        """Sample number of events from Poisson distribution."""
        expected = rate * self.dt
        if expected < 0:
            return 0
        return np.random.poisson(expected)
    
    def _step_reactions(self, current_time):
        """Execute all reactions for one time step using tau-leap."""
        rates = self.rates
        
        # TS switching (simplified: flip based on probability)
        if not self.TS_state:
            if np.random.random() < rates['k_on'] * self.dt:
                self.TS_state = True
        else:
            if np.random.random() < rates['k_off'] * self.dt:
                self.TS_state = False
            else:
                # RNA initiation when TS is ON
                n_new_rnas = self._sample_poisson(rates['k_r'])
                for _ in range(n_new_rnas):
                    self._create_nascent_rna(current_time)
        
        # RNA degradation
        rnas_to_delete = []
        for rna_id, rna in self.RNAs.items():
            if np.random.random() < rates['gamma_r'] * self.dt:
                rnas_to_delete.append(rna_id)
        
        # Delete proteins attached to degraded RNAs first
        for rna_id in rnas_to_delete:
            proteins_to_delete = [
                pid for pid, p in self.Proteins.items()
                if p.get('state') == 'nascent' and p.get('parent_rna_id') == rna_id
            ]
            for pid in proteins_to_delete:
                del self.Proteins[pid]
            del self.RNAs[rna_id]
        
        # RNA transport (mature nuclear RNAs near envelope)
        for rna_id, rna in list(self.RNAs.items()):
            if rna.get('state') == 'mature' and not rna['in_cytosol']:
                if self._is_near_envelope(rna['position']):
                    if np.random.random() < rates['transport_rate'] * self.dt:
                        self._transport_rna(rna)
        
        # Protein initiation from mature cytoplasmic RNAs
        for rna_id, rna in self.RNAs.items():
            if rna.get('state') == 'mature' and rna['in_cytosol']:
                n_new_proteins = self._sample_poisson(rates['k_p'])
                for _ in range(n_new_proteins):
                    self._create_nascent_protein(rna_id, current_time)
        
        # Protein degradation
        proteins_to_delete = []
        for prot_id, prot in self.Proteins.items():
            if np.random.random() < rates['gamma_p'] * self.dt:
                proteins_to_delete.append(prot_id)
        for pid in proteins_to_delete:
            del self.Proteins[pid]
    
    def _create_nascent_rna(self, current_time):
        """Create a nascent RNA at the transcription site."""
        self.RNAs[self.next_rna_id] = {
            'id': self.next_rna_id,
            'position': self.transcription_site.tolist(),
            'in_cytosol': False,
            'entity_type': 'RNA',
            'state': 'nascent',
            'completion_time': current_time + self.RNA_elongation_time,
            'time': current_time,
        }
        self.next_rna_id += 1
    
    def _create_nascent_protein(self, parent_rna_id, current_time):
        """Create a nascent protein attached to parent RNA."""
        rna = self.RNAs[parent_rna_id]
        self.Proteins[self.next_protein_id] = {
            'id': self.next_protein_id,
            'position': list(rna['position']),
            'in_cytosol': True,
            'entity_type': 'Protein',
            'state': 'nascent',
            'parent_rna_id': parent_rna_id,
            'completion_time': current_time + self.Protein_elongation_time,
            'time': current_time,
        }
        self.next_protein_id += 1
    
    def _transport_rna(self, rna):
        """Transport RNA from nucleus to cytosol."""
        rna['in_cytosol'] = True
        nucleus_radii = self._nucleus_radii  # [rx, ry, rz]
        
        # Current position relative to nucleus center
        pos = np.array(rna['position'])
        rel_pos = pos - self.nucleus_center
        
        # Normalize by radii to get direction in "spherical" space
        normalized_dir = rel_pos / nucleus_radii
        norm = np.linalg.norm(normalized_dir)
        
        if norm > 1e-10:
            # Scale to unit sphere, then scale back to ellipsoid surface
            normalized_dir /= norm
            # Point on ellipsoid surface: center + radii * normalized_direction
            surface_point = self.nucleus_center + nucleus_radii * normalized_dir
        else:
            # RNA is at nucleus center - place at top of nucleus
            surface_point = self.nucleus_center + np.array([0.0, 0.0, nucleus_radii[2]])
            normalized_dir = np.array([0.0, 0.0, 1.0])
        
        # Place RNA just outside the surface (in the outward direction)
        outward_offset = normalized_dir * self.small_distance_outside_nucleus
        new_pos = surface_point + outward_offset
        
        # For half-ellipsoid: ensure Z >= 0 (stay on coverslip side)
        if new_pos[2] < 0:
            new_pos[2] = self.small_distance_outside_nucleus
        
        rna['position'] = new_pos.tolist()
    
    def _step_diffusion(self):
        """Batch diffusion update for all molecules."""
        k_diff_r = self.rates['k_diff_r']
        k_diff_p = self.rates['k_diff_p']
        
        # RNA diffusion (only mature)
        for rna in self.RNAs.values():
            if rna.get('state') != 'mature':
                continue
            # Physical diffusion D requires variance = 2 * D * dt
            displacement = np.random.normal(scale=np.sqrt(2 * k_diff_r * self.dt), size=3)
            new_pos = np.array(rna['position']) + displacement
            
            if rna['in_cytosol']:
                if self._is_within_cytosol(new_pos):
                    rna['position'] = new_pos.tolist()
            else:
                if self._is_within_nucleus(new_pos):
                    rna['position'] = new_pos.tolist()
        
        # Protein diffusion (only mature)
        for prot in self.Proteins.values():
            if prot.get('state') != 'mature':
                continue
            # Physical diffusion D requires variance = 2 * D * dt
            displacement = np.random.normal(scale=np.sqrt(2 * k_diff_p * self.dt), size=3)
            new_pos = np.array(prot['position']) + displacement
            
            if self.movement_protein_into_nucleus:
                if self._is_within_cytosol(new_pos) or self._is_within_nucleus(new_pos):
                    prot['position'] = new_pos.tolist()
            else:
                if self._is_within_cytosol(new_pos):
                    prot['position'] = new_pos.tolist()
    
    def _check_maturations(self, current_time):
        """Check if nascent molecules should mature."""
        # RNAs
        for rna in self.RNAs.values():
            if rna.get('state') == 'nascent' and current_time >= rna.get('completion_time', 0):
                rna['state'] = 'mature'
                rna.pop('completion_time', None)
        
        # Proteins
        proteins_to_delete = []
        for prot_id, prot in self.Proteins.items():
            if prot.get('state') == 'nascent':
                parent_id = prot.get('parent_rna_id')
                if parent_id not in self.RNAs:
                    proteins_to_delete.append(prot_id)
                else:
                    # Sync position with parent
                    prot['position'] = list(self.RNAs[parent_id]['position'])
                    if current_time >= prot.get('completion_time', 0):
                        prot['state'] = 'mature'
                        prot.pop('completion_time', None)
                        prot.pop('parent_rna_id', None)
        
        for pid in proteins_to_delete:
            del self.Proteins[pid]
    
    def _save_state(self, current_time):
        """Save current state (same format as full model)."""
        adjusted_time = round(current_time - self.burnin_time, 2)
        self.time_steps.append(adjusted_time)
        
        for rna_id, rna in self.RNAs.items():
            if rna_id not in self.RNA_trajectories:
                self.RNA_trajectories[rna_id] = []
            snapshot = rna.copy()
            snapshot['time'] = adjusted_time
            snapshot['id'] = rna_id
            self.RNA_trajectories[rna_id].append(snapshot)
        
        for prot_id, prot in self.Proteins.items():
            if prot_id not in self.Protein_trajectories:
                self.Protein_trajectories[prot_id] = []
            snapshot = prot.copy()
            snapshot['time'] = adjusted_time
            snapshot['id'] = prot_id
            self.Protein_trajectories[prot_id].append(snapshot)
        
        self.TS_trajectory.append({
            'position': self.transcription_site.tolist(),
            'state': self.TS_state,
            'time': adjusted_time
        })
    
    def _generate_masks(self):
        """Generate nucleus and cytosol masks."""
        x, y, z = np.meshgrid(
            np.linspace(0, self.simulation_volume_size[0] - 1, self.simulation_volume_size[0]),
            np.linspace(0, self.simulation_volume_size[1] - 1, self.simulation_volume_size[1]),
            np.linspace(0, self.simulation_volume_size[2] - 1, self.simulation_volume_size[2]),
            indexing='ij')
        
        positions = np.stack((x, y, z), axis=-1)
        
        # Cytosol half-ellipsoid
        norm_dist_cyto = np.sum(((positions - self._cytosol_center) / self._cytosol_radii)**2, axis=-1)
        self.cytosol_mask = (norm_dist_cyto <= 1) & (z >= 0)
        
        # Nucleus half-ellipsoid
        norm_dist_nuc = np.sum(((positions - self._nucleus_center) / self._nucleus_radii)**2, axis=-1)
        self.nucleus_mask = (norm_dist_nuc <= 1) & (z >= 0)
        
        # Remove nucleus from cytosol
        self.cytosol_mask &= ~self.nucleus_mask
    
    def simulate(self):
        """Run tau-leap simulation."""
        current_time = 0.0
        step_count = 0
        
        while current_time < self.total_time:
            # Drug application
            if self.apply_drug and current_time >= self.drug_application_time and not self.parameters_updated:
                for param, value in self.inhibited_parameters.items():
                    if param in self.rates:
                        self.rates[param] = value
                self.parameters_updated = True
            
            # Execute reactions
            self._step_reactions(current_time)
            
            # Check maturations
            self._check_maturations(current_time)
            
            # Diffusion (batch)
            self._step_diffusion()
            
            # Save state at frame rate
            if current_time >= self.burnin_time and current_time >= self.next_save_time:
                self._save_state(current_time)
                self.next_save_time += self.frame_rate
            
            current_time += self.dt
            step_count += 1
            
            # Progress indicator (every ~10% of burnin or total time)
            total = self.total_time
            if step_count % int(total / self.dt / 10) == 0:
                pct = min(100, int(100 * current_time / total))
                print(f"  Progress: {pct}% (t={current_time:.1f}s, molecules: {len(self.RNAs)} RNAs, {len(self.Proteins)} proteins)")
    
    def run(self):
        """Run simulation and return results."""
        self.simulate()
        self._generate_masks()
        
        return {
            'RNA_trajectories': self.RNA_trajectories,
            'Protein_trajectories': self.Protein_trajectories,
            'TS_trajectory': self.TS_trajectory,
            'nucleus_mask': self.nucleus_mask,
            'cytosol_mask': self.cytosol_mask,
            'time_steps': self.time_steps,
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified 3D Gene Expression Simulation (Tau-Leap)')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()
    
    print(f"Loading configuration from {args.config}...")
    try:
        sim_params = SimplifiedSimulationParameters.from_yaml(args.config)
        print("Configuration validated successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Using default parameters instead.")
        sim_params = SimplifiedSimulationParameters()
    except ValueError as e:
        print(f"Configuration error:\n{e}")
        exit(1)
    
    params = sim_params.to_dict()
    
    print(f"Starting SIMPLIFIED simulation (dt={params['dt']}s)...")
    print(f"  Total time: {params['total_time']}s, Burnin: {params['burnin_time']}s")
    
    simulator = SimplifiedGeneExpressionSimulator(params)
    results = simulator.run()
    print("Simulation finished.")
    
    print(f"  Final counts: {len(simulator.RNAs)} RNAs, {len(simulator.Proteins)} proteins")
    print(f"  Saved {len(results['time_steps'])} frames")
    
    print("Plotting molecule concentrations...")
    plot_molecule_concentrations(results, output_filename='concentration_plot.png',
                                  show_nascent_separately=params.get('show_nascent_separately', True))
    
    print("Plotting all projections (XY, XZ, YZ)...")
    plot_all_projections(results, params['simulation_volume_size'], 
                         results['nucleus_mask'], results['cytosol_mask'],
                         transcription_site=simulator.transcription_site, 
                         output_filename='projections.png')
    
    if params.get('generate_gif', False):
        print("Generating temporal evolution GIF...")
        generate_temporal_gif(
            results, params['simulation_volume_size'],
            results['nucleus_mask'], results['cytosol_mask'],
            transcription_site=simulator.transcription_site,
            output_filename='simulation.gif',
            fps=params.get('gif_fps', 5),
            skip_frames=params.get('gif_skip_frames', 1),
            dpi=params.get('gif_dpi', 80),
            show_surfaces=params.get('gif_show_surfaces', True),
            surface_decimation=params.get('gif_surface_decimation', 4)
        )
    
    print("Done!")
