
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml
from pathlib import Path
# Import plotting functions from separate module
from plotting import plot_molecule_concentrations, plot_all_projections, generate_temporal_gif


@dataclass
class SimulationParameters:
    """
    Parameters for 3D gene expression simulation.
    
    All parameters are validated on initialization. Invalid parameters
    will raise a ValueError with a descriptive message.
    
    Attributes
    ----------
    k_on : float
        Transcription site activation rate (1/s)
    k_off : float
        Transcription site deactivation rate (1/s)
    k_r : float
        RNA production rate when TS is ON (1/s)
    gamma_r : float
        RNA degradation rate (1/s)
    k_p : float
        Protein production rate per cytoplasmic RNA (1/s)
    gamma_p : float
        Protein degradation rate (1/s)
    k_diff_r : float
        RNA diffusion coefficient (px²/s)
    k_diff_p : float
        Protein diffusion coefficient (px²/s)
    transport_rate : float
        Nuclear export rate for RNA near envelope (1/s)
    simulation_volume_size : List[int]
        Simulation box dimensions [X, Y, Z] in pixels
    cytosol_size : List[int]
        Cytosol ellipsoid diameters [X, Y, Z] in pixels
    nucleus_size : List[int]
        Nucleus ellipsoid diameters [X, Y, Z] in pixels
    nucleus_xy_offset : List[int]
        Nucleus center offset from box center [X, Y] in pixels
    total_time : float
        Recorded simulation time in seconds (appears in output plots)
    frame_rate : float
        State save interval in seconds
    burnin_time : float
        Equilibration time before recording starts (added to total_time internally)
    position_TS : str
        Transcription site position: 'center' or 'random'
    movement_protein_into_nucleus : bool
        Whether proteins can enter the nucleus
    apply_drug : bool
        Whether to apply drug perturbation
    drug_application_time : float
        Time to apply drug in seconds
    inhibited_parameters : dict
        Parameters to change after drug application
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
    RNA_elongation_time: float = 60.0       # Seconds to synthesize one RNA
    Protein_elongation_time: float = 30.0   # Seconds to synthesize one protein
    
    # Geometry
    simulation_volume_size: List[int] = field(default_factory=lambda: [512, 512, 100])
    cytosol_size: List[int] = field(default_factory=lambda: [350, 350, 80])
    nucleus_size: List[int] = field(default_factory=lambda: [120, 100, 60])
    nucleus_xy_offset: List[int] = field(default_factory=lambda: [0, 0])
    
    # Simulation settings
    total_time: float = 100.0
    frame_rate: float = 1.0
    burnin_time: float = 0.0
    random_seed: int = None  # Random seed for reproducibility (None = random)
    
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
    gif_skip_frames: int = 1              # Render every Nth frame
    gif_dpi: int = 80                     # DPI for GIF frames
    gif_show_surfaces: bool = True        # Show cell surfaces in GIF (slower)
    gif_surface_decimation: int = 4       # Downsample surfaces by this factor
    show_nascent_separately: bool = True  # Show nascent molecules with different colors
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate()
    
    def _validate(self):
        """Check all parameters are valid."""
        errors = []
        
        # Check positive rates
        rate_params = ['k_on', 'k_off', 'k_r', 'gamma_r', 'k_p', 'gamma_p', 
                       'k_diff_r', 'k_diff_p', 'transport_rate']
        for rate_name in rate_params:
            value = getattr(self, rate_name)
            if not isinstance(value, (int, float)):
                errors.append(f"{rate_name} must be a number, got {type(value).__name__}")
            elif value < 0:
                errors.append(f"{rate_name} must be non-negative, got {value}")
        
        # Check geometry dimensions
        geom_params = ['simulation_volume_size', 'cytosol_size', 'nucleus_size']
        for geom_name in geom_params:
            geom = getattr(self, geom_name)
            if not isinstance(geom, (list, tuple)):
                errors.append(f"{geom_name} must be a list, got {type(geom).__name__}")
            elif len(geom) != 3:
                errors.append(f"{geom_name} must have 3 elements [X, Y, Z], got {len(geom)}")
            elif any(not isinstance(v, (int, float)) or v <= 0 for v in geom):
                errors.append(f"{geom_name} values must be positive numbers, got {geom}")
        
        # Check nucleus_xy_offset
        if not isinstance(self.nucleus_xy_offset, (list, tuple)):
            errors.append(f"nucleus_xy_offset must be a list, got {type(self.nucleus_xy_offset).__name__}")
        elif len(self.nucleus_xy_offset) != 2:
            errors.append(f"nucleus_xy_offset must have 2 elements [X, Y], got {len(self.nucleus_xy_offset)}")
        
        # Check nucleus fits inside cytosol
        if not errors:  # Only check if geometry is valid
            for i, (n, c) in enumerate(zip(self.nucleus_size, self.cytosol_size)):
                if n > c:
                    axis = ['X', 'Y', 'Z'][i]
                    errors.append(f"nucleus_size {axis}={n} exceeds cytosol_size {axis}={c}")
            
            # Check cytosol fits inside simulation volume
            for i, (c, v) in enumerate(zip(self.cytosol_size, self.simulation_volume_size)):
                if c > v:
                    axis = ['X', 'Y', 'Z'][i]
                    errors.append(f"cytosol_size {axis}={c} exceeds simulation_volume_size {axis}={v}")
        
        # Check time parameters
        if self.total_time <= 0:
            errors.append(f"total_time must be positive, got {self.total_time}")
        if self.frame_rate <= 0:
            errors.append(f"frame_rate must be positive, got {self.frame_rate}")
        if self.burnin_time < 0:
            errors.append(f"burnin_time must be non-negative, got {self.burnin_time}")
        
        # Check elongation time parameters
        if self.RNA_elongation_time < 0:
            errors.append(f"RNA_elongation_time must be non-negative, got {self.RNA_elongation_time}")
        if self.Protein_elongation_time < 0:
            errors.append(f"Protein_elongation_time must be non-negative, got {self.Protein_elongation_time}")
        
        # Check position_TS
        if self.position_TS not in ['center', 'random']:
            errors.append(f"position_TS must be 'center' or 'random', got '{self.position_TS}'")
        
        # Check drug parameters
        if self.apply_drug and self.drug_application_time < 0:
            errors.append(f"drug_application_time must be non-negative, got {self.drug_application_time}")
        
        if errors:
            raise ValueError("Invalid simulation parameters:\n  - " + "\n  - ".join(errors))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility with simulator."""
        return {
            'k_on': self.k_on,
            'k_off': self.k_off,
            'k_r': self.k_r,
            'gamma_r': self.gamma_r,
            'k_p': self.k_p,
            'gamma_p': self.gamma_p,
            'k_diff_r': self.k_diff_r,
            'k_diff_p': self.k_diff_p,
            'transport_rate': self.transport_rate,
            'RNA_elongation_time': self.RNA_elongation_time,
            'Protein_elongation_time': self.Protein_elongation_time,
            'simulation_volume_size': list(self.simulation_volume_size),
            'cytosol_size': list(self.cytosol_size),
            'nucleus_size': list(self.nucleus_size),
            'nucleus_xy_offset': list(self.nucleus_xy_offset),
            'total_time': self.total_time,
            'frame_rate': self.frame_rate,
            'burnin_time': self.burnin_time,
            'random_seed': self.random_seed,
            'position_TS': self.position_TS,
            'movement_protein_into_nucleus': self.movement_protein_into_nucleus,
            'apply_drug': self.apply_drug,
            'drug_application_time': self.drug_application_time,
            'inhibited_parameters': dict(self.inhibited_parameters),
            'generate_gif': self.generate_gif,
            'gif_fps': self.gif_fps,
            'gif_skip_frames': self.gif_skip_frames,
            'gif_dpi': self.gif_dpi,
            'gif_show_surfaces': self.gif_show_surfaces,
            'gif_surface_decimation': self.gif_surface_decimation,
            'show_nascent_separately': self.show_nascent_separately,
        }
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'SimulationParameters':
        """Load parameters from a YAML configuration file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Flatten nested YAML structure
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
            if 'generate_gif' in config['output']:
                params['generate_gif'] = config['output']['generate_gif']
            if 'gif_fps' in config['output']:
                params['gif_fps'] = config['output']['gif_fps']
            if 'gif_skip_frames' in config['output']:
                params['gif_skip_frames'] = config['output']['gif_skip_frames']
            if 'gif_dpi' in config['output']:
                params['gif_dpi'] = config['output']['gif_dpi']
            if 'gif_show_surfaces' in config['output']:
                params['gif_show_surfaces'] = config['output']['gif_show_surfaces']
            if 'gif_surface_decimation' in config['output']:
                params['gif_surface_decimation'] = config['output']['gif_surface_decimation']
            if 'show_nascent_separately' in config['output']:
                params['show_nascent_separately'] = config['output']['show_nascent_separately']
        
        return cls(**params)



class GeneExpressionSimulator:
    def __init__(self, params):
        # Set random seed for reproducibility (if provided)
        random_seed = params.get('random_seed', None)
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.TS_state = False  # Initial state of the Transcription Site (TS)
        self.RNAs = {}
        self.Proteins = {}
        self.next_rna_id = 1
        self.next_protein_id = 1
        
        # 3D simulation volume
        simulation_volume_size = params['simulation_volume_size'][:3]
        center_of_box = np.array(simulation_volume_size) / 2
        self.center_of_box = center_of_box
        self.simulation_volume_size = simulation_volume_size
        
        # Nucleus positioning for half-ellipsoid cell morphology
        # Half-ellipsoid: nucleus sits on Z=0 (coverslip), center is at Z=0
        nucleus_xy_offset = params.get('nucleus_xy_offset', [0, 0])
        
        self.nucleus_mask = np.zeros(simulation_volume_size, dtype=bool)
        self.cytosol_mask = np.zeros(simulation_volume_size, dtype=bool)
        self.nucleus_size = params['nucleus_size']
        
        # Set rates early so is_within_nucleus can use them
        self.rates = params.copy()
        
        # Initialize half-ellipsoid attributes for boundary checking (MUST be before TS placement)
        cytosol_radii = np.array(params['cytosol_size']) / 2
        nucleus_radii = np.array(params['nucleus_size']) / 2
        
        # Cytosol: half-ellipsoid centered at Z=0
        self._cytosol_radii = cytosol_radii
        self._cytosol_center = np.array([center_of_box[0], center_of_box[1], 0])
        
        # Nucleus: half-ellipsoid centered at Z=0, possibly offset in XY
        self._nucleus_radii = nucleus_radii
        self._nucleus_center = np.array([
            center_of_box[0] + nucleus_xy_offset[0],
            center_of_box[1] + nucleus_xy_offset[1],
            0
        ])
        self.nucleus_center = self._nucleus_center
        
        # Transport zone threshold and shrunk radii (for performance)
        self.transport_zone_threshold = 2
        shrunk_size = np.array(params['nucleus_size']) - 2 * self.transport_zone_threshold
        shrunk_size = np.clip(shrunk_size, a_min=self.transport_zone_threshold, a_max=None)
        self._shrunk_radii = shrunk_size / 2
        
        # Now we can place the transcription site
        if params['position_TS'] == 'center':
            # Place TS at geometric center of the nucleus dome (halfway up in Z)
            nucleus_radius_z = params['nucleus_size'][2] / 2
            self.transcription_site = np.array([
                self.nucleus_center[0],
                self.nucleus_center[1],
                nucleus_radius_z / 2  # Center of the dome, not at Z=0
            ])
        else:
            self.transcription_site = self.find_random_TS_position_inside_nucleus()
            
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
        
        # Elongation times (deterministic)
        self.RNA_elongation_time = params.get('RNA_elongation_time', 60.0)
        self.Protein_elongation_time = params.get('Protein_elongation_time', 30.0)
        
    def update_rates_with_drug_effect(self):
        for param, value in self.inhibited_parameters.items():
            if param in self.rates:
                self.rates[param] = value
    
    def find_random_TS_position_inside_nucleus(self):
        """Find random position inside nucleus (half-ellipsoid).
        
        Samples within nucleus bounding box for efficiency.
        """
        nucleus_radii = np.array(self.rates['nucleus_size']) / 2
        nucleus_center = self.nucleus_center
        
        max_attempts = 10000
        for _ in range(max_attempts):
            # Sample within nucleus bounding box
            # For half-ellipsoid: Z from 0 to nucleus_radius_z
            random_offset = np.array([
                np.random.uniform(-nucleus_radii[0], nucleus_radii[0]),
                np.random.uniform(-nucleus_radii[1], nucleus_radii[1]),
                np.random.uniform(0, nucleus_radii[2])  # Only upper half
            ])
            random_pos = nucleus_center + random_offset
            
            # Check if inside nucleus ellipsoid
            temp_entity = {'position': random_pos.tolist()}
            if self.is_within_nucleus(entity=temp_entity):
                return random_pos
        
        # Fallback: return nucleus center
        print("Warning: Could not find random TS position, using nucleus center")
        return nucleus_center.copy()
        
    def save_state(self, current_time):
        # Adjust time to start from 0 after burnin, round to 2 decimal places for precision
        adjusted_time = round(current_time - self.burnin_time, 2)
        self.time_steps.append(adjusted_time)
        
        for rna_id, rna_info in self.RNAs.items():
            if rna_id not in self.RNA_trajectories:
                self.RNA_trajectories[rna_id] = []
            rna_snapshot = rna_info.copy()
            rna_snapshot['time'] = adjusted_time
            rna_snapshot['id'] = rna_id
            # Position is already in rna_info, no need to copy again
            self.RNA_trajectories[rna_id].append(rna_snapshot)
        
        for protein_id, protein_info in self.Proteins.items():
            if protein_id not in self.Protein_trajectories:
                self.Protein_trajectories[protein_id] = []
            protein_snapshot = protein_info.copy()
            protein_snapshot['time'] = adjusted_time
            protein_snapshot['id'] = protein_id
            # Position is already in protein_info, no need to copy again
            self.Protein_trajectories[protein_id].append(protein_snapshot)
        
        # For the transcription site, if it also has an 'id', include it as well
        TS_info = {
            'position': self.transcription_site.tolist(),
            'state': self.TS_state,
            'time': adjusted_time
        }
        self.TS_trajectory.append(TS_info)
    
    def generate_masks(self):
        """
        Generate nucleus and cytosol masks.
        
        Shape: Half-ellipsoid (dome sitting on coverslip at Z=0)
        - Cytosol: Upper half of ellipsoid, center at Z=0
        - Nucleus: Upper half of ellipsoid inside cytosol, center at Z=0
        """
        # 3D: Half-ellipsoid shapes (domes sitting on coverslip at Z=0)
        x, y, z = np.meshgrid(
            np.linspace(0, self.simulation_volume_size[0] - 1, self.simulation_volume_size[0]),
            np.linspace(0, self.simulation_volume_size[1] - 1, self.simulation_volume_size[1]),
            np.linspace(0, self.simulation_volume_size[2] - 1, self.simulation_volume_size[2]),
            indexing='ij')
        
        positions = np.stack((x, y, z), axis=-1)
        
        # Cytosol: half-ellipsoid (dome) sitting on Z=0 (use pre-computed values)
        normalized_sq_dist_cytosol = np.sum(((positions - self._cytosol_center) / self._cytosol_radii)**2, axis=-1)
        self.cytosol_mask = (normalized_sq_dist_cytosol <= 1) & (z >= 0)
        
        # Nucleus: half-ellipsoid sitting on Z=0 (use pre-computed values)
        normalized_sq_dist_nucleus = np.sum(((positions - self._nucleus_center) / self._nucleus_radii)**2, axis=-1)
        self.nucleus_mask = (normalized_sq_dist_nucleus <= 1) & (z >= 0)
        
        # Remove nucleus from cytosol
        self.cytosol_mask &= ~self.nucleus_mask

    def is_within_nucleus(self, entity=None, pos=None, nucleus_diameter=None):
        """Check if position is within nucleus (half-ellipsoid).
        
        Optimized: Uses pure Python math for single point operations.
        """
        if entity is not None:
            p = entity['position']
            rx, ry, rz = self._nucleus_radii
        else:
            p = pos
            rx, ry, rz = nucleus_diameter[0]/2, nucleus_diameter[1]/2, nucleus_diameter[2]/2
        
        cx, cy, cz = self._nucleus_center
        dx, dy, dz = p[0] - cx, p[1] - cy, p[2] - cz
        normalized_sq_dist = (dx/rx)**2 + (dy/ry)**2 + (dz/rz)**2
        # Half-ellipsoid: require Z >= 0
        return (normalized_sq_dist <= 1) and (p[2] >= 0)
    
    def is_within_cytosol(self, entity):
        """Check if entity is within cytosol (half-ellipsoid).
        
        Optimized: Uses pure Python math for single point operations.
        """
        p = entity['position']
        cx, cy, cz = self._cytosol_center
        rx, ry, rz = self._cytosol_radii
        dx, dy, dz = p[0] - cx, p[1] - cy, p[2] - cz
        normalized_sq_dist = (dx/rx)**2 + (dy/ry)**2 + (dz/rz)**2
        is_in_cytosol = (normalized_sq_dist <= 1) and (p[2] >= 0)
        return is_in_cytosol and not self.is_within_nucleus(entity)
    
    def move_particle(self, entity, rate):
        """Move particle with Brownian motion, confined to appropriate region."""
        displacement = np.random.normal(scale=np.sqrt(rate), size=3)
        current_position = np.array(entity['position'][:3])
        new_position = current_position + displacement
        temp_entity = {'position': new_position.tolist()}
        
        if entity['entity_type'] == 'RNA':
            # RNA in cytosol: stay in cytosol
            if entity['in_cytosol'] and self.is_within_cytosol(temp_entity):
                entity['position'] = new_position.tolist()
            # RNA in nucleus: stay in nucleus
            elif not entity['in_cytosol'] and self.is_within_nucleus(temp_entity):
                entity['position'] = new_position.tolist()
        elif entity['entity_type'] == 'Protein':
            if self.movement_protein_into_nucleus:
                # Allow movement in both nucleus and cytosol
                if self.is_within_cytosol(temp_entity) or self.is_within_nucleus(temp_entity):
                    entity['position'] = new_position.tolist()
            else:
                # Restrict to cytosol only
                if self.is_within_cytosol(temp_entity):
                    entity['position'] = new_position.tolist()
        return entity
    
    def is_near_nuclear_envelope(self, rna_position):
        """Check if RNA is near nuclear envelope (transport zone).
        
        Optimized: Uses pre-computed values and pure Python math.
        """
        # Use pre-computed values from __init__
        pos = rna_position if isinstance(rna_position, (list, tuple)) else rna_position[:3]
        
        # Calculate normalized squared distance for outer boundary
        dx = pos[0] - self._nucleus_center[0]
        dy = pos[1] - self._nucleus_center[1]
        dz = pos[2] - self._nucleus_center[2]
        
        # Outer boundary check (within nucleus)
        outer_dist_sq = (dx / self._nucleus_radii[0])**2 + \
                        (dy / self._nucleus_radii[1])**2 + \
                        (dz / self._nucleus_radii[2])**2
        within_nucleus = outer_dist_sq <= 1
        
        # Inner boundary check (outside shrunk nucleus = in transport zone)
        inner_dist_sq = (dx / self._shrunk_radii[0])**2 + \
                        (dy / self._shrunk_radii[1])**2 + \
                        (dz / self._shrunk_radii[2])**2
        in_transport_zone = inner_dist_sq > 1
        
        return within_nucleus and in_transport_zone

    def calculate_rates_and_reactions(self):
        """Calculate reaction rates based on current state.
        
        Key changes for elongation model:
        - Nascent RNAs cannot move, transport, or initiate translation
        - Only mature cytoplasmic RNAs can initiate protein translation
        - Nascent proteins cannot move (attached to parent RNA)
        """
        rates = []
        reactions = []
        
        # Transcription site switching
        if not self.TS_state:
            rates.append(self.rates['k_on'])
            reactions.append(('TS_on', None))
        else:
            rates.append(self.rates['k_off'])
            reactions.append(('TS_off', None))
            # RNA initiation (produces nascent RNA)
            rates.append(self.rates['k_r'])
            reactions.append(('initiate_RNA', None))
        
        # RNA reactions
        for rna_id, rna_info in self.RNAs.items():
            # Degradation applies to all RNAs
            rates.append(self.rates['gamma_r'])
            reactions.append(('degrade_RNA', rna_id))
            
            # Only MATURE RNAs can move
            if rna_info.get('state', 'mature') == 'mature':
                rates.append(self.rates['k_diff_r'])
                reactions.append(('move_RNA', rna_id))
                
                # Only MATURE cytoplasmic RNAs can initiate translation
                if rna_info['in_cytosol']:
                    rates.append(self.rates['k_p'])
                    reactions.append(('initiate_Protein', rna_id))
                
                # Only MATURE nuclear RNAs can be transported
                if not rna_info['in_cytosol'] and self.is_near_nuclear_envelope(rna_info['position']):
                    rates.append(self.rates['transport_rate'])
                    reactions.append(('transport_RNA_to_cytosol', rna_id))
        
        # Protein reactions
        for protein_id, protein_info in self.Proteins.items():
            # Degradation applies to all proteins
            rates.append(self.rates['gamma_p'])
            reactions.append(('degrade_Protein', protein_id))
            
            # Only MATURE proteins can move independently
            if protein_info.get('state', 'mature') == 'mature':
                rates.append(self.rates['k_diff_p'])
                reactions.append(('move_Protein', protein_id))
        
        return rates, reactions

    def execute_reaction(self, reaction, current_time):
        """Execute a single reaction.
        
        Elongation model changes:
        - initiate_RNA: Creates NASCENT RNA with completion_time
        - initiate_Protein: Creates NASCENT protein with parent_rna_id and completion_time
        - degrade_RNA: Also degrades any attached nascent proteins
        """
        reaction_type, entity_id = reaction
        
        if reaction_type == 'TS_off':
            self.TS_state = False
        elif reaction_type == 'TS_on':
            self.TS_state = True
        
        elif reaction_type == 'initiate_RNA':
            # Create NASCENT RNA at transcription site (cannot move until mature)
            new_rna = {
                'id': self.next_rna_id,
                'position': self.transcription_site.tolist(),
                'in_cytosol': False,
                'entity_type': 'RNA',
                'state': 'nascent',  # NEW: nascent state
                'completion_time': current_time + self.RNA_elongation_time,  # NEW: when it matures
                'time': current_time,
            }
            self.RNAs[self.next_rna_id] = new_rna
            self.next_rna_id += 1
            
        elif reaction_type == 'degrade_RNA' and entity_id in self.RNAs:
            # Before deleting RNA, also delete any nascent proteins attached to it
            proteins_to_delete = [
                pid for pid, pinfo in self.Proteins.items()
                if pinfo.get('state') == 'nascent' and pinfo.get('parent_rna_id') == entity_id
            ]
            for pid in proteins_to_delete:
                del self.Proteins[pid]
            # Now delete the RNA
            del self.RNAs[entity_id]
            
        elif reaction_type == 'move_RNA':
            if entity_id in self.RNAs:
                rna_info = self.RNAs[entity_id]
                # Only move mature RNAs (double check, should already be filtered)
                if rna_info.get('state', 'mature') == 'mature':
                    self.move_particle(rna_info, self.rates['k_diff_r'])
                    
        elif reaction_type == 'transport_RNA_to_cytosol':
            if entity_id in self.RNAs:
                rna_info = self.RNAs[entity_id]
                rna_info['in_cytosol'] = True
                
                nucleus_radius = np.array(self.rates['nucleus_size']) / 2
                nucleus_center = self.nucleus_center
                
                # Direction vector points outward from nucleus center
                direction_vector = np.array(rna_info['position'][:3]) - nucleus_center
                norm = np.linalg.norm(direction_vector)
                if norm > 1e-10:  # Avoid division by zero
                    direction_vector /= norm
                else:
                    # RNA is at nucleus center - default to upward direction
                    direction_vector = np.array([0.0, 0.0, 1.0])
                
                # Calculate new position just outside the nucleus
                new_position = nucleus_center + direction_vector * (nucleus_radius + self.small_distance_outside_nucleus)
                
                # For half-ellipsoid: ensure Z >= 0 (stay on coverslip side)
                if new_position[2] < 0:
                    xy_dist_sq = (new_position[0] - nucleus_center[0])**2 / nucleus_radius[0]**2 + \
                                 (new_position[1] - nucleus_center[1])**2 / nucleus_radius[1]**2
                    if xy_dist_sq < 1:
                        # Inside XY footprint - place just above nucleus top
                        z_top = nucleus_center[2] + nucleus_radius[2] * np.sqrt(max(0, 1 - xy_dist_sq))
                        new_position[2] = z_top + self.small_distance_outside_nucleus
                    else:
                        # Outside XY footprint - set Z to small positive value
                        new_position[2] = self.small_distance_outside_nucleus
                
                rna_info['position'] = new_position.tolist()
                
        elif reaction_type == 'initiate_Protein' and entity_id in self.RNAs:
            # Create NASCENT protein attached to parent RNA
            rna_info = self.RNAs[entity_id]
            if rna_info['in_cytosol'] and rna_info.get('state', 'mature') == 'mature':
                new_protein = {
                    'id': self.next_protein_id,
                    'position': rna_info['position'].copy() if isinstance(rna_info['position'], list) else rna_info['position'],
                    'in_cytosol': True,
                    'entity_type': 'Protein',
                    'state': 'nascent',  # NEW: nascent state
                    'parent_rna_id': entity_id,  # NEW: track parent RNA
                    'completion_time': current_time + self.Protein_elongation_time,  # NEW: when it matures
                    'time': current_time,
                }
                self.Proteins[self.next_protein_id] = new_protein
                self.next_protein_id += 1
                
        elif reaction_type == 'degrade_Protein' and entity_id in self.Proteins:
            del self.Proteins[entity_id]
            
        elif reaction_type == 'move_Protein' and entity_id in self.Proteins:
            protein_info = self.Proteins[entity_id]
            # Only move mature proteins (double check, should already be filtered)
            if protein_info.get('state', 'mature') == 'mature':
                self.move_particle(protein_info, self.rates['k_diff_p'])
    
    def check_maturations(self, current_time):
        """Check if any nascent molecules should mature (deterministic elongation).
        
        Also synchronizes nascent protein positions with their parent RNA.
        """
        # Check nascent RNAs
        for rna_id, rna_info in self.RNAs.items():
            if rna_info.get('state') == 'nascent':
                if current_time >= rna_info.get('completion_time', 0):
                    rna_info['state'] = 'mature'
                    # Remove completion_time field (no longer needed)
                    if 'completion_time' in rna_info:
                        del rna_info['completion_time']
        
        # Check nascent Proteins and sync their positions
        proteins_to_delete = []
        for protein_id, protein_info in self.Proteins.items():
            if protein_info.get('state') == 'nascent':
                parent_rna_id = protein_info.get('parent_rna_id')
                
                if parent_rna_id not in self.RNAs:
                    # Parent RNA was degraded - degrade protein too
                    proteins_to_delete.append(protein_id)
                else:
                    # Sync position with parent RNA
                    parent_rna = self.RNAs[parent_rna_id]
                    protein_info['position'] = parent_rna['position'].copy() if isinstance(parent_rna['position'], list) else list(parent_rna['position'])
                    
                    # Check if protein should mature
                    if current_time >= protein_info.get('completion_time', 0):
                        protein_info['state'] = 'mature'
                        # Remove tracking fields (no longer needed)
                        if 'completion_time' in protein_info:
                            del protein_info['completion_time']
                        if 'parent_rna_id' in protein_info:
                            del protein_info['parent_rna_id']
        
        # Delete orphaned proteins
        for pid in proteins_to_delete:
            del self.Proteins[pid]
    

    def simulate(self):
        current_time = 0
        while current_time < self.total_time:
            # Calculate rates and reactions for the current state
            rates, reactions = self.calculate_rates_and_reactions()
            if not rates:
                # If there are no reactions left, advance to the next significant time point (next save time or total time)
                next_time_point = min(self.next_save_time, self.total_time)
                if current_time < next_time_point:
                    current_time = next_time_point
                else:
                    break  # End simulation if beyond total time and no actions are pending
            else:
                # Determine the time until the next reaction occurs
                time_step = np.random.exponential(1 / sum(rates))
                current_time += time_step
                
            # Check for maturation of nascent molecules (deterministic elongation)
            self.check_maturations(current_time)
            
            # Update parameters if the drug application time has been reached and it's not already updated
            if (self.apply_drug==True) and (current_time >= self.drug_application_time) and (self.parameters_updated==False):
                self.update_rates_with_drug_effect()
                self.parameters_updated = True
            # Check if the current time is beyond the burnin time and it's time to save the state
            if current_time >= self.burnin_time and current_time >= self.next_save_time:
                self.save_state(current_time)
                self.next_save_time += self.frame_rate  # Schedule the next state save
            # Execute any reactions that were supposed to happen at this time
            if rates:
                reaction_index = np.random.choice(len(rates), p=np.array(rates) / sum(rates))
                reaction = reactions[reaction_index]
                self.execute_reaction(reaction, current_time)


    def run(self):
        self.simulate()
        self.generate_masks()
        return {
            'RNA_trajectories': self.RNA_trajectories,
            'Protein_trajectories': self.Protein_trajectories,
            'TS_trajectory': self.TS_trajectory,
            'nucleus_mask': self.nucleus_mask,
            'cytosol_mask': self.cytosol_mask,
            'time_steps': self.time_steps,  # Include time_steps in the returned dictionary
        }





if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='3D Gene Expression Simulation')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()
    
    # Load and validate parameters
    print(f"Loading configuration from {args.config}...")
    try:
        sim_params = SimulationParameters.from_yaml(args.config)
        print("Configuration validated successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Using default parameters instead.")
        sim_params = SimulationParameters()
    except ValueError as e:
        print(f"Configuration error:\n{e}")
        exit(1)
    
    # Convert to dict for simulator
    params = sim_params.to_dict()
    
    print("Starting simulation...")
    simulator = GeneExpressionSimulator(params)
    results = simulator.run()
    print("Simulation finished.")
    
    print("Plotting molecule concentrations...")
    plot_molecule_concentrations(results, output_filename='concentration_plot.png',
                                  show_nascent_separately=params.get('show_nascent_separately', True))
    
    print("Plotting all projections (XY, XZ, YZ)...")
    plot_all_projections(results, params['simulation_volume_size'], results['nucleus_mask'], results['cytosol_mask'],
                         transcription_site=simulator.transcription_site, output_filename='projections.png')
    
    # Generate animated GIF if enabled
    if params.get('generate_gif', False):
        print("Generating temporal evolution GIF...")
        generate_temporal_gif(
            results, 
            params['simulation_volume_size'], 
            results['nucleus_mask'], 
            results['cytosol_mask'],
            transcription_site=simulator.transcription_site,
            output_filename='simulation.gif',
            fps=params.get('gif_fps', 5),
            skip_frames=params.get('gif_skip_frames', 1),
            dpi=params.get('gif_dpi', 80),
            show_surfaces=params.get('gif_show_surfaces', True),
            surface_decimation=params.get('gif_surface_decimation', 4)
        )
