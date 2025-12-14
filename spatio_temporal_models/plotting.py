"""
Plotting functions for gene expression simulation visualization.

This module provides functions to visualize simulation results including:
- Molecule concentration time series
- 3D spatial distribution
- 2D orthogonal projections (XY, XZ, YZ)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure


def plot_molecule_concentrations(trajectories, output_filename='concentration_plot.png'):
    """
    Plot RNA and Protein molecule counts over time.
    
    Parameters
    ----------
    trajectories : dict
        Dictionary containing 'RNA_trajectories', 'Protein_trajectories', 
        'TS_trajectory', and 'time_steps'.
    output_filename : str
        Path to save the output plot.
        
    Returns
    -------
    tuple
        (time_steps, rna_counts, protein_counts)
    """
    time_steps = trajectories['time_steps']
    
    # Count molecules at each time step
    rna_counts = []
    protein_counts = []
    ts_states = []
    
    for t in time_steps:
        # Count RNAs at this time step
        rna_count = 0
        for rna_list in trajectories['RNA_trajectories'].values():
            for snapshot in rna_list:
                if snapshot['time'] == t:
                    rna_count += 1
                    break  # Each RNA only counted once per time
        rna_counts.append(rna_count)
        
        # Count Proteins at this time step
        protein_count = 0
        for protein_list in trajectories['Protein_trajectories'].values():
            for snapshot in protein_list:
                if snapshot['time'] == t:
                    protein_count += 1
                    break  # Each Protein only counted once per time
        protein_counts.append(protein_count)
        
        # Get TS state at this time step
        for ts_snapshot in trajectories['TS_trajectory']:
            if ts_snapshot['time'] == t:
                ts_states.append(1 if ts_snapshot['state'] else 0)
                break
    
    # Colors matching the 3D simulation view
    RNA_COLOR = '#EF5350'      # Red
    PROTEIN_COLOR = '#64B5F6'  # Blue
    TS_COLOR = '#4CAF50'       # Green
    GRID_COLOR = '#333333'
    TEXT_COLOR = '#CCCCCC'
    BG_COLOR = 'black'
    
    # Create the plot with unequal height ratios (main plot taller, TS much smaller)
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True, 
                              gridspec_kw={'height_ratios': [7, 1], 'hspace': 0.05},
                              facecolor=BG_COLOR)
    
    # Top panel: RNA and Protein counts
    ax1 = axes[0]
    ax1.set_facecolor(BG_COLOR)
    ax1.plot(time_steps, rna_counts, color=RNA_COLOR, linewidth=2, label='RNA', alpha=0.9)
    ax1.plot(time_steps, protein_counts, color=PROTEIN_COLOR, linewidth=2, label='Protein', alpha=0.9)
    ax1.set_ylabel('Molecule Count', fontsize=11, fontweight='medium', color=TEXT_COLOR)
    ax1.set_title('Molecule Concentrations Over Time', fontsize=13, fontweight='bold', pad=10, color='white')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.8, edgecolor=GRID_COLOR, 
               facecolor=BG_COLOR, labelcolor=TEXT_COLOR)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.3, color=GRID_COLOR)
    ax1.set_xlim([time_steps[0], time_steps[-1]])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(GRID_COLOR)
    ax1.spines['bottom'].set_color(GRID_COLOR)
    ax1.tick_params(axis='both', which='major', labelsize=10, colors=TEXT_COLOR)
    
    # Bottom panel: Transcription Site state (very compact)
    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)
    ax2.fill_between(time_steps, ts_states, step='mid', alpha=0.7, color=TS_COLOR, linewidth=0)
    ax2.set_ylabel('TS', fontsize=9, fontweight='medium', color=TEXT_COLOR)
    ax2.set_xlabel('Time (s)', fontsize=11, fontweight='medium', color=TEXT_COLOR)
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['OFF', 'ON'], fontsize=8, color=TEXT_COLOR)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(GRID_COLOR)
    ax2.spines['bottom'].set_color(GRID_COLOR)
    ax2.tick_params(axis='x', which='major', labelsize=10, colors=TEXT_COLOR)
    ax2.tick_params(axis='y', which='major', labelsize=8, colors=TEXT_COLOR)
    ax2.set_xlim([time_steps[0], time_steps[-1]])
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight', facecolor=BG_COLOR, edgecolor='none')
    print(f"Concentration plot saved to {output_filename}")
    plt.close()
    
    return time_steps, rna_counts, protein_counts


def plot_all_projections(trajectories, simulation_volume_size, masks_nucleus, masks_cytosol, 
                         transcription_site=None, output_filename='projections.png'):
    """
    Plot 3D view and all three 2D max projections (XY, XZ, YZ) in a single figure.
    
    Layout: [3D View] [XY] [XZ] [YZ]
    
    Parameters
    ----------
    trajectories : dict
        Simulation results containing RNA and Protein trajectories
    simulation_volume_size : list
        [X, Y, Z] dimensions of the simulation volume
    masks_nucleus : ndarray
        3D boolean mask for nucleus
    masks_cytosol : ndarray
        3D boolean mask for cytosol
    transcription_site : ndarray, optional
        Position of transcription site [X, Y, Z]
    output_filename : str
        Path to save the output plot
        
    Returns
    -------
    dict
        Dictionary with projection masks for each view
    """
    time_steps = trajectories['time_steps']
    last_time_step = time_steps[-1]
    
    # Extract positions at the last time step
    rna_positions = []
    protein_positions = []
    
    for rna_list in trajectories['RNA_trajectories'].values():
        for snapshot in rna_list:
            if snapshot['time'] == last_time_step:
                rna_positions.append(snapshot['position'])
                
    for protein_list in trajectories['Protein_trajectories'].values():
        for snapshot in protein_list:
            if snapshot['time'] == last_time_step:
                protein_positions.append(snapshot['position'])
    
    rna_pos = np.array(rna_positions) if rna_positions else np.array([]).reshape(0, 3)
    protein_pos = np.array(protein_positions) if protein_positions else np.array([]).reshape(0, 3)
    
    # Create max projections along each axis
    nucleus_xy = np.max(masks_nucleus, axis=2)  # XY: max along Z
    cytosol_xy = np.max(masks_cytosol, axis=2)
    nucleus_xz = np.max(masks_nucleus, axis=1)  # XZ: max along Y
    cytosol_xz = np.max(masks_cytosol, axis=1)
    nucleus_yz = np.max(masks_nucleus, axis=0)  # YZ: max along X
    cytosol_yz = np.max(masks_cytosol, axis=0)
    
    # Create figure with 4 subplots: 3D + 3 projections
    fig = plt.figure(figsize=(20, 5), facecolor='black')
    
    # Grid colors for dark theme
    grid_color = '#333333'
    text_color = '#CCCCCC'
    
    # Helper function to create RGB cell image on black background
    def create_cell_image(nucleus_proj, cytosol_proj):
        img = np.zeros((*nucleus_proj.shape, 3))  # Black background
        # Cytosol: more transparent
        img[cytosol_proj] = [0.08, 0.10, 0.11]  # Very dark, more transparent
        # Nucleus: more transparent
        img[nucleus_proj] = [0.20, 0.08, 0.08]  # Faded coral, more transparent
        return img
    
    # Helper function to style 2D axes
    def style_2d_axis(ax, xlabel, ylabel, title, xlim, ylim, origin_lower=False):
        ax.set_facecolor('black')
        ax.set_xlim(xlim)
        if origin_lower:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim([ylim[1], ylim[0]])  # Flip for microscopy convention
        ax.set_xlabel(xlabel, fontsize=9, color=text_color)
        ax.set_ylabel(ylabel, fontsize=9, color=text_color)
        ax.set_title(title, fontsize=10, fontweight='bold', color='white', pad=8)
        ax.tick_params(colors=text_color, labelsize=8)
        ax.grid(True, linewidth=0.3, alpha=0.4, color=grid_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
            spine.set_linewidth(0.5)
    
    # --- 3D View (first subplot) ---
    ax3d = fig.add_subplot(141, projection='3d', facecolor='black')
    ax3d.set_facecolor('black')
    
    # Plot 3D surfaces using marching cubes
    try:
        verts_n, faces_n, _, _ = measure.marching_cubes(masks_nucleus, level=0.5)
        ax3d.plot_trisurf(verts_n[:, 0], verts_n[:, 1], faces_n, verts_n[:, 2], 
                          color='#E57373', alpha=0.4, shade=True)
    except:
        pass
    
    try:
        verts_c, faces_c, _, _ = measure.marching_cubes(masks_cytosol, level=0.5)
        ax3d.plot_trisurf(verts_c[:, 0], verts_c[:, 1], faces_c, verts_c[:, 2], 
                          color='#78909C', alpha=0.3, shade=True)
    except:
        pass
    
    # Plot particles in 3D
    if len(protein_pos) > 0:
        ax3d.scatter(protein_pos[:, 0], protein_pos[:, 1], protein_pos[:, 2], 
                     c='#64B5F6', s=4, alpha=0.7, label='Protein')
    if len(rna_pos) > 0:
        ax3d.scatter(rna_pos[:, 0], rna_pos[:, 1], rna_pos[:, 2], 
                     c='#EF5350', s=4, alpha=0.9, label='RNA')
    # Plot transcription site as green circle (3x average of RNA/Protein size)
    if transcription_site is not None:
        ax3d.scatter([transcription_site[0]], [transcription_site[1]], [transcription_site[2]], 
                     c='#4CAF50', s=20, alpha=1.0, label='TS')
    
    # Style 3D axis
    ax3d.set_xlim([0, simulation_volume_size[0]])
    ax3d.set_ylim([0, simulation_volume_size[1]])
    ax3d.set_zlim([0, simulation_volume_size[2]])
    # Set realistic aspect ratio based on actual dimensions
    ax3d.set_box_aspect([simulation_volume_size[0], simulation_volume_size[1], simulation_volume_size[2]])
    ax3d.set_xlabel('X', fontsize=8, color=text_color, labelpad=2)
    ax3d.set_ylabel('Y', fontsize=8, color=text_color, labelpad=2)
    ax3d.set_zlabel('Z', fontsize=8, color=text_color, labelpad=2)
    ax3d.set_title('3D View', fontsize=10, fontweight='bold', color='white', pad=5)
    ax3d.tick_params(colors=text_color, labelsize=6)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor(grid_color)
    ax3d.yaxis.pane.set_edgecolor(grid_color)
    ax3d.zaxis.pane.set_edgecolor(grid_color)
    ax3d.xaxis._axinfo['grid']['color'] = grid_color
    ax3d.yaxis._axinfo['grid']['color'] = grid_color
    ax3d.zaxis._axinfo['grid']['color'] = grid_color
    ax3d.xaxis._axinfo['grid']['linewidth'] = 0.3
    ax3d.yaxis._axinfo['grid']['linewidth'] = 0.3
    ax3d.zaxis._axinfo['grid']['linewidth'] = 0.3
    ax3d.legend(loc='upper left', fontsize=7, facecolor='black', edgecolor=grid_color, labelcolor='white')
    
    # --- XY Projection ---
    ax_xy = fig.add_subplot(142)
    cell_xy = create_cell_image(nucleus_xy, cytosol_xy)
    ax_xy.imshow(cell_xy.transpose(1, 0, 2), origin='upper', aspect='equal')
    if len(protein_pos) > 0:
        ax_xy.scatter(protein_pos[:, 0], protein_pos[:, 1], c='#64B5F6', s=3, alpha=0.7)
    if len(rna_pos) > 0:
        ax_xy.scatter(rna_pos[:, 0], rna_pos[:, 1], c='#EF5350', s=3, alpha=0.9)
    if transcription_site is not None:
        ax_xy.scatter([transcription_site[0]], [transcription_site[1]], c='#4CAF50', s=24, alpha=1.0)
    style_2d_axis(ax_xy, 'X (px)', 'Y (px)', 'XY (top view)', 
                  [0, simulation_volume_size[0]], [0, simulation_volume_size[1]])
    
    # --- XZ Projection ---
    ax_xz = fig.add_subplot(143)
    cell_xz = create_cell_image(nucleus_xz, cytosol_xz)
    ax_xz.imshow(cell_xz.transpose(1, 0, 2), origin='lower', aspect='equal')
    if len(protein_pos) > 0:
        ax_xz.scatter(protein_pos[:, 0], protein_pos[:, 2], c='#64B5F6', s=3, alpha=0.7)
    if len(rna_pos) > 0:
        ax_xz.scatter(rna_pos[:, 0], rna_pos[:, 2], c='#EF5350', s=3, alpha=0.9)
    if transcription_site is not None:
        ax_xz.scatter([transcription_site[0]], [transcription_site[2]], c='#4CAF50', s=24, alpha=1.0)
    style_2d_axis(ax_xz, 'X (px)', 'Z (px)', 'XZ (front view)', 
                  [0, simulation_volume_size[0]], [0, simulation_volume_size[2]], origin_lower=True)
    
    # --- YZ Projection ---
    ax_yz = fig.add_subplot(144)
    cell_yz = create_cell_image(nucleus_yz, cytosol_yz)
    ax_yz.imshow(cell_yz.transpose(1, 0, 2), origin='lower', aspect='equal')
    if len(protein_pos) > 0:
        ax_yz.scatter(protein_pos[:, 1], protein_pos[:, 2], c='#64B5F6', s=3, alpha=0.7)
    if len(rna_pos) > 0:
        ax_yz.scatter(rna_pos[:, 1], rna_pos[:, 2], c='#EF5350', s=3, alpha=0.9)
    if transcription_site is not None:
        ax_yz.scatter([transcription_site[1]], [transcription_site[2]], c='#4CAF50', s=24, alpha=1.0)
    style_2d_axis(ax_yz, 'Y (px)', 'Z (px)', 'YZ (side view)', 
                  [0, simulation_volume_size[1]], [0, simulation_volume_size[2]], origin_lower=True)
    
    fig.suptitle(f'Cell Simulation at t={last_time_step}s', fontsize=12, fontweight='bold', 
                 color='white', y=0.98)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight', facecolor='black', edgecolor='none')
    print(f"All projections saved to {output_filename}")
    plt.close()
    
    return {'xy': (nucleus_xy, cytosol_xy), 'xz': (nucleus_xz, cytosol_xz), 'yz': (nucleus_yz, cytosol_yz)}
