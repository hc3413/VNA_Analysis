"""Canonical plot style for all HC analysis repositories.

Copied verbatim from Thin_Film_Analysis/plot_style.py (the reference
implementation, because it is the only one that carries _LAST_STYLE /
apply_plot_style) and extended with save_figure() for paired SVG+TIFF export.

Drop-in replacement for:   Thin_Film_Analysis/plot_style.py
                           IS_Analysis/plot_style.py
                           Electronic_properties_of_thin_films/Functions_style.py
                           MPMS/Functions_style.py
NOT a drop-in for:         TEM_Analysis/Functions_style.py
                           (different signature: set_plot_style(plot_type=, multiplier=))

Inside a repo that has import_dep.py, keep the original first line
`from import_dep import *`. Standalone, the guarded import below is used.
"""

try:
    from import_dep import *          # inside an HC analysis repo
except ImportError:                    # standalone / plugin context
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from cycler import cycler


# Remembers the arguments of the most recent set_plot_style() call - i.e. the
# style the notebook asked for at the top of the session. Plotting functions
# re-apply it through apply_plot_style() so that they never silently discard
# settings the notebook chose (see that function for why this matters).
_LAST_STYLE = {'export_data': False, 'powerpoint_data': False,
               'use_tex': True, 'markersize': None}


def set_plot_style(export_data = False, powerpoint_data = False, use_tex=True, markersize=None):
    """
    Set publication-quality plot styles.

    Parameters:
    -----------
    use_tex : bool
        Whether to use LaTeX for rendering text (default: True)
    markersize : float, optional
        Override the default markersize. If None, uses default (4.0 for export, 6.0 for display)
    """
    _LAST_STYLE.update({'export_data': export_data,
                        'powerpoint_data': powerpoint_data,
                        'use_tex': use_tex, 'markersize': markersize})

    # Set the figure size based on whether we are visualising or exporting the data
    if export_data == True:
        fig_size = [3.5, 2.625] # Publication ready sizes
    else:
        fig_size = [9, 6] # Better for visualisation
    
    # Use a colorblind-friendly colormap with at least 10 distinct colors
    cmap_colors =   sns.color_palette("colorblind", 12) #sns.color_palette("bright", 10)
    color_cycler = cycler('color', cmap_colors)
    color_cycler_2 = cycler('color', ['#0C5DA5', '#00B945', '#FF9500', 
                                           '#FF2C00', '#845B97', '#474747', '#9e9e9e'])
    
    # Science style settings
    plt.rcParams.update({
        # Figure settings
        'figure.figsize':fig_size,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        # Project default for raster exports (png/tiff/jpg). Vector formats
        # (svg/pdf) ignore it except for any embedded raster elements.
        'savefig.dpi': 600,
        
        # Font and text settings
        'font.family': ['serif'],
        'font.size': 9,  # Base font size
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'mathtext.fontset': 'dejavuserif',
        'text.usetex': use_tex,
        'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
        
        # Axes settings
        'axes.linewidth': 0.5,
        'axes.prop_cycle': color_cycler ,
        
        # Grid settings
        'grid.linewidth': 0.5,
        'axes.grid': True,
        'axes.axisbelow': True,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.4,
        
        # Line settings
        'lines.linewidth': 1.0,
        'lines.markersize': markersize if markersize is not None else (4.0 if export_data else 6.0),
        # Errorbar settings
        'errorbar.capsize': 0,
        
        # Tick settings
        'xtick.direction': 'in',
        'xtick.major.size': 3.0,
        'xtick.major.width': 0.5,
        'xtick.minor.size': 1.5,
        'xtick.minor.visible': True,
        'xtick.minor.width': 0.5,
        'xtick.top': True,
        
        'ytick.direction': 'in',
        'ytick.major.size': 3.0,
        'ytick.major.width': 0.5,
        'ytick.minor.size': 1.5,
        'ytick.minor.visible': True,
        'ytick.minor.width': 0.5,
        'ytick.right': True,
        
        # Prevent autolayout to ensure that the figure size is obeyed
        'figure.autolayout': False,
    })
    
    return fig_size


def apply_plot_style(export_data=None):
    """Re-apply the style the notebook set up, for use inside plotting functions.

    Plotting functions need the figure size, but calling
    ``set_plot_style(export_data=..., use_tex=True)`` to get it also *resets*
    every other rcParam to that call's defaults. The notebook's

        fig_size = set_plot_style(export_data=export_data, use_tex=True, markersize=0.1)

    was therefore being undone by the first plot call, which put markersize back
    to 4.0/6.0 and forced ``use_tex`` on even if the notebook had turned it off.

    This applies the remembered settings instead, overriding only
    ``export_data`` when a function is asked to render at publication size.
    Returns the figure size, exactly as ``set_plot_style`` does.

    An ``export_data`` override is not remembered: it styles that one figure and
    the next plotting call restores the notebook's own settings. In normal use
    the point is moot, because notebooks pass ``export_data=export_data`` and so
    never override anything.
    """
    remembered = dict(_LAST_STYLE)
    kwargs = dict(remembered)
    if export_data is not None:
        kwargs['export_data'] = export_data
    try:
        return set_plot_style(**kwargs)
    finally:
        # An export_data override applies to this figure only - it must not
        # become the session's new baseline.
        _LAST_STYLE.update(remembered)


def add_colorbar(fig, ax, sm, min_val, max_val, fig_size, field = True):
    """
    Add and adjust a colorbar to the given axis.
    
    Parameters:
    - fig: The figure object.
    - ax: The axis object to which the colorbar will be added.
    - sm: The ScalarMappable object for the colorbar.
    - min_field: The minimum value for the colorbar ticks.
    - max_field: The maximum value for the colorbar ticks.
    - fig_size: The size of the figure.
    - field: Whether the colorbar represents a magnetic field (True) or temperature (False).
    """
    cax = fig.add_subplot(ax)  # Use the provided axis for the colorbar
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_ticks([min_val, max_val])
    
    if field:
        cbar.set_ticklabels([f'{min_val:.1f} T', f'{max_val:.1f} T']) # Field
    else:
        cbar.set_ticklabels([f'{min_val:.1f} K', f'{max_val:.1f} K']) # Temperature
   
    cbar.minorticks_off()  # Remove minor ticks
    cbar.outline.set_linewidth(0.5)

    # Adjust colorbar position and size based on figure size
    if fig_size == [3.5, 2.625]:
        height_scale = 0.8
        pos = cax.get_position()
        new_height = pos.height * height_scale
        new_y0 = pos.y0 + (pos.height - new_height) / 2  # Center the colorbar vertically
        cax.set_position([
            pos.x0 + 0.03,  # Adjust x0 to move the colorbar to the right
            new_y0,  # Center the colorbar vertically
            pos.width * 0.3,  # Adjust width to shrink the colorbar
            new_height  # Adjust height to shrink the colorbar
        ])
    else:
        height_scale = 0.8
        pos = cax.get_position()
        new_height = pos.height * height_scale
        new_y0 = pos.y0 + (pos.height - new_height) / 2  # Center the colorbar vertically
        cax.set_position([
            pos.x0 + 0.03,  # Adjust x0 to move the colorbar to the right
            new_y0,  # Center the colorbar vertically
            pos.width * 0.3,  # Adjust width to shrink the colorbar
            new_height  # Adjust height to shrink the colorbar
        ])


from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt

def generate_colormaps_and_normalizers(dat):
    """
    Generate colormaps and normalizers for temperature and field values.
    Taking the input data to work out the maximum and minimum values for the colormaps.
    Parameters:
    -----------
    dat : list
        List of data objects containing temperature and field information.

    Returns:
    --------
    cmap_temp : Colormap
        Colormap for temperature values.
    cmap_field : Colormap
        Colormap for field values.
    norm_temp : Normalize
        Normalizer for temperature values.
    norm_field : Normalize
        Normalizer for field values.
    cmap_dat : ndarray
        Colormap for distinguishing between datasets.
    """
    # Extract raw temperature values to prevent rounding
    all_temps = []
    for d in dat:
        temps = np.copy(d.tf_av).reshape((d.ctf[4] * d.ctf[5], 2))
        all_temps = np.concatenate([all_temps, temps[:, 0]])  # Extract the temperature values (first column)

    # Concatenate all field arrays
    all_fields = np.concatenate([d.ctf[2] for d in dat])

    # Find the min and max values
    min_temp = np.min(all_temps)
    max_temp = np.max(all_temps)
    min_field = np.min(all_fields)
    max_field = np.max(all_fields)

    # Normalize the temperature and field values
    norm_temp = Normalize(vmin=min_temp, vmax=max_temp)
    norm_field = Normalize(vmin=min_field, vmax=max_field)

    # Generate colormaps
    cmap_temp = plt.get_cmap('coolwarm')
    cmap_field = plt.get_cmap('coolwarm')
    
    # Generate a list of markers for the data
    mark_p = [ 'x', 'o', '*', 'd', '^', 'v','+',  '<', '>', 'p', 'P', 'h', 'H', 'X', 'D', '|', '_', '1', '2', '3', '4', '8', 's', 'p', 'P', 'o', 'h', 'H', 'X', 'd', 'D', '|', '_', '1', '2', '3', '4', '8', 's']

   

    return cmap_temp, cmap_field, norm_temp, norm_field, mark_p, min_temp, max_temp, min_field, max_field

# ---------------------------------------------------------------------------
# Style contract. check_plot_style.py audits every repo against this dict.
# These are the rcParams that must be identical everywhere. Anything not
# listed here a repo is free to override.
# ---------------------------------------------------------------------------
STYLE_CONTRACT = {
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.family': ['serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'axes.grid': True,
    'axes.axisbelow': True,
    'legend.frameon': True,
    'legend.framealpha': 0.4,
    'lines.linewidth': 1.0,
    'errorbar.capsize': 0,
    'xtick.direction': 'in',
    'xtick.major.size': 3.0,
    'xtick.major.width': 0.5,
    'xtick.minor.size': 1.5,
    'xtick.minor.visible': True,
    'xtick.minor.width': 0.5,
    'xtick.top': True,
    'ytick.direction': 'in',
    'ytick.major.size': 3.0,
    'ytick.major.width': 0.5,
    'ytick.minor.size': 1.5,
    'ytick.minor.visible': True,
    'ytick.minor.width': 0.5,
    'ytick.right': True,
    'figure.autolayout': False,
}

# Publication figure size, inches. Single column of a two-column journal.
FIG_SIZE_EXPORT = [3.5, 2.625]
# Double = exactly two singles side by side, same height (interview 2026-08-08).
FIG_SIZE_EXPORT_DOUBLE = [7.0, 2.625]
FIG_SIZE_SCREEN = [9, 6]

# Fixed colour semantics across every repository (interview 2026-08-08):
# swept physical quantities use coolwarm; a single trace is black; categorical
# families use the seaborn colorblind cycle set in set_plot_style.
COLOR_SEMANTICS = {
    "temperature": "coolwarm",
    "dc_offset": "coolwarm",
    "magnetic_field": "coolwarm",
    "single_line": "black",
    "categorical": "colorblind",
}

# Axis labels: full word, capitalised, unit in parentheses. Use these strings
# verbatim so multi-panel figures never mix conventions.
AXIS_LABELS = {
    "frequency": "Frequency (Hz)",
    "voltage": "Voltage (V)",
    "current": "Current (A)",
    "current_density": "Current density (A/cm$^2$)",
    "temperature": "Temperature (K)",
    "capacitance": "Capacitance (F)",
    "impedance": "Impedance ($\Omega$)",
    "phase": "Phase (deg)",
    "polarisation": "Polarisation ($\mu$C/cm$^2$)",
    "electric_field": "Electric field (kV/cm)",
    "magnetic_field": "Magnetic field (T)",
    "resistance": "Resistance ($\Omega$)",
    "sheet_resistance": "Sheet resistance ($\Omega$/sq)",
    "mobility": "Mobility (cm$^2$/Vs)",
    "carrier_density_2d": "Carrier density (cm$^{-2}$)",
    "carrier_density_3d": "Carrier density (cm$^{-3}$)",
    "time": "Time (s)",
    "thickness": "Thickness (nm)",
}


def save_figure(fig, output_stem, formats=('svg', 'tiff'), dpi=600,
                transparent=False, close=False):
    """Save one figure as a matched set of files sharing a single stem.

    The pipeline convention is that every exported figure exists as an SVG
    (vector, for Affinity Designer layout) and a TIFF (raster, for journals
    that will not take vector). Both are written from the same figure object
    in the same call so they can never drift apart.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    output_stem : str or pathlib.Path
        Path WITHOUT extension. Should sit in the sample's Output/ directory
        and reuse the source data filename stem, per the layout spec.
    formats : tuple of str
        Extensions to write. Default ('svg', 'tiff').
    dpi : int
        Raster resolution. Ignored by SVG except for embedded rasters.
    transparent : bool
        Transparent background. Default False - journals prefer white.
    close : bool
        Close the figure after saving.

    Returns
    -------
    list of pathlib.Path : the files written, in the order given.
    """
    from pathlib import Path
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        out = stem.with_suffix('.' + ext.lstrip('.'))
        fig.savefig(out, dpi=dpi, transparent=transparent)
        written.append(out)
    if close:
        plt.close(fig)
    return written
