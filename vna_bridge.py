"""vna_bridge — de-embedded VNA S-parameters -> IS_Analysis ISdata objects.

Phase-4 work items A and B (claude_code_brief_p4_v1.md, VNA_Analysis repo).

Item A (the bridge): after ABCD de-embedding, extract the DUT's complex
impedance and package it as IS_Analysis `ISdata` objects, so every IS
plot/fit function applies unchanged. "The VNA is then just a means to an end
for high freq."

Conversion path (assumptions documented here once):
  * The de-embedding is the repo's existing ABCD path: X = sqrt(A_thru)^-1,
    A_DUT = X . A_meas . X (function_store.calibration_ABCD / deembed_ABCD).
    Reference planes therefore sit at the centre of the on-wafer thru, i.e.
    at the DUT gap; the taper/feed is what the thru halves remove.
  * The memristor bridges the CPW signal-line gap -> a SERIES element in a
    matched Z0 = 50 ohm line. Its ABCD matrix is [[1, Z], [0, 1]], so the
    de-embedded B element IS the device impedance: Z_DUT = A_DUT[:, 0, 1].
    This is the preferred path (brief section 2): it needs no matched-load
    assumption beyond what the ABCD cascade already encodes.
  * The S21 shortcut Z = 2*Z0*(1 - S21)/S21 is implemented for cross-checking
    only; it is exact for an ideal series element in a matched system and its
    agreement with the B element is reported by the demo.
  * Network-analysis validity: the method resolves our impedance range only
    above ~2.6 MHz. The bridge does NOT clip - it tags every ISdata with
    instrument='vna' (IS_plot_stitch shades the <2.6 MHz floor automatically)
    and attaches `valid_band_Hz` to the instance for downstream shading.

Cross-repo import mechanism: IS_Analysis is expected as a sibling checkout
(../IS_Analysis). Its path is appended (not prepended) to sys.path, so
VNA_Analysis's own modules keep precedence. NOTE the two repos both own a
`plot_style` module; in bridge sessions the VNA one wins - it carries the
same save_figure/set_plot_style contract, but NOT IS-only extras such as
STATE_COLORS (import those from the IS_Analysis file directly if needed).
Run bridge scripts with an environment holding BOTH stacks, e.g.
../IS_Analysis/ISvenv (scikit-rf 1.0.0 pinned to match VNAenv).

Item B (de-embedding shown, not asserted) lives at the bottom:
deembed_validation() (held-out standard vs ideal, per-band error-vector FOM,
one-line residual attribution) and calibration_comparison() (LRRM vs LRM+).
"""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import numpy as np

from function_store import S2PFile, deembed_ABCD  # noqa: F401 (re-export)

# --- IS_Analysis sibling import ---------------------------------------------
_IS_ANALYSIS = Path(__file__).resolve().parent.parent / 'IS_Analysis'
if str(_IS_ANALYSIS) not in sys.path:
    sys.path.append(str(_IS_ANALYSIS))          # append: VNA modules keep precedence
try:
    from IS_Import import ISdata, transform_measurement_data
except ImportError as e:                         # pragma: no cover
    raise ImportError(
        f"vna_bridge needs the IS_Analysis repo as a sibling checkout at "
        f"{_IS_ANALYSIS} (and an environment with its dependencies): {e}")

# The VNA network-analysis method resolves our impedance range only above this
VNA_VALID_BAND_HZ = (2.6e6, None)

# TODO(paper chat): per-device geometry for normalised (permittivity/modulus)
# views. Placeholder = the ISdata default (20 um square, 30 nm) so epsilon'
# axes are NOT quantitative for CPW devices until real areas/thicknesses (FIB
# trims differ per device) are supplied.
PLACEHOLDER_C_VAC = 8.854e-12 * (20e-6) ** 2 / (30e-9)


def parse_dc_offset(filename: str):
    """DC bias in volts from a VNA filename, or None when not a bias point.

    Handles the campaign tokens: 'pos0.8dc' / 'neg1.2dc' / '_0dc',
    'Vdc0' / 'Vdc0.8' / 'Vdcneg0.4'. Bare state files (no token) -> None.
    """
    stem = filename.lower()
    m = re.search(r'(pos|neg)(\d+(?:\.\d+)?)dc', stem)
    if m:
        val = float(m.group(2))
        return -val if m.group(1) == 'neg' else val
    m = re.search(r'_(\d+(?:\.\d+)?)dc', stem)
    if m:
        return float(m.group(1))
    m = re.search(r'vdc(neg|pos)?(\d+(?:\.\d+)?)', stem)
    if m:
        val = float(m.group(2))
        return -val if m.group(1) == 'neg' else val
    return None


def series_impedance_abcd(network):
    """Device impedance from the ABCD B element (PREFERRED path).

    For a series element bridging the line, A = [[1, Z], [0, 1]] -> Z = B.
    Call on a de-embedded network.
    """
    return network.a[:, 0, 1]


def series_impedance_s21(network):
    """Device impedance from Z = 2*Z0*(1-S21)/S21 (cross-check path only)."""
    s21 = network.s[:, 1, 0]
    z0 = np.real(network.z0[:, 0])
    with np.errstate(divide='ignore', invalid='ignore'):
        return 2.0 * z0 * (1.0 - s21) / s21


def conversion_agreement(network, band=VNA_VALID_BAND_HZ):
    """Median/max relative deviation |Z_s21 - Z_abcd| / |Z_abcd| in band."""
    z_a = series_impedance_abcd(network)
    z_s = series_impedance_s21(network)
    f = network.f
    sel = np.isfinite(z_a) & np.isfinite(z_s) & (np.abs(z_a) > 0)
    if band[0]:
        sel &= f >= band[0]
    if band[1]:
        sel &= f <= band[1]
    rel = np.abs(z_s[sel] - z_a[sel]) / np.abs(z_a[sel])
    return float(np.median(rel)), float(np.max(rel))


def deembed_copies(s2p_files, abcd_cal):
    """De-embed via the existing ABCD path WITHOUT mutating the raw imports.

    function_store.deembed_ABCD rewrites networks in place (fine in the
    notebooks, wrong for a bridge that must leave raw data untouched); this
    deep-copies first and then reuses that function unchanged.
    """
    copies = copy.deepcopy(list(s2p_files))
    return deembed_ABCD(copies, abcd_cal)


def s2p_to_isdata(s2p, method='abcd_series', C_vac=None, C_pad=0.0,
                  valid_band=VNA_VALID_BAND_HZ):
    """Package one (de-embedded) S2PFile as an IS_Analysis ISdata.

    Metadata mapping (brief section 2): wafer -> 'CPW-W<n>',
    row/col -> device_name 'r<row>_c<col>', state -> res_state (as in the
    filename), instrument='vna', DC_offset from the filename bias token,
    run_number = chronological run. area stays None
    (TODO(paper chat): FIB-trimmed areas differ per device - do not guess).
    C_pad = 0 (no pad on CPW). C_vac defaults to PLACEHOLDER_C_VAC (see TODO
    above). The instance also carries `valid_band_Hz` (not an ISdata field;
    plain attribute) so downstream plots can shade validity.

    The full measured band is kept - nothing below 2.6 MHz is dropped.
    """
    z_dut = series_impedance_abcd(s2p.network) if method == 'abcd_series' \
        else series_impedance_s21(s2p.network)
    freq = s2p.network.f
    zabsphi = np.column_stack((freq, np.abs(z_dut), np.angle(z_dut, deg=True)))

    device_name = None
    if s2p.dev_row is not None and s2p.dev_col is not None:
        device_name = f"r{s2p.dev_row}_c{s2p.dev_col}"

    m = ISdata(
        file_name=s2p.filename,
        run_number=s2p.run,
        Zabsphi=zabsphi,
        folder_path=Path(getattr(s2p, 'source_dir', '.')),
        DC_offset=parse_dc_offset(s2p.filename),
        res_state=s2p.state,
        device_name=device_name,
        wafer=f"CPW-W{s2p.wafer_number}",
        area=None,   # TODO(paper chat): per-device area (FIB trims differ)
        instrument='vna',
        C_pad=C_pad,
        C_vac=C_vac if C_vac is not None else PLACEHOLDER_C_VAC,
    )
    transform_measurement_data(m)
    m.valid_band_Hz = valid_band
    return m


def bridge_batch(s2p_files, abcd_cal=None, **kwargs):
    """De-embed (optional) then convert a batch of S2PFile -> [ISdata].

    abcd_cal: the X = sqrt(A_thru)^-1 array from calibration_ABCD; None means
    the inputs are already de-embedded (or raw, if you want raw impedance).
    """
    files = deembed_copies(s2p_files, abcd_cal) if abcd_cal is not None else s2p_files
    return [s2p_to_isdata(f, **kwargs) for f in files]


# ============================================================================
# Item B — de-embedding shown, not asserted (Kasmi-style QA)
# ============================================================================

QA_BANDS_HZ = [
    (9e3, 2.6e6),      # below the network-analysis validity floor
    (2.6e6, 1e8),
    (1e8, 1e9),
    (1e9, 5e9),
    (5e9, 1e10),
    (1e10, 2e10),
    (2e10, 6.7e10),    # 67 GHz session (James set reaches it)
]


def _band_label(lo, hi):
    def fmt(v):
        for unit, s in ((1e9, 'GHz'), (1e6, 'MHz'), (1e3, 'kHz')):
            if v >= unit:
                return f"{v/unit:g} {s}"
        return f"{v:g} Hz"
    return f"{fmt(lo)}-{fmt(hi)}"


def deembed_validation(raw_std, deembedded_std, label='held-out thru',
                       bands=None, export_stem=None):
    """Validate a de-embedding against a KNOWN standard (ideal thru).

    raw_std / deembedded_std: the same standard before and after applying a
    calibration built WITHOUT it (hold it out). After perfect de-embedding a
    thru is the identity two-port: S21 = 1 + 0j. FOM per frequency band =
    RMS error-vector magnitude |S21 - 1| (raw and de-embedded) and the
    improvement factor. A one-line residual attribution states whether the
    remaining error is magnitude-like (loss/contact repeatability) or
    phase-like (reference-plane / electrical-length mismatch).

    Returns (fig, rows) where rows is a list of per-band dicts; also writes
    '<export_stem>_FOM.csv' and the SVG+TIFF panel when export_stem is given.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from plot_style import save_figure, set_plot_style

    fig_size = set_plot_style(export_data=True)
    bands = bands or QA_BANDS_HZ

    f = raw_std.network.f
    s21_raw = raw_std.network.s[:, 1, 0]
    s21_dm = deembedded_std.network.s[:, 1, 0]

    rows = []
    for lo, hi in bands:
        sel = (f >= lo) & (f < hi)
        if sel.sum() < 3:
            continue
        evm_raw = float(np.sqrt(np.mean(np.abs(s21_raw[sel] - 1.0) ** 2)))
        evm_dm = float(np.sqrt(np.mean(np.abs(s21_dm[sel] - 1.0) ** 2)))
        mag_part = float(np.sqrt(np.mean((np.abs(s21_dm[sel]) - 1.0) ** 2)))
        phi_part = float(np.sqrt(np.mean(np.angle(s21_dm[sel]) ** 2)))
        rows.append({'band': _band_label(lo, hi), 'f_lo_Hz': lo, 'f_hi_Hz': hi,
                     'N': int(sel.sum()), 'EVM_raw_rms': evm_raw,
                     'EVM_deembedded_rms': evm_dm,
                     'improvement': evm_raw / evm_dm if evm_dm > 0 else np.inf,
                     'mag_residual_rms': mag_part,
                     'phase_residual_rms_rad': phi_part})

    # one-line residual attribution over the VALID bands
    valid = [r for r in rows if r['f_lo_Hz'] >= VNA_VALID_BAND_HZ[0]]
    mag_rms = np.sqrt(np.mean([r['mag_residual_rms'] ** 2 for r in valid]))
    phi_rms = np.sqrt(np.mean([r['phase_residual_rms_rad'] ** 2 for r in valid]))
    if phi_rms > mag_rms:
        attribution = (f"residual is PHASE-dominated ({np.degrees(phi_rms):.2f} deg rms vs "
                       f"{mag_rms:.4f} mag rms) -> reference-plane/electrical-length mismatch")
    else:
        attribution = (f"residual is MAGNITUDE-dominated ({mag_rms:.4f} rms vs "
                       f"{np.degrees(phi_rms):.2f} deg rms) -> loss/contact repeatability")
    print(f"[{label}] residual attribution: {attribution}")

    # methods-grade figure: |S21| dB + phase, raw vs de-embedded vs ideal
    fig, ax = plt.subplots(2, 1, figsize=(fig_size[0], fig_size[1] * 1.6),
                           sharex=True, constrained_layout=True)
    with np.errstate(divide='ignore'):
        ax[0].semilogx(f, 20 * np.log10(np.abs(s21_raw)), '--', lw=0.8,
                       label=f'raw ({label})')
        ax[0].semilogx(f, 20 * np.log10(np.abs(s21_dm)), '-', lw=0.9,
                       label='de-embedded')
    ax[0].axhline(0, color='black', lw=0.6, label='ideal thru')
    ax[1].semilogx(f, np.degrees(np.angle(s21_raw)), '--', lw=0.8)
    ax[1].semilogx(f, np.degrees(np.angle(s21_dm)), '-', lw=0.9)
    ax[1].axhline(0, color='black', lw=0.6)
    for a in ax:
        a.axvspan(f.min() / 2, VNA_VALID_BAND_HZ[0], color='0.5', alpha=0.12,
                  hatch='///', lw=0)
    ax[0].set_ylabel(r'$|S_{21}|$ (dB)')
    ax[1].set_ylabel(r'Phase of $S_{21}$ ($^{\circ}$)')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].legend(fontsize=5.5)

    if export_stem is not None:
        df = pd.DataFrame(rows)
        csv_path = Path(str(export_stem) + '_FOM.csv')
        df.to_csv(csv_path, index=False)
        print(f"FOM table -> {csv_path}")
        save_figure(fig, export_stem)
    return fig, rows


def calibration_comparison(net_a, net_b, labels=('LRRM', 'LRM+'), bands=None,
                           export_stem=None):
    """Compare the same physical standard measured under two VNA calibrations.

    Panels: |S21| dB and phase for both, plus the per-band RMS complex
    difference |S21_a - S21_b| printed as a table. Quantifies how much the
    calibration CHOICE (LRRM vs LRM+) moves the answer.

    Returns (fig, rows).
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from plot_style import save_figure, set_plot_style

    fig_size = set_plot_style(export_data=True)
    bands = bands or QA_BANDS_HZ

    f_a, f_b = net_a.f, net_b.f
    s_a = net_a.s[:, 1, 0]
    # interpolate B onto A's grid where grids differ (log campaigns vs linear)
    if len(f_a) != len(f_b) or not np.allclose(f_a, f_b):
        s_b = np.interp(f_a, f_b, net_b.s[:, 1, 0].real) + \
            1j * np.interp(f_a, f_b, net_b.s[:, 1, 0].imag)
        f_lo_common, f_hi_common = max(f_a.min(), f_b.min()), min(f_a.max(), f_b.max())
    else:
        s_b = net_b.s[:, 1, 0]
        f_lo_common, f_hi_common = f_a.min(), f_a.max()

    rows = []
    for lo, hi in bands:
        sel = (f_a >= max(lo, f_lo_common)) & (f_a < min(hi, f_hi_common))
        if sel.sum() < 3:
            continue
        diff = float(np.sqrt(np.mean(np.abs(s_a[sel] - s_b[sel]) ** 2)))
        rows.append({'band': _band_label(lo, hi), 'N': int(sel.sum()),
                     'rms_S21_difference': diff})
    print(f"Calibration comparison {labels[0]} vs {labels[1]} (rms |dS21| per band):")
    for r in rows:
        print(f"  {r['band']:>16}: {r['rms_S21_difference']:.4g}  (N={r['N']})")

    fig, ax = plt.subplots(2, 1, figsize=(fig_size[0], fig_size[1] * 1.6),
                           sharex=True, constrained_layout=True)
    with np.errstate(divide='ignore'):
        ax[0].semilogx(f_a, 20 * np.log10(np.abs(s_a)), '-', lw=0.9, label=labels[0])
        ax[0].semilogx(f_b, 20 * np.log10(np.abs(net_b.s[:, 1, 0])), '--', lw=0.9,
                       label=labels[1])
    ax[1].semilogx(f_a, np.degrees(np.angle(s_a)), '-', lw=0.9)
    ax[1].semilogx(f_b, np.degrees(np.angle(net_b.s[:, 1, 0])), '--', lw=0.9)
    ax[0].set_ylabel(r'$|S_{21}|$ (dB)')
    ax[1].set_ylabel(r'Phase of $S_{21}$ ($^{\circ}$)')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].legend(fontsize=6)

    if export_stem is not None:
        pd.DataFrame(rows).to_csv(Path(str(export_stem) + '_diff.csv'), index=False)
        save_figure(fig, export_stem)
    return fig, rows
