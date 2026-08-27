"""Phase-4 item A acceptance demo — the VNA -> ISdata bridge on Wafer2 r2_c1.

Loads the CPW_mem_oscillator_220524/Memristor session, builds the ABCD
de-embedding from the four tapered r10 thrus of the SAME session, de-embeds
the r2_c1 state series, converts to IS_Analysis ISdata via the B-element
(series ABCD) path, and renders the brief's definition-of-done:
  (a) a Bode |Z|+phase per-state family via IS_plot,
  (b) a Nyquist via IS_plot (validity-clipped to >=2.6 MHz),
plus a formed-state DC ladder (coolwarm by bias) showing DC_offset flows.

Also prints the ABCD-vs-S21 conversion agreement over the valid band.

Run with an environment holding BOTH stacks (skrf + the IS_Analysis deps):
    ../IS_Analysis/ISvenv/bin/python p4_demo_bridge.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')

import io
import contextlib

import numpy as np

from VNAdata import VNAdata
from function_store import calibration_ABCD, duplicate_check
from vna_bridge import bridge_batch, conversion_agreement, deembed_copies
from plot_style import set_plot_style, save_figure

REPO = Path(__file__).resolve().parent
OUT = REPO / 'Output_p4_scratch'
OUT.mkdir(exist_ok=True)

FIG_SIZE = tuple(set_plot_style(export_data=True))

# IS_Analysis plotting arrives via vna_bridge's sys.path setup
from IS_Functions import IS_plot  # noqa: E402


def main():
    print('=== Importing CPW_mem_oscillator_220524/Memristor ===')
    with contextlib.redirect_stdout(io.StringIO()):
        vna = VNAdata.load_batch(str(REPO / 'CPW_mem_oscillator_220524' / 'Memristor'))
        duplicate_check(vna.measurements)  # drop non-monotonic freq rows (existing tool)
    print(f'  {len(vna)} S2P files')

    # ABCD calibration from the SAME session's tapered r10 thrus (c1-c4);
    # 'thrunotaper' is the straight-line variant and is excluded on purpose.
    thrus = [m for m in vna if m.wafer_number == 2 and m.dev_row == 10
             and m.state == 'thru']
    print(f'  tapered thrus for calibration: {[m.filename for m in thrus]}')
    with contextlib.redirect_stdout(io.StringIO()):
        X = calibration_ABCD(thrus)

    # r2_c1 state series -> de-embed (copies; raw imports untouched) -> ISdata
    r2c1 = vna.filter(wafer=2, row=2, col=1)
    members = bridge_batch(list(r2c1), abcd_cal=X)
    print(f'  bridged {len(members)} r2_c1 spectra to ISdata '
          f'(instrument=vna, wafer=CPW-W2)')

    # Conversion cross-check: B element vs 2*Z0*(1-S21)/S21, valid band only
    dm = deembed_copies(list(r2c1), X)
    med_all, max_all = [], []
    for f in dm:
        med, mx = conversion_agreement(f.network)
        med_all.append(med)
        max_all.append(mx)
    print(f'  ABCD-B vs S21 conversion agreement (>=2.6 MHz): median of medians '
          f'{np.median(med_all):.2e}, worst point {np.max(max_all):.2e} (relative)')

    # ---- (a) Bode |Z|+phase per-state family --------------------------------
    def pick(state, dc=None, contains=None):
        for m in members:
            if m.res_state != state:
                continue
            if contains and contains not in m.file_name.lower():
                continue
            if dc is None and m.DC_offset is None:
                return m
            if dc is not None and m.DC_offset is not None and abs(m.DC_offset - dc) < 1e-9:
                return m
        return None

    family = [m for m in (pick('pristine'),
                          pick('formed', dc=0.0),
                          pick('set'),
                          pick('reset')) if m is not None]
    print('  state family:', [(m.res_state, m.file_name) for m in family])

    fig, ax = IS_plot([family], 'Zabsphi')
    if fig is not None:
        save_figure(fig, OUT / 'CPWW2_r2c1_states_Zabsphi', close=True)
        print('  -> CPWW2_r2c1_states_Zabsphi.{svg,tiff} (full band; '
              'network-analysis validity starts at 2.6 MHz)')

    # ---- (b) Nyquist, clipped to the valid band -----------------------------
    valid_lim = (2.6e6, 2.1e10)   # IS_plot freq_lim needs finite bounds
    # Display choice on top of the 2.6 MHz validity floor: near the floor these
    # high-|Z| states sit at the method's resolution ceiling, so the Nyquist is
    # rendered from 10 MHz where the capacitive arc is resolved.
    fig, ax = IS_plot([family], 'colecole', freq_lim=(1e7, 2.1e10))
    if fig is not None:
        save_figure(fig, OUT / 'CPWW2_r2c1_states_nyquist', close=True)
        print('  -> CPWW2_r2c1_states_nyquist.{svg,tiff} (display >=10 MHz)')

    # ---- bonus: formed DC ladder, coolwarm by bias --------------------------
    ladder = [m for m in members if m.res_state == 'formed' and m.DC_offset is not None]
    fig, ax = IS_plot([ladder], 'Zabsphi', c_bar=2, freq_lim=valid_lim)
    if fig is not None:
        save_figure(fig, OUT / 'CPWW2_r2c1_formed_DCladder_Zabsphi', close=True)
        print('  -> CPWW2_r2c1_formed_DCladder_Zabsphi.{svg,tiff}')

    print('Done. Exports in', OUT)


if __name__ == '__main__':
    main()
