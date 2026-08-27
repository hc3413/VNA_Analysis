"""Phase-4 item B acceptance demo — de-embedding shown, not asserted.

1. Held-out-standard validation on the dedicated r10 de-embedding set
   (CPW_mem_James_161024): the ABCD calibration is built from thrus c2+c3 and
   applied to the HELD-OUT thru c1; after perfect de-embedding a thru is the
   identity two-port, so the residual |S21 - 1| IS the de-embedding error.
   Output: methods-grade raw/de-embedded/ideal panel + per-band error-vector
   FOM table (CSV) + one-line residual attribution.

2. Calibration-comparison panel: the same physical ISS thru measured in
   mag_angle_260424 under the LRRM and LRM+ VNA calibrations - the per-band
   rms S21 difference quantifies how much the calibration choice moves any
   downstream number.

Run with an environment holding BOTH stacks:
    ../IS_Analysis/ISvenv/bin/python p4_demo_deembed_qa.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')

import io
import contextlib

from function_store import import_data, calibration_ABCD
from vna_bridge import deembed_copies, deembed_validation, calibration_comparison

REPO = Path(__file__).resolve().parent
OUT = REPO / 'Output_p4_scratch'
OUT.mkdir(exist_ok=True)


def main():
    # ---- 1. r10 held-out thru validation (James de-embedding set) ----------
    print('=== De-embedding validation: CPW_mem_James_161024, r10 thru set ===')
    with contextlib.redirect_stdout(io.StringIO()):
        james = import_data(str(REPO / 'CPW_mem_James_161024'))
    thrus = sorted([m for m in james if m.state == 'thru'],
                   key=lambda m: m.dev_col)
    print('  thrus:', [m.filename for m in thrus])
    held_out, cal_thrus = thrus[0], thrus[1:]
    print(f'  held out: {held_out.filename}; calibration from '
          f'{[m.filename for m in cal_thrus]}')
    with contextlib.redirect_stdout(io.StringIO()):
        X = calibration_ABCD(cal_thrus)
    dm_held = deembed_copies([held_out], X)[0]

    fig, rows = deembed_validation(
        held_out, dm_held, label=f'held-out {held_out.filename}',
        export_stem=OUT / 'CPWW2_r10_deembed_validation_James161024')
    print('  FOM per band (EVM rms |S21-1|):')
    for r in rows:
        print(f"    {r['band']:>16}: raw {r['EVM_raw_rms']:.4f} -> de-embedded "
              f"{r['EVM_deembedded_rms']:.4f}  (x{r['improvement']:.1f} better, N={r['N']})")

    # ---- 2. LRRM vs LRM+ on the same ISS thru (mag_angle_260424) -----------
    print('\n=== Calibration comparison: ISS thru, LRRM vs LRM+ (mag_angle_260424) ===')
    with contextlib.redirect_stdout(io.StringIO()):
        batch = import_data(str(REPO / 'mag_angle_260424'))
    lrrm = next(m for m in batch if 'lrrm' in m.filename.lower())
    lrmp = next(m for m in batch if 'lrm+' in m.filename.lower()
                and 'wafer0' in m.filename.lower())
    print(f'  LRRM: {lrrm.filename} | LRM+: {lrmp.filename}')
    fig, rows = calibration_comparison(
        lrrm.network, lrmp.network, labels=('LRRM', 'LRM+'),
        export_stem=OUT / 'ISS_thru_LRRM_vs_LRMplus_260424')

    print('Done. Exports in', OUT)


if __name__ == '__main__':
    main()
