# claude_code_brief_p4_v1.md — VNA_Analysis tooling for the IS paper, Phase 4 (2026-08-26)

*Work order for Claude Code running in this repo (VS Code). Self-contained; the paper
chat (Cowork) reads only your report file (§4). Companion brief in `IS_Analysis/` covers
that repo; if you work on both, do IS_Analysis item A (metric) first — nothing here
depends on it, but the paper chat wants consistent NRMSE reporting.*

## 0. Context in three sentences

The paper's GHz arm is CPW-W1/W2 (SiOx memristor in a tapered CPW centre, ~18–20 µm,
some FIB-trimmed; wafer 2 device r2_c1 = the main state series), measured on a VNA
9 kHz–20 GHz plus one 67 GHz session; raw S2P sets live in this repo's data folders
(`mag_angle_240424`, `mag_angle_260424`, `CPW_mem_270824/{SiOx,SiOxFIB,HfOx}`,
`CPW_mem_James_161024` = the r10 thru/open/short/line de-embedding set,
`CPW_mem_oscillator_220524`). The paper needs the VNA arm to speak the same language as
the low-frequency impedance arms: after de-embedding, convert to the device's complex
impedance and hand it to the IS_Analysis toolchain. HC: "the VNA is then just a means to
an end for high freq."

## 1. Ground rules

- Read `README.md`, `VNAdata.py`, `function_store.py`, `plot_style.py` first. Raw data
  folders are read-only inputs — never modify or move an .s2p/.S2P file.
- Style contract identical to IS_Analysis (SVG+TIFF via `save_figure`, text-as-text
  SVG, full-word axis labels, 3.5/7.0 in widths, colorblind categorical cycle).
- Existing calibration functions (`calibration_OS/2x/ABCD`, `deembed_ABCD`) are the
  starting point — HC used the **ABCD path**; extend, don't replace. Any robust variant
  is acceptable if validated (§3).
- Branch (suggest `p4-bridge`); reviewable commits.

## 2. Work item A — the VNA→ISdata bridge (the unlock; K89/G2)

After de-embedding, produce the DUT's complex impedance vs frequency and package it as
**IS_Analysis `ISdata` objects**, so every IS plot/fit function applies unchanged.

- **Conversion**: from the de-embedded two-port, extract the device impedance
  consistent with the CPW configuration (series element in a matched Z₀ = 50 Ω line:
  Z_DUT = 2·Z₀·(1 − S₂₁)/S₂₁ — verify this against the ABCD representation you already
  compute; if the ABCD path gives Z directly (B element of a series network), prefer
  that and say so in the report). One function, documented assumptions (Z₀, reference
  planes, config).
- **Packaging**: build `ISdata` instances (import from `../IS_Analysis/IS_Import.py` —
  choose a clean import mechanism and document it): `Zabsphi` = (frequency, |Z|, φ);
  metadata mapped — wafer → `wafer` ('CPW-W2'), row/col → `device_name` (e.g.
  'r2_c1'), state → `res_state` (pristine/HRS/LRS_n as in filenames), `instrument` =
  'vna', DC offset where a bias series, area where known (FIB-trimmed areas differ —
  leave `area` None with a TODO where unknown; do NOT guess). Set `C_pad = 0` (no pad
  on CPW) and leave `C_vac` at a placeholder with a TODO comment (the paper chat will
  supply per-device geometry for normalised views).
- **Acceptance**: a demo script/notebook cell that loads the Wafer2 r2_c1 state series
  (`CPW_mem_oscillator_220524` holds it), de-embeds, converts, and renders (a) a Bode
  |Z|+φ per-state family and (b) a Nyquist, BOTH via IS_Analysis's `IS_plot`, exported
  SVG+TIFF to a scratch folder. That demo is the definition of done.

## 3. Work item B — de-embedding shown, not asserted (Kasmi-style QA)

A validation function/panel: raw vs reference vs de-embedded overlay for a chosen
device + a **residual figure-of-merit per frequency band** against a known standard
(use the r10 thru/open/short/line set and/or ISS thrus in `mag_angle_260424`), with a
one-line residual attribution printed. Output = one methods-grade figure + the FOM
table (CSV or printed). Also: a calibration-comparison panel (LRRM vs LRM+ where both
exist in `mag_angle_260424`).

## 4. Report back

Write `claude_code_report_p4_v1.md` at this repo's root: what was built, signatures,
one example call each, the conversion formula/path actually used and its validation
numbers (FOM per band), demo output paths, deviations, limitations, what was NOT done.
The paper chat reads only that file.

## 5. Notes and cautions

- Bias-T data: V1c series sub-~100 MHz is invalid (bias-T floor); V5 (67 GHz session)
  first block applied no actual bias and only S11 is meaningful after the re-rig —
  carry these as warnings in any demo involving those sets, do not silently drop data.
- Fourier filter functions exist (`fourier_filter/convolve/inverse`) — leave them as
  is; not needed for the bridge.
- The VNA network-analysis method is valid ≥ ~2.6 MHz for our impedance range; the
  bridge should carry frequency bounds so downstream plots can shade validity.
