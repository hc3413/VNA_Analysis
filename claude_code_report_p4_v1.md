# claude_code_report_p4_v1.md — VNA_Analysis Phase-4 bridge report (2026-08-26)

*Report back for this repo's `claude_code_brief_p4_v1.md`. All work is on branch
**`p4-bridge`** (3 commits on top of `main`; `main` and every .s2p/.S2P file
untouched). Companion work in `IS_Analysis` (branch `p4-tooling`) was done first, so
NRMSE reporting there is consistent; nothing here depends on it beyond the shared
`ISdata` container. Demo exports are matched SVG+TIFF pairs in `Output_p4_scratch/`
(git-ignored).*

Demo scripts (run from this repo's root, **with the combined environment**):
`../IS_Analysis/ISvenv/bin/python p4_demo_bridge.py` and `p4_demo_deembed_qa.py`.
(Reason: the bridge needs skrf AND the IS_Analysis stack in one process. `ISvenv` now
carries `scikit-rf==1.0.0` — pinned to the same version as `VNAenv`; the ABCD numbers
were verified identical under both. `VNAenv` itself is unchanged.)

---

## 1. Item A — the VNA→ISdata bridge (DONE)

New module **`vna_bridge.py`** (builds ON `function_store`; raw folders read-only):

| Function | Signature |
|---|---|
| `series_impedance_abcd` | `series_impedance_abcd(network) -> Z_complex[f]` |
| `series_impedance_s21` | `series_impedance_s21(network) -> Z_complex[f]` |
| `conversion_agreement` | `conversion_agreement(network, band=(2.6e6, None)) -> (median_rel, max_rel)` |
| `deembed_copies` | `deembed_copies(s2p_files, abcd_cal) -> [S2PFile]` (deep-copies, then reuses the existing `deembed_ABCD` unchanged — raw imports are never mutated) |
| `s2p_to_isdata` | `s2p_to_isdata(s2p, method='abcd_series', C_vac=None, C_pad=0.0, valid_band=(2.6e6, None)) -> ISdata` |
| `bridge_batch` | `bridge_batch(s2p_files, abcd_cal=None, **kw) -> [ISdata]` |
| `parse_dc_offset` | `parse_dc_offset(filename) -> float or None` (`pos0.8dc`/`neg1.2dc`/`_0dc`/`Vdc0`/`Vdcneg0.4` tokens) |

**Conversion formula/path actually used:** the **ABCD B element**. The de-embedding is
the repo's existing ABCD path (`calibration_ABCD` → `X = sqrt(A_thru)^-1`,
`A_DUT = X·A_meas·X`), which puts the reference planes at the on-wafer thru centre =
the DUT gap. The memristor bridges the CPW signal-line gap, i.e. a series element with
`A = [[1, Z],[0, 1]]`, so **`Z_DUT = A_DUT[:, 0, 1]` directly** — preferred over the
`Z = 2·Z₀·(1−S₂₁)/S₂₁` shortcut exactly as the brief anticipated, because it needs no
matched-load assumption beyond the cascade itself. The S₂₁ shortcut is implemented as a
cross-check: on the de-embedded r2_c1 series the two agree to **median 2.6 %** over the
valid band, worst single point 59 % (worst points sit where S₂₁ → 1 and both estimators
lose leverage — near the high-f end for high-Z states). One function each, assumptions
documented in the module docstring (Z₀ = 50 Ω system as recorded per file, series
configuration, thru-centre reference planes).

**Packaging / metadata mapping (as briefed):** `Zabsphi=(f, |Z|, φ°)`; wafer →
`'CPW-W2'` (generic `CPW-W<n>`), row/col → `device_name='r2_c1'`, state → `res_state`
verbatim from filenames (`pristine`/`formed`/`set`/`set2`…/`reset`…), `instrument='vna'`,
`DC_offset` from the bias token where the file is a bias point, `run_number` = the
chronological run. `area=None` with a TODO (FIB-trimmed areas differ — not guessed).
`C_pad=0` (no pad on CPW). `C_vac` = placeholder (ISdata's 20 µm/30 nm default) with a
TODO for the paper chat's per-device geometry — permittivity/modulus views are NOT
quantitative until that lands. Full band kept (9 kHz–20 GHz); the validity floor is
carried two ways: `instrument='vna'` (IS_Analysis's `IS_plot_stitch` auto-shades
< 2.6 MHz for VNA traces) and a `valid_band_Hz=(2.6e6, None)` attribute on each
instance. All derived arrays (`Zrealimag`, permittivity, modulus, …) are populated via
`transform_measurement_data`, so every IS plot/fit function applies unchanged.

**Acceptance demo (`p4_demo_bridge.py`) — definition of done, delivered:** loads
`CPW_mem_oscillator_220524/Memristor` (69 files) through `VNAdata.load_batch`/
`import_data` + `duplicate_check`, builds the ABCD cal from the four tapered r10 thrus
of the SAME session, de-embeds the 46-spectrum r2_c1 series, bridges to ISdata, and
renders via **IS_Analysis's `IS_plot`**:

- `Output_p4_scratch/CPWW2_r2c1_states_Zabsphi.{svg,tiff}` — Bode |Z|+φ family
  (pristine / formed 0 V / set / reset, full band).
- `Output_p4_scratch/CPWW2_r2c1_states_nyquist.{svg,tiff}` — Nyquist via
  `IS_plot(..., 'colecole')`; displayed from 10 MHz (on top of the 2.6 MHz floor these
  high-Z states are at the method's resolution ceiling until ~10 MHz).
- Bonus: `CPWW2_r2c1_formed_DCladder_Zabsphi.{svg,tiff}` — formed-state DC ladder
  (coolwarm by bias, ≥2.6 MHz), showing `DC_offset` flows into the IS colour semantics.

Headline physics from the demo (screening): all four r2_c1 states **overlap** above
~10 MHz on the capacitor line (C ≈ 0.8 pF) — the wafer-2 rough-substrate control result
the manuscript expects (pristine ≈ formed, no TiOx signature); under negative DC the
formed device's phase departs −90° between ~3–100 MHz.

## 2. Item B — de-embedding shown, not asserted (DONE)

Also in `vna_bridge.py`:

| Function | Signature |
|---|---|
| `deembed_validation` | `deembed_validation(raw_std, deembedded_std, label, bands=QA_BANDS_HZ, export_stem=None) -> (fig, rows)` |
| `calibration_comparison` | `calibration_comparison(net_a, net_b, labels=('LRRM','LRM+'), bands=..., export_stem=None) -> (fig, rows)` |

Method: hold a standard OUT of the calibration, de-embed it, compare to its ideal.
After perfect de-embedding a thru is the identity two-port, so the per-band RMS
**error-vector magnitude |S₂₁ − 1|** is the de-embedding error itself.

**Validation numbers** (`p4_demo_deembed_qa.py`; cal from r10 c2+c3 thrus of the
dedicated `CPW_mem_James_161024` set, applied to the held-out c1 thru; that session
reaches 67 GHz):

| Band | EVM raw | EVM de-embedded | Improvement |
|---|---|---|---|
| 2.6–100 MHz | 0.0038 | 0.0053 | ×0.7 (N=5 — both tiny; below the raw error floor) |
| 100 MHz–1 GHz | 0.0139 | 0.0036 | ×3.8 |
| 1–5 GHz | 0.0706 | 0.0086 | ×8.2 |
| 5–10 GHz | 0.1631 | 0.0111 | ×14.7 |
| 10–20 GHz | 0.3195 | 0.0095 | ×33.5 |
| 20–67 GHz | 0.7830 | 0.0132 | ×59.2 |

One-line residual attribution (printed by the function): *residual is
MAGNITUDE-dominated (0.008 rms vs 0.25° rms) → loss/contact repeatability*, i.e. the
remaining ~1 % error is probe-contact/loss variation between nominally identical thrus,
not a reference-plane error. Outputs:
`CPWW2_r10_deembed_validation_James161024.{svg,tiff}` (methods-grade
raw/de-embedded/ideal panel, invalid band hatched) + `..._FOM.csv` (full table incl.
magnitude/phase residual split).

**Calibration-comparison panel (LRRM vs LRM+)**: same physical ISS thru measured under
both calibrations in `mag_angle_260424` (`ISS_thru_LRRM_1` vs
`Wafer0_r0_c0_ISS_thru_LRM+_1`; grids interpolated where they differ). Per-band rms
|ΔS₂₁|: 9.6e-5 (2.6–100 MHz) → 2.7e-4 (0.1–1 GHz) → 1.3e-3 (1–5 GHz) → 3.2e-3
(5–10 GHz) → 6.2e-3 (10–20 GHz). The calibration choice moves S₂₁ by **< 0.7 % worst
case** — far below the state contrasts of interest. Outputs:
`ISS_thru_LRRM_vs_LRMplus_260424.{svg,tiff}` + `..._diff.csv`.

## 3. Supporting changes

- `function_store.import_data` keyword-order fix: bare `reset` files were labelled
  `set` ('set' is a substring of 'reset' and was checked first). `reset` now precedes
  `set`; numbered tokens (`reset2`, `set3`, …) were already correct.
- The pre-existing working-tree changes (shared style-contract enforcement in
  `keyplot`/`sub_plot`, SVG text-as-text, `use_tex=False` defaults) and the previously
  untracked `VNAdata.py` container are committed unchanged as the branch's first commit.
- `requirements.txt` notes the combined-environment requirement for bridge scripts.
- `../IS_Analysis/ISvenv` gained `scikit-rf==1.0.0` (the one environment change;
  version-matched to `VNAenv`, numerically verified identical on the ABCD path).

## 4. Deviations, limitations, cautions

- **Import-order note**: both repos own a `plot_style` module. Bridge sessions append
  (not prepend) `../IS_Analysis` to `sys.path`, so the VNA `plot_style` wins — same
  `save_figure`/`set_plot_style` contract, but IS-only extras (e.g. `STATE_COLORS`)
  must be imported from the IS file explicitly if ever needed in a bridge session.
- The 2.6 MHz validity floor is a frequency bound; in practice resolution also depends
  on |Z| — the pristine/HRS-like states are noise-limited up to ~10 MHz (visible in the
  Bode demo; the Nyquist display window states this). The bridge drops nothing.
- `valid_band_Hz` is a plain attribute on the ISdata instances (adding a real dataclass
  field belongs to IS_Analysis; not done unilaterally). It does not survive pickling
  conventions the repos don't use anyway.
- Brief §5 cautions carried, not tested here: the V1c bias-T series (<~100 MHz invalid)
  and the V5/67 GHz first-block bias issue (only S11 meaningful after the re-rig) — the
  demos deliberately use neither; the James r2_c1 `formed_Vdc*` files (same 67 GHz
  session) were NOT bridged for that reason. When you bridge them, filter to S11-based
  quantities or exclude the first block.
- 2.6–100 MHz FOM band has N=5 points in the James set (sweep starts ~10 MHz) — the
  ×0.7 there compares two numbers already at the error floor, not a de-embedding failure.
- `mag_angle_240424`/`260424` device series (r1 row, HRS/LRS-named files) were not
  bridged in the demo — the bridge is generic (`bridge_batch` on any imported set), but
  note `import_data` has no 'hrs'/'lrs' keywords, so those files come through with
  `state=None` until a keyword extension is agreed (kept out of scope to avoid touching
  the shared keyword list twice in one phase).
- `fourier_filter/convolve/inverse` untouched, as instructed.

## 5. NOT done

- No skrf-2.x migration (both envs pinned at 1.0.0 behaviour).
- No line/linelong-based (TRL-style) validation — the FOM uses thru standards only;
  the line data sits ready in the James set if wanted.
- No 67 GHz-session device bridging (see the V5 caution above).
- No per-device C_vac/area table (paper chat supplies; placeholders + TODOs in place).
- Raw data folders untouched; notebooks untouched; `main` untouched.
