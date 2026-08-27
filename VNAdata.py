"""VNAdata — container and batch loader for S-parameter measurements.

Replicates the ISdata/ImpedanceData pattern from IS_Analysis (interview
2026-08-08): a thin container over the existing `S2PFile` objects with
filtering, grouping and publication export, so notebooks stop hand-slicing
lists. Builds ON `function_store.import_data` — the parsing, chronological
ordering and state-keyword logic stay exactly where they are.

Data source: the legacy `Antonio Lombardo/CPW Memristor 1` folders, read
as-is (agreed legacy-exempt; nothing there is renamed).

    from VNAdata import VNAdata
    vna = VNAdata.load_batch("/path/to/CPW mem oscillator 220524")
    formed = vna.filter(wafer=2, state="formed")
    for m in formed:  print(m.label)
    vna.summary()
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from function_store import S2PFile, import_data


class VNAdata:
    """Ordered collection of S2PFile measurements with metadata filtering."""

    def __init__(self, measurements=None, source=""):
        self.measurements: list[S2PFile] = list(measurements or [])
        self.source = source

    # -- construction -------------------------------------------------------
    @classmethod
    def load_batch(cls, *data_paths: str) -> "VNAdata":
        """Import every .s2p under each path (chronological within a path).

        Accepts several campaign folders; run numbers restart per folder as
        import_data assigns them, so `source_dir` disambiguates.
        """
        out = cls(source=";".join(str(p) for p in data_paths))
        for p in data_paths:
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(p)
            batch = import_data(str(p))
            for m in batch:
                m.source_dir = p.name          # attach provenance
            out.measurements.extend(batch)
        return out

    # -- access --------------------------------------------------------------
    def __iter__(self):
        return iter(self.measurements)

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, i):
        return self.measurements[i]

    def filter(self, wafer=None, state=None, row=None, col=None,
               runs=None, exclude_runs=None, device=None) -> "VNAdata":
        """Subset by metadata. `runs` / `exclude_runs` are iterables of run
        numbers; `device` is an (row, col) tuple; `state` matches the start of
        the state string so state='formed' catches formed3 etc."""
        keep = []
        for m in self.measurements:
            if wafer is not None and m.wafer_number != wafer:
                continue
            if state is not None and not (m.state or "").startswith(state.lower()):
                continue
            if row is not None and m.dev_row != row:
                continue
            if col is not None and m.dev_col != col:
                continue
            if device is not None and (m.dev_row, m.dev_col) != tuple(device):
                continue
            if runs is not None and m.run not in set(runs):
                continue
            if exclude_runs is not None and m.run in set(exclude_runs):
                continue
            keep.append(m)
        return VNAdata(keep, source=self.source)

    def devices(self):
        """Sorted unique (wafer, row, col) triples present."""
        return sorted({(m.wafer_number, m.dev_row, m.dev_col)
                       for m in self.measurements})

    def states(self) -> Counter:
        return Counter((m.state or "?") for m in self.measurements)

    def summary(self) -> str:
        lines = ["%d measurements from %s" % (len(self), self.source or "-")]
        lines.append("wafers: %s" %
                     dict(Counter(m.wafer_number for m in self.measurements)))
        lines.append("states: %s" % dict(self.states()))
        lines.append("devices: %s" % self.devices())
        text = "\n".join(lines)
        print(text)
        return text

    # -- export --------------------------------------------------------------
    def save_figure(self, fig, output_stem, **kw):
        """Matched SVG+TIFF pair via the shared style contract."""
        from plot_style import save_figure as _sf
        return _sf(fig, output_stem, **kw)
