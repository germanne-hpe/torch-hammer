#!/usr/bin/env python3
"""
torch-hammer-reporter
Generates an HTML benchmark report from torch-hammer CSV output files.

Usage:
    # Single merged CSV (with hostname column):
    python hammer_report.py results.csv

    # Directory of per-node CSV files:
    python hammer_report.py /path/to/results/dir/

    # Explicit output path:
    python hammer_report.py results.csv -o report.html
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GPUResult:
    gpu: int
    gpu_model: str
    serial: str
    benchmark: str
    dtype: str
    iterations: int
    runtime_s: float
    min: float
    mean: float
    max: float
    unit: str
    power_avg_w: float
    temp_max_c: float


@dataclass
class NodeResult:
    hostname: str
    gpus: list[GPUResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.gpus) > 0

    @property
    def mean_values(self) -> list[float]:
        return [g.mean for g in self.gpus]

    @property
    def node_mean(self) -> float:
        v = self.mean_values
        return sum(v) / len(v) if v else 0.0

    @property
    def intra_spread_pct(self) -> float:
        v = self.mean_values
        if len(v) < 2:
            return 0.0
        return (max(v) - min(v)) / self.node_mean * 100

    @property
    def unit(self) -> str:
        return self.gpus[0].unit if self.gpus else "GFLOP/s"

    @property
    def benchmark(self) -> str:
        return self.gpus[0].benchmark if self.gpus else "—"

    @property
    def dtype(self) -> str:
        return self.gpus[0].dtype if self.gpus else "—"

    @property
    def gpu_model(self) -> str:
        return self.gpus[0].gpu_model if self.gpus else "—"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

EXPECTED_COLS = {
    "hostname", "gpu", "gpu_model", "serial", "benchmark", "dtype",
    "iterations", "runtime_s", "min", "mean", "max", "unit",
    "power_avg_w", "temp_max_c",
}


def _parse_row(row: dict) -> Optional[GPUResult]:
    try:
        return GPUResult(
            gpu=int(row["gpu"]),
            gpu_model=row["gpu_model"].strip(),
            serial=row["serial"].strip(),
            benchmark=row["benchmark"].strip(),
            dtype=row["dtype"].strip(),
            iterations=int(row["iterations"]),
            runtime_s=float(row["runtime_s"]),
            min=float(row["min"]),
            mean=float(row["mean"]),
            max=float(row["max"]),
            unit=row["unit"].strip(),
            power_avg_w=float(row["power_avg_w"]),
            temp_max_c=float(row["temp_max_c"]),
        )
    except (KeyError, ValueError):
        return None


def load_csv(path: Path, hostname_override: Optional[str] = None) -> dict[str, NodeResult]:
    nodes: dict[str, NodeResult] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if not EXPECTED_COLS.issubset(cols) and "hostname" not in cols and hostname_override is None:
            print(f"  Warning: {path.name} missing expected columns", file=sys.stderr)
        for row in reader:
            hostname = hostname_override or row.get("hostname", path.stem).strip()
            if hostname not in nodes:
                nodes[hostname] = NodeResult(hostname=hostname)
            result = _parse_row(row)
            if result:
                nodes[hostname].gpus.append(result)
    return nodes


def load_inputs(source: Path) -> dict[str, NodeResult]:
    nodes: dict[str, NodeResult] = {}
    if source.is_dir():
        csv_files = sorted(source.glob("*.csv"))
        if not csv_files:
            # Try bare files (no extension) like the example output
            bare = [p for p in sorted(source.iterdir()) if p.is_file() and not p.suffix]
            csv_files = bare
        for f in csv_files:
            partial = load_csv(f, hostname_override=f.stem)
            nodes.update(partial)
        # Also check for per-node files without header (like the original shell output)
        # where the file IS the node name and contains a CSV with hostname column
        if not nodes:
            for f in sorted(source.iterdir()):
                if f.is_file():
                    partial = load_csv(f)
                    nodes.update(partial)
    else:
        nodes = load_csv(source)
    return nodes


def load_shell_output(source: Path) -> dict[str, NodeResult]:
    """
    Parse the raw output of:
        for file in *; do echo "file: $file"; cat $file; done
    where each section is 'file: <nodename>' followed by CSV lines.
    """
    nodes: dict[str, NodeResult] = {}
    current_host: Optional[str] = None
    current_rows: list[str] = []
    header: Optional[str] = None

    def flush():
        nonlocal header
        if current_host is None:
            return
        nodes[current_host] = NodeResult(hostname=current_host)
        if header and current_rows:
            import io
            text = header + "\n" + "\n".join(current_rows)
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                r = _parse_row(row)
                if r:
                    nodes[current_host].gpus.append(r)

    with source.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("file: "):
                flush()
                current_host = line[6:].strip()
                current_rows = []
                header = None
            elif current_host is not None:
                if header is None and line.startswith("hostname"):
                    header = line
                elif header and line:
                    current_rows.append(line)

    flush()
    return nodes


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_stats(nodes: dict[str, NodeResult]) -> dict:
    healthy = [n for n in nodes.values() if n.ok]
    bad = [n for n in nodes.values() if not n.ok]

    all_means = [n.node_mean for n in healthy]
    global_min = min(all_means) if all_means else 0
    global_max = max(all_means) if all_means else 0
    inter_spread = (global_max - global_min) / ((global_min + global_max) / 2) * 100 if global_min else 0

    worst_intra = max((n.intra_spread_pct for n in healthy), default=0)

    return {
        "healthy_count": len(healthy),
        "bad_count": len(bad),
        "inter_spread_pct": inter_spread,
        "worst_intra_pct": worst_intra,
        "global_min": global_min,
        "global_max": global_max,
        "healthy": healthy,
        "bad": bad,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

GPU_COLORS = ["#378ADD", "#1D9E75", "#BA7517", "#D4537E",
              "#534AB7", "#D85A30", "#639922", "#E24B4A"]


def _fmt(v: float, decimals: int = 0) -> str:
    if decimals == 0:
        return f"{v:,.0f}"
    return f"{v:,.{decimals}f}"


def build_chart_datasets(healthy: list[NodeResult]) -> tuple[list[str], str]:
    gpu_indices = sorted({g.gpu for n in healthy for g in n.gpus})
    labels = [n.hostname for n in healthy]

    datasets = []
    for i, gpu_idx in enumerate(gpu_indices):
        color = GPU_COLORS[i % len(GPU_COLORS)]
        data = []
        for n in healthy:
            gpu = next((g for g in n.gpus if g.gpu == gpu_idx), None)
            data.append(round(gpu.mean, 1) if gpu else None)
        datasets.append({
            "label": f"GPU {gpu_idx}",
            "data": data,
            "backgroundColor": color,
            "borderRadius": 3,
            "barPercentage": 0.75,
        })

    return labels, json.dumps(datasets)


def build_node_table_rows(healthy: list[NodeResult]) -> str:
    rows = []
    for n in sorted(healthy, key=lambda x: x.hostname):
        spread_cls = "warn" if n.intra_spread_pct > 3 else ""
        gpu_cells = ""
        for g in sorted(n.gpus, key=lambda x: x.gpu):
            gpu_cells += f'<td class="mono">{_fmt(g.mean)}</td>'
        rows.append(
            f'<tr>'
            f'<td class="mono host">{n.hostname}</td>'
            f'{gpu_cells}'
            f'<td class="mono">{_fmt(n.node_mean)}</td>'
            f'<td class="mono {spread_cls}">{n.intra_spread_pct:.1f}%</td>'
            f'<td class="mono">{_fmt(max(n.mean_values) - min(n.mean_values))}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def build_bad_badges(bad: list[NodeResult]) -> str:
    if not bad:
        return '<span style="color:var(--muted)">none</span>'
    return " ".join(
        f'<span class="badge badge-bad">{n.hostname}</span>'
        for n in bad
    )


def build_legend(healthy: list[NodeResult]) -> str:
    gpu_indices = sorted({g.gpu for n in healthy for g in n.gpus})
    items = []
    for i, idx in enumerate(gpu_indices):
        color = GPU_COLORS[i % len(GPU_COLORS)]
        items.append(
            f'<span class="legend-item">'
            f'<span class="legend-swatch" style="background:{color}"></span>'
            f'GPU {idx}'
            f'</span>'
        )
    return "\n".join(items)


def render_html(nodes: dict[str, NodeResult], source_name: str) -> str:
    stats = compute_stats(nodes)
    healthy: list[NodeResult] = stats["healthy"]
    bad: list[NodeResult] = stats["bad"]

    if not healthy:
        y_min = 0
        y_max = 50000
        unit = "GFLOP/s"
        benchmark = "—"
        dtype = "—"
        gpu_model = "—"
    else:
        all_means = [g.mean for n in healthy for g in n.gpus]
        y_min = max(0, math.floor(min(all_means) / 1000) * 1000 - 1000)
        y_max = math.ceil(max(all_means) / 1000) * 1000 + 500
        unit = healthy[0].unit
        benchmark = healthy[0].benchmark
        dtype = healthy[0].dtype
        gpu_model = healthy[0].gpu_model

    labels, datasets_json = build_chart_datasets(healthy)
    labels_json = json.dumps(labels)
    table_rows = build_node_table_rows(healthy)
    bad_badges = build_bad_badges(bad)
    legend_html = build_legend(healthy)

    # GPU header columns
    gpu_indices = sorted({g.gpu for n in healthy for g in n.gpus})
    gpu_headers = "".join(f"<th>GPU {i}</th>" for i in gpu_indices)

    # inter-node best/worst
    if healthy:
        best_node = max(healthy, key=lambda n: n.node_mean)
        worst_node = min(healthy, key=lambda n: n.node_mean)
        best_label = f"{best_node.hostname} ({_fmt(best_node.node_mean)} {unit})"
        worst_label = f"{worst_node.hostname} ({_fmt(worst_node.node_mean)} {unit})"
    else:
        best_label = worst_label = "—"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>torch-hammer report — {source_name}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root {{
    --bg: #f8f7f4;
    --surface: #ffffff;
    --surface2: #f1efe8;
    --border: rgba(0,0,0,0.10);
    --border2: rgba(0,0,0,0.18);
    --text: #1a1a18;
    --muted: #6b6b65;
    --hint: #9a9a92;
    --success: #0f6e56;
    --success-bg: #e1f5ee;
    --danger: #a32d2d;
    --danger-bg: #fcebeb;
    --warn: #854f0b;
    --warn-bg: #faeeda;
    --info: #185fa5;
    --info-bg: #e6f1fb;
    --radius: 8px;
    --radius-lg: 12px;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #1a1a18;
      --surface: #242422;
      --surface2: #2c2c2a;
      --border: rgba(255,255,255,0.10);
      --border2: rgba(255,255,255,0.18);
      --text: #e8e6df;
      --muted: #9a9a92;
      --hint: #6b6b65;
      --success: #5dcaa5;
      --success-bg: #085041;
      --danger: #f09595;
      --danger-bg: #791f1f;
      --warn: #ef9f27;
      --warn-bg: #633806;
      --info: #85b7eb;
      --info-bg: #0c447c;
    }}
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: ui-monospace, "SF Mono", "Cascadia Code", "Fira Code", monospace;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.6;
  }}
  a {{ color: var(--info); text-decoration: none; }}

  .page {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }}

  /* Header */
  .header {{ margin-bottom: 2rem; border-bottom: 0.5px solid var(--border2); padding-bottom: 1.25rem; }}
  .header h1 {{ font-size: 20px; font-weight: 500; letter-spacing: -0.02em; color: var(--text); }}
  .header .meta {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .header .meta span {{ margin-right: 1.5rem; }}

  /* Metric cards */
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-bottom: 2rem; }}
  .metric-card {{
    background: var(--surface2);
    border-radius: var(--radius);
    padding: 1rem;
  }}
  .metric-card .label {{ font-size: 11px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.06em; }}
  .metric-card .value {{ font-size: 26px; font-weight: 500; letter-spacing: -0.03em; }}
  .metric-card.ok .value {{ color: var(--success); }}
  .metric-card.bad .value {{ color: var(--danger); }}
  .metric-card.warn .value {{ color: var(--warn); }}

  /* Section */
  .section {{ margin-bottom: 2rem; }}
  .section-title {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.75rem; }}

  /* Chart */
  .chart-wrap {{
    background: var(--surface);
    border: 0.5px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.25rem;
  }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 1rem; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); }}
  .legend-swatch {{ width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }}
  .chart-canvas-wrap {{ position: relative; width: 100%; height: 300px; }}

  /* Table */
  .table-wrap {{
    background: var(--surface);
    border: 0.5px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  thead tr {{ background: var(--surface2); }}
  th {{
    text-align: right;
    padding: 10px 14px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 0.5px solid var(--border2);
  }}
  th:first-child {{ text-align: left; }}
  td {{
    padding: 9px 14px;
    text-align: right;
    border-bottom: 0.5px solid var(--border);
    color: var(--text);
  }}
  td:first-child {{ text-align: left; }}
  td.host {{ color: var(--text); font-weight: 500; }}
  td.warn {{ color: var(--warn); font-weight: 500; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}

  /* Bad nodes */
  .badge {{
    display: inline-block;
    font-size: 12px;
    padding: 3px 10px;
    border-radius: var(--radius);
    font-weight: 500;
  }}
  .badge-bad {{ background: var(--danger-bg); color: var(--danger); }}

  /* Inter-node summary */
  .inter-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  .inter-card {{
    background: var(--surface);
    border: 0.5px solid var(--border);
    border-radius: var(--radius);
    padding: 0.875rem 1rem;
  }}
  .inter-card .lbl {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 3px; }}
  .inter-card .val {{ font-size: 14px; font-weight: 500; }}

  .mono {{ font-family: inherit; }}
  .unit {{ font-size: 11px; color: var(--muted); }}

  footer {{ margin-top: 3rem; font-size: 11px; color: var(--hint); border-top: 0.5px solid var(--border); padding-top: 1rem; }}
</style>
</head>
<body>
<div class="page">

  <div class="header">
    <h1>torch-hammer benchmark report</h1>
    <div class="meta">
      <span>source: {source_name}</span>
      <span>benchmark: {benchmark}</span>
      <span>dtype: {dtype}</span>
      <span>GPU: {gpu_model}</span>
      <span>unit: {unit}</span>
    </div>
  </div>

  <div class="metrics">
    <div class="metric-card ok">
      <div class="label">healthy nodes</div>
      <div class="value">{stats["healthy_count"]}</div>
    </div>
    <div class="metric-card bad">
      <div class="label">no results</div>
      <div class="value">{stats["bad_count"]}</div>
    </div>
    <div class="metric-card">
      <div class="label">node-to-node spread</div>
      <div class="value">{stats["inter_spread_pct"]:.1f}%</div>
    </div>
    <div class="metric-card warn">
      <div class="label">worst intra-node spread</div>
      <div class="value">{stats["worst_intra_pct"]:.1f}%</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">mean {unit} per GPU per node</div>
    <div class="chart-wrap">
      <div class="legend">{legend_html}</div>
      <div class="chart-canvas-wrap">
        <canvas id="gpuChart"></canvas>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">per-node detail <span class="unit">(mean {unit})</span></div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>node</th>
            {gpu_headers}
            <th>node avg</th>
            <th>intra spread</th>
            <th>range ({unit})</th>
          </tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <div class="section-title">inter-node comparison</div>
    <div class="inter-grid">
      <div class="inter-card">
        <div class="lbl">best node</div>
        <div class="val">{best_label}</div>
      </div>
      <div class="inter-card">
        <div class="lbl">worst node</div>
        <div class="val">{worst_label}</div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">nodes with no results</div>
    <div>{bad_badges}</div>
  </div>

  <footer>generated by torch-hammer-reporter &nbsp;|&nbsp; nodes: {stats["healthy_count"] + stats["bad_count"]} total</footer>

</div>

<script>
const labels = {labels_json};
const datasets = {datasets_json};

new Chart(document.getElementById('gpuChart'), {{
  type: 'bar',
  data: {{ labels, datasets }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y.toLocaleString(undefined, {{maximumFractionDigits: 0}})}} {unit}`
        }}
      }}
    }},
    scales: {{
      x: {{
        ticks: {{ font: {{ family: 'ui-monospace, "SF Mono", monospace', size: 11 }}, autoSkip: false }},
        grid: {{ color: 'rgba(128,128,128,0.08)' }}
      }},
      y: {{
        min: {y_min},
        max: {y_max},
        ticks: {{
          font: {{ family: 'ui-monospace, "SF Mono", monospace', size: 11 }},
          callback: v => (v/1000).toFixed(0) + 'k'
        }},
        grid: {{ color: 'rgba(128,128,128,0.08)' }},
        title: {{ display: true, text: 'mean {unit}', font: {{ size: 11 }} }}
      }}
    }}
  }}
}});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML benchmark report from torch-hammer CSV output."
    )
    parser.add_argument(
        "source",
        help="Path to a CSV file, a directory of CSV files, or a shell output file.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output HTML file path (default: <source>.html or report.html)",
    )
    parser.add_argument(
        "--shell-output",
        action="store_true",
        help="Parse a raw 'for file in *; do echo file: $f; cat $f; done' shell dump.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Error: {source} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {source}")

    if args.shell_output:
        nodes = load_shell_output(source)
    elif source.is_dir():
        nodes = load_inputs(source)
    else:
        # Heuristic: if the file contains "file: nid" lines, treat as shell output
        with source.open() as f:
            first = f.read(512)
        if "file: " in first and "hostname,gpu" in first:
            nodes = load_shell_output(source)
        else:
            nodes = load_inputs(source)

    if not nodes:
        print("Error: no data found", file=sys.stderr)
        sys.exit(1)

    healthy = sum(1 for n in nodes.values() if n.ok)
    bad = sum(1 for n in nodes.values() if not n.ok)
    print(f"  Found {len(nodes)} nodes: {healthy} healthy, {bad} with no results")

    html = render_html(nodes, source.name)

    if args.output:
        out = Path(args.output)
    elif source.is_dir():
        out = source / "report.html"
    else:
        out = source.with_suffix(".html")

    out.write_text(html, encoding="utf-8")
    print(f"  Report written to: {out}")


if __name__ == "__main__":
    main()
