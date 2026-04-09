# torch-hammer-reporter

Generates a self-contained HTML benchmark report from `torch-hammer` CSV output.  
No dependencies beyond the Python standard library.

## Usage

```bash
# Raw shell dump (from: for file in *; do echo "file: $file"; cat $file; done)
python hammer_report.py shell_dump.txt

# Single merged CSV with a hostname column
python hammer_report.py results.csv

# Directory of per-node CSV files (filename = hostname)
python hammer_report.py /path/to/results/

# Explicit output path
python hammer_report.py results.csv -o my_report.html

# Force shell-output parse mode
python hammer_report.py dump.txt --shell-output
```

The script auto-detects the input format.

## Input formats

### Per-node CSV files (directory)
Each file is named after its node (e.g. `nid005193`) and contains a standard torch-hammer CSV:

```
hostname,gpu,gpu_model,serial,benchmark,dtype,iterations,runtime_s,min,mean,max,unit,power_avg_w,temp_max_c
nid005193,0,NVIDIA GH200 120GB,1654123074012,...
```

Nodes with only a header and no data rows are reported as **no results**.

### Merged CSV (single file)
All rows in one file, with a `hostname` column to distinguish nodes.

### Shell dump (auto-detected)
Raw output of:
```bash
for file in *; do echo "file: $file"; cat $file; done
```
The script detects the `file: <name>` markers and splits accordingly.

## Report contents

- **Summary metrics**: healthy vs. no-result node counts, inter-node spread %, worst intra-node spread %
- **Bar chart**: mean GFLOP/s per GPU, grouped by node (Chart.js, embedded CDN)
- **Detail table**: per-GPU mean values, node average, intra-node spread, range
- **Inter-node comparison**: best and worst performing healthy nodes
- **Bad node list**: nodes with no benchmark results

## Output

A single self-contained `.html` file. Opens in any browser, no server needed.  
Dark mode is supported via `@media (prefers-color-scheme: dark)`.
