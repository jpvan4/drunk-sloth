#!/usr/bin/env python3
import sys
import math
import re


def load_fplll_like_basis(path):
    """
    Load a Kannan/fplll-style .basis file that looks like:
      [[a b c ...]
       [d e f ...]
       ...
      ]
    We ignore [, ], commas and just extract integer literals per line.
    """
    rows = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            # Extract all integer tokens (handles [[...]], negative numbers, etc.)
            nums = re.findall(r"-?\d+", line)
            if not nums:
                continue  # skip lines with no numbers

            row = [int(tok) for tok in nums]
            rows.append(row)

    if not rows:
        raise ValueError(f"No numeric rows found in {path}")

    # Make it strictly rectangular (pad with zeros if necessary)
    max_len = max(len(r) for r in rows)
    rect_rows = []
    for i, r in enumerate(rows, start=1):
        if len(r) < max_len:
            r = r + [0] * (max_len - len(r))
        elif len(r) > max_len:
            raise ValueError(
                f"Row {i} has {len(r)} entries, expected {max_len} – "
                f"this is unusual for a lattice basis"
            )
        rect_rows.append(r)

    return rect_rows


def choose_scale_factor(max_abs, target_max=9_000_000_000_000_000_000):
    """
    Choose a power-of-10 scale factor so that:
        max_abs // scale <= target_max
    """
    if max_abs <= target_max:
        return 1

    scale = 1
    # Increase scale by powers of 10 until the scaled max fits in i64
    while max_abs // scale > target_max:
        scale *= 10

    return scale


def scale_basis(rows):
    """
    Scale the entire basis so all entries fit in i64.
    Returns (scaled_rows, scale_factor).
    """
    max_abs = max(abs(v) for row in rows for v in row)
    scale = choose_scale_factor(max_abs)

    if scale == 1:
        return rows, 1

    scaled = []
    for row in rows:
        # Use floor division; you can switch to round(...) if you want
        scaled_row = [int(v // scale) for v in row]
        scaled.append(scaled_row)

    return scaled, scale


def write_lattice_file(path, rows):
    """
    Write in the format your Rust loader expects:
      first line: "<rows> <cols>"
      then rows of integers separated by spaces.
    """
    n_rows = len(rows)
    n_cols = len(rows[0]) if n_rows > 0 else 0

    with open(path, "w") as f:
        f.write(f"{n_rows} {n_cols}\n")
        for r in rows:
            f.write(" ".join(str(v) for v in r) + "\n")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} INPUT.basis OUTPUT.basis")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    print(f"[scale_basis] Loading {in_path} ...")
    rows = load_fplll_like_basis(in_path)

    print(f"[scale_basis] Loaded {len(rows)} rows with {len(rows[0])} columns")
    max_abs = max(abs(v) for row in rows for v in row)
    print(f"[scale_basis] Max |entry| before scaling: {max_abs}")

    scaled, scale = scale_basis(rows)
    if scale != 1:
        print(f"[scale_basis] Applied scale factor 1/{scale}")
        print(
            "[scale_basis] NOTE: The lattice is now scaled down. "
            "Short vectors are preserved up to this global factor."
        )
    else:
        print("[scale_basis] No scaling needed; all entries already fit in i64")

    write_lattice_file(out_path, scaled)
    print(f"[scale_basis] Wrote scaled basis to {out_path}")


if __name__ == "__main__":
    main()
