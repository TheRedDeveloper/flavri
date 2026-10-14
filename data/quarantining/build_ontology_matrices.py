"""
data/quarantining/build_ontology_matrices.py
Extracts data-driven descriptor Hierarchy (Parent-Child) and Antagonism (Mutual Exclusion)
matrices directly from singles_train.csv.

Saves:
- data/quarantining/hierarchy_matrix.pt  (137, 137) float32 tensor: H[parent, child] = 1.0
- data/quarantining/antagonism_matrix.pt (137, 137) float32 tensor: A[i, j] = 1.0
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flavri.constants import ODOR_CHANNELS


def main():
    print("=" * 70)
    print("BUILDING DATA-DRIVEN DESCRIPTOR ONTOLOGY MATRICES")
    print("=" * 70)

    csv_path = REPO_ROOT / "data" / "quarantining" / "singles_train.csv"
    print(f"Reading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    n_samples = len(df)
    n_channels = len(ODOR_CHANNELS)
    print(f"Loaded {n_samples:,} molecules across {n_channels} odor channels.")

    target_cols = [f"y_{ch}" for ch in ODOR_CHANNELS]
    mask_cols = [f"mask_{ch}" for ch in ODOR_CHANNELS]

    y_mat = (df[target_cols].to_numpy() >= 0.5).astype(int)
    mask_mat = (df[mask_cols].to_numpy() > 0).astype(int)
    active_y = y_mat * mask_mat  # (N, 137)

    # Frequency of each descriptor
    counts = active_y.sum(axis=0)  # (137,)

    print("\nDescriptor frequency summary:")
    for i, ch in enumerate(ODOR_CHANNELS):
        if counts[i] >= 500:
            print(f"  {ch:<18}: {counts[i]:5d} ({counts[i]/n_samples:.2%})")

    # -------------------------------------------------------------------------
    # 1. Hierarchy Matrix: H[parent, child] = 1.0
    # -------------------------------------------------------------------------
    # Condition:
    # 1. Child count >= 15
    # 2. P(Parent | Child) >= 0.80
    # 3. Parent count > Child count
    # -------------------------------------------------------------------------
    H = np.zeros((n_channels, n_channels), dtype=np.float32)
    hierarchy_pairs = []

    for child_idx in range(n_channels):
        n_child = counts[child_idx]
        if n_child < 15:
            continue

        child_mask = active_y[:, child_idx] == 1
        for parent_idx in range(n_channels):
            if child_idx == parent_idx:
                continue

            n_parent = counts[parent_idx]
            if n_parent <= n_child:
                continue

            # Joint occurrences
            joint = (active_y[child_mask, parent_idx] == 1).sum()
            cond_prob = joint / n_child

            if cond_prob >= 0.80:
                H[parent_idx, child_idx] = 1.0
                hierarchy_pairs.append((
                    ODOR_CHANNELS[parent_idx],
                    ODOR_CHANNELS[child_idx],
                    cond_prob,
                    n_parent,
                    n_child,
                ))

    print(f"\nDiscovered {len(hierarchy_pairs)} Hierarchy (Parent-Child) relationships (P >= 0.80):")
    for parent, child, p, np_cnt, nc_cnt in sorted(hierarchy_pairs, key=lambda x: -x[2])[:30]:
        print(f"  [Parent] {parent:<16}  <---  [Child] {child:<16}  (P = {p:.1%}, N_p={np_cnt}, N_c={nc_cnt})")

    # -------------------------------------------------------------------------
    # 2. Antagonism Matrix: A[i, j] = 1.0 (Mutual Exclusion)
    # -------------------------------------------------------------------------
    # Condition for sparse multi-label mutual exclusion:
    # 1. Both descriptors frequent: N_i >= 40, N_j >= 40
    # 2. Observed joint co-occurrence == 0
    # 3. Expected joint co-occurrence under independence >= 2.5
    # -------------------------------------------------------------------------
    A = np.zeros((n_channels, n_channels), dtype=np.float32)
    antagonism_pairs = []

    for i in range(n_channels):
        if counts[i] < 40:
            continue
        for j in range(i + 1, n_channels):
            if counts[j] < 40:
                continue

            joint = (active_y[:, i] * active_y[:, j]).sum()
            expected = (counts[i] * counts[j]) / n_samples

            if joint == 0 and expected >= 2.5:
                A[i, j] = 1.0
                A[j, i] = 1.0
                antagonism_pairs.append((
                    ODOR_CHANNELS[i],
                    ODOR_CHANNELS[j],
                    joint,
                    expected,
                ))

    print(f"\nDiscovered {len(antagonism_pairs)} Antagonistic (Mutually Exclusive) pairs (Joint == 0, Expected >= 2.5):")
    for ch_a, ch_b, joint, exp in sorted(antagonism_pairs, key=lambda x: -x[3])[:30]:
        print(f"  [Antagonist] {ch_a:<16}  <--X-->  {ch_b:<16}  (Observed: {joint}, Expected: {exp:.1f})")

    # Save tensors
    out_dir = REPO_ROOT / "data" / "quarantining"
    h_path = out_dir / "hierarchy_matrix.pt"
    a_path = out_dir / "antagonism_matrix.pt"

    torch.save(torch.tensor(H, dtype=torch.float32), str(h_path))
    torch.save(torch.tensor(A, dtype=torch.float32), str(a_path))

    print("\nSaved ontology matrices:")
    print(f"  Hierarchy:  {h_path} (shape: {H.shape}, active links: {int(H.sum())})")
    print(f"  Antagonism: {a_path} (shape: {A.shape}, active pairs: {len(antagonism_pairs)})")
    print("=" * 70)


if __name__ == "__main__":
    main()
