"""
evaluate_all_versions.py
Evaluates FLAVORI 0.2, FLAVORI 0.3, and FLAVORI 0.4 on the expanded test sets:
- singles_test.csv (2,570 molecules with expanded negatives)
- pair_test.csv (11,037 mixtures)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path("c:/Users/RedDev/clones/flavri")
sys.path.insert(0, str(REPO_ROOT))

from flavri.constants import (
    NUM_CHANNELS,
    NUM_TASTE,
    NUM_ODOR,
    ALL_CHANNELS,
    TASTE_CHANNELS,
    ODOR_CHANNELS,
)
from flavri.data import FlavriSinglesDataset, FlavriPairsDataset, SinglesCollator, PairsCollator
from flavri.featurizer import BatchMolGraph, ATOM_FDIM, BOND_FDIM
from flavri.models.heads import SingleMoleculeHead, CommutativeSynergyHead
from flavri.models.model import FlavriMultimodalModel

# Define FLAVORI 0.2 Architecture Wrapper
class AttentiveGraphPoolingV2(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.w_att = nn.Linear(hidden_dim, hidden_dim)
        self.v_att = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, atom_h: torch.Tensor, a2g: torch.Tensor, batch_size: int) -> torch.Tensor:
        att_scores = self.v_att(torch.tanh(self.w_att(atom_h))).squeeze(-1)
        max_scores = torch.full((batch_size,), -1e9, device=atom_h.device)
        max_scores.scatter_reduce_(0, a2g, att_scores, reduce='amax')
        exp_scores = torch.exp(att_scores - max_scores[a2g])
        sum_exp = torch.zeros(batch_size, device=atom_h.device)
        sum_exp.scatter_add_(0, a2g, exp_scores)
        weights = (exp_scores / (sum_exp[a2g] + 1e-9)).unsqueeze(-1)
        weighted_h = atom_h * weights
        pooled = torch.zeros((batch_size, atom_h.size(-1)), device=atom_h.device)
        pooled.scatter_add_(0, a2g.unsqueeze(-1).expand_as(weighted_h), weighted_h)
        return pooled

class DMPNNBackboneV2(nn.Module):
    def __init__(self, atom_fdim=ATOM_FDIM, bond_fdim=BOND_FDIM, hidden_dim=256, depth=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.W_i = nn.Linear(atom_fdim + bond_fdim, hidden_dim, bias=False)
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_a = nn.Linear(atom_fdim + hidden_dim, hidden_dim)
        self.readout = AttentiveGraphPoolingV2(hidden_dim)

    def forward(self, batch_graph: BatchMolGraph) -> torch.Tensor:
        device = batch_graph.f_atoms.device
        if batch_graph.total_bonds == 0:
            atom_h = torch.relu(self.W_a(torch.cat([
                batch_graph.f_atoms,
                torch.zeros((batch_graph.total_atoms, self.hidden_dim), device=device)
            ], dim=-1)))
            return self.readout(atom_h, batch_graph.a2g, batch_graph.batch_size)

        u_atoms = batch_graph.b2u[1:]
        f_u_atoms = batch_graph.f_atoms[u_atoms]
        f_bonds = batch_graph.f_bonds[1:]
        init_input = torch.cat([f_u_atoms, f_bonds], dim=-1)
        m_0 = torch.relu(self.W_i(init_input))
        zero_row = torch.zeros((1, self.hidden_dim), device=device)
        m = torch.cat([zero_row, m_0], dim=0)

        for _ in range(self.depth - 1):
            incoming_m = m[batch_graph.b2b]
            sum_incoming = incoming_m.sum(dim=1)
            m_update = torch.relu(m_0 + self.W_m(sum_incoming[1:]))
            m_update = self.dropout(m_update)
            m = torch.cat([zero_row, m_update], dim=0)

        incoming_to_atoms = m[batch_graph.a2b]
        sum_m_v = incoming_to_atoms.sum(dim=1)
        atom_input = torch.cat([batch_graph.f_atoms, sum_m_v], dim=-1)
        atom_h = torch.relu(self.W_a(atom_input))
        atom_h = self.dropout(atom_h)
        return self.readout(atom_h, batch_graph.a2g, batch_graph.batch_size)

class FlavriV2Model(nn.Module):
    def __init__(self, hidden_dim=256, depth=4, num_channels=NUM_CHANNELS, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = DMPNNBackboneV2(hidden_dim=hidden_dim, depth=depth, dropout=dropout)
        self.single_head = SingleMoleculeHead(hidden_dim=hidden_dim, num_channels=num_channels, dropout=dropout)
        self.synergy_head = CommutativeSynergyHead(hidden_dim=hidden_dim, num_channels=num_channels, dropout=dropout)
        self.fp_proj = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def encode(self, batch_graph: BatchMolGraph) -> torch.Tensor:
        z_graph = self.backbone(batch_graph)
        if hasattr(batch_graph, 'fps') and batch_graph.fps is not None:
            z_fp = self.fp_proj(batch_graph.fps)
            return self.fusion(torch.cat([z_graph, z_fp], dim=-1))
        return z_graph

    def predict_singles(self, batch_graph: BatchMolGraph):
        z = self.encode(batch_graph)
        return z, self.single_head(z)

    def predict_pairs_batch(self, unique_graphs, idx_a, idx_b, w_a=None, w_b=None):
        z_unique = self.encode(unique_graphs)
        v_unique = self.single_head(z_unique)
        z_a = z_unique[idx_a]
        z_b = z_unique[idx_b]
        v_a = v_unique[idx_a]
        v_b = v_unique[idx_b]
        delta_ab = self.synergy_head(z_a, z_b)
        w_a = torch.full((len(idx_a), 1), 0.5, device=idx_a.device)
        w_b = torch.full((len(idx_b), 1), 0.5, device=idx_b.device)
        y_blend = torch.clamp(w_a * v_a + w_b * v_b + 4.0 * w_a * w_b * delta_ab, 0.0, 1.0)
        return delta_ab, y_blend, v_a, v_b


@torch.no_grad()
def evaluate_singles(model, loader, device):
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    for batch in loader:
        _, preds = model.predict_singles(batch["graphs"])
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch["targets"].cpu().numpy())
        all_masks.append(batch["masks"].cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # Taste channels (0..10)
    taste_aurocs = {}
    for c, ch in enumerate(TASTE_CHANNELS):
        m_c = masks[:, c] > 0
        y_t = targets[m_c, c]
        y_p = preds[m_c, c]
        if m_c.sum() >= 5 and len(np.unique(y_t > 0.5)) > 1:
            try:
                auc = float(roc_auc_score(y_t > 0.5, y_p))
                taste_aurocs[ch] = auc
            except:
                pass
    taste_macro = float(np.mean(list(taste_aurocs.values()))) if taste_aurocs else 0.0

    # Odor channels (11..147)
    odor_p = preds[:, NUM_TASTE:]
    odor_t = targets[:, NUM_TASTE:]
    odor_m = masks[:, NUM_TASTE:]

    odor_aurocs = []
    for c in range(NUM_ODOR):
        m_c = odor_m[:, c] > 0
        if m_c.sum() >= 10:
            y_t = odor_t[m_c, c]
            y_p = odor_p[m_c, c]
            if len(np.unique(y_t)) > 1:
                try:
                    odor_aurocs.append(roc_auc_score(y_t, y_p))
                except:
                    pass
    odor_macro = float(np.mean(odor_aurocs)) if odor_aurocs else 0.0

    odor_active = odor_m > 0
    odor_micro = float(roc_auc_score(odor_t[odor_active], odor_p[odor_active])) if odor_active.sum() > 0 else 0.0

    # Top-5 precision
    top5_precs = []
    for i in range(len(odor_p)):
        if odor_m[i].sum() > 0:
            top5_idx = np.argsort(odor_p[i])[-5:]
            true_pos = odor_t[i, top5_idx].sum()
            top5_precs.append(true_pos / 5.0)
    top5_prec = float(np.mean(top5_precs)) if top5_precs else 0.0

    return {
        "taste_macro": taste_macro,
        "taste_breakdown": taste_aurocs,
        "odor_macro": odor_macro,
        "odor_micro": odor_micro,
        "odor_top5": top5_prec,
    }


@torch.no_grad()
def evaluate_pairs(model, loader, device):
    model.eval()
    records = {
        "half_cold": {"p_delta": [], "t_delta": [], "mask": [], "p_blend": [], "t_blend": []},
        "double_cold": {"p_delta": [], "t_delta": [], "mask": [], "p_blend": [], "t_blend": []},
    }

    for batch in loader:
        unique_graphs = batch["unique_graphs"]
        idx_a = batch["idx_a"]
        idx_b = batch["idx_b"]
        deltas = batch["deltas"].cpu().numpy()
        masks = batch["masks"].cpu().numpy()
        cold_types = batch["cold_types"]

        pred_deltas, y_blend, v_a, v_b = model.predict_pairs_batch(unique_graphs, idx_a, idx_b)
        pred_deltas = pred_deltas.cpu().numpy()
        y_blend = y_blend.cpu().numpy()

        v_a_np = v_a.cpu().numpy()
        v_b_np = v_b.cpu().numpy()
        true_blend = np.clip(0.5 * v_a_np + 0.5 * v_b_np + deltas, 0.0, 1.0)

        for i, ct in enumerate(cold_types):
            key = ct if ct in records else "half_cold"
            records[key]["p_delta"].append(pred_deltas[i])
            records[key]["t_delta"].append(deltas[i])
            records[key]["mask"].append(masks[i])
            records[key]["p_blend"].append(y_blend[i])
            records[key]["t_blend"].append(true_blend[i])

    results = {}
    for cohort in ["half_cold", "double_cold"]:
        if not records[cohort]["p_delta"]:
            continue
        p_d = np.array(records[cohort]["p_delta"])
        t_d = np.array(records[cohort]["t_delta"])
        m = np.array(records[cohort]["mask"])
        p_b = np.array(records[cohort]["p_blend"])
        t_b = np.array(records[cohort]["t_blend"])

        active = m > 0
        p_act = p_d[active]
        t_act = t_d[active]
        p_b_act = p_b[active]
        t_b_act = t_b[active]

        r_val, _ = pearsonr(p_act, t_act) if len(p_act) > 1 and np.std(p_act) > 1e-6 and np.std(t_act) > 1e-6 else (0.0, 0.0)
        rho_val, _ = spearmanr(p_act, t_act) if len(p_act) > 1 and np.std(p_act) > 1e-6 and np.std(t_act) > 1e-6 else (0.0, 0.0)
        blend_mae = float(np.mean(np.abs(p_b_act - t_b_act))) if len(p_b_act) > 0 else 0.0

        sig = np.abs(t_act) >= 0.05
        dir_acc = float((np.sign(p_act[sig]) == np.sign(t_act[sig])).mean()) if sig.sum() > 0 else 1.0

        results[cohort] = {
            "pearson_r": float(r_val),
            "spearman_rho": float(rho_val),
            "blend_mae": blend_mae,
            "direction_acc": dir_acc,
        }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    singles_ds = FlavriSinglesDataset(REPO_ROOT / "data/quarantining/singles_test.csv")
    pairs_ds = FlavriPairsDataset(REPO_ROOT / "data/quarantining/pair_test.csv")

    singles_loader = DataLoader(singles_ds, batch_size=256, shuffle=False, collate_fn=SinglesCollator(device=device))
    pairs_loader = DataLoader(pairs_ds, batch_size=256, shuffle=False, collate_fn=PairsCollator(device=device))

    models_to_eval = [
        ("FLAVORI 0.2", "E:/downloads/flavori-backup/flavori-0.2.pt", "v2"),
        ("FLAVORI 0.3", "E:/downloads/flavori-backup/flavori-0.3.pt", "v3"),
        ("FLAVORI 0.4", "E:/downloads/flavori-backup/flavori-0.4.pt", "v4"),
    ]

    all_res = {}
    for name, path, arch in models_to_eval:
        print(f"\n==========================================")
        print(f"Evaluating {name} from {path}...")
        print(f"==========================================")
        if arch == "v2":
            model = FlavriV2Model().to(device)
        else:
            model = FlavriMultimodalModel(hidden_dim=384, depth=6).to(device)

        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)

        res_singles = evaluate_singles(model, singles_loader, device)
        res_pairs = evaluate_pairs(model, pairs_loader, device)

        all_res[name] = {"singles": res_singles, "pairs": res_pairs}

        print(f"Taste Macro AUROC: {res_singles['taste_macro']:.4f}")
        for t_ch in ["taste_kokumi", "taste_astringent", "taste_cooling", "taste_heat", "taste_bitter", "taste_sour", "taste_salty", "taste_umami", "taste_sweet"]:
            print(f"  {t_ch}: {res_singles['taste_breakdown'].get(t_ch, 0.0):.4f}")
        print(f"Odor Micro AUROC: {res_singles['odor_micro']:.4f}")
        print(f"Odor Macro AUROC: {res_singles['odor_macro']:.4f}")
        print(f"Odor Top-5 Prec:  {res_singles['odor_top5']:.4f}")
        print(f"Half-Cold:")
        print(f"  Blend MAE:     {res_pairs['half_cold']['blend_mae']:.4f}")
        print(f"  Direction Acc: {res_pairs['half_cold']['direction_acc']:.2%}")
        print(f"  Pearson r:     {res_pairs['half_cold']['pearson_r']:.4f}")
        print(f"  Spearman rho:  {res_pairs['half_cold']['spearman_rho']:.4f}")
        print(f"Double-Cold:")
        print(f"  Blend MAE:     {res_pairs['double_cold']['blend_mae']:.4f}")
        print(f"  Direction Acc: {res_pairs['double_cold']['direction_acc']:.2%}")
        print(f"  Pearson r:     {res_pairs['double_cold']['pearson_r']:.4f}")
        print(f"  Spearman rho:  {res_pairs['double_cold']['spearman_rho']:.4f}")

    print("\nDONE!")

if __name__ == "__main__":
    main()
