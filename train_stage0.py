"""
train_stage0.py
FLAVRI Stage 0: Self-Supervised Chemical Foundation Pre-training.
Pre-trains the D-MPNN Backbone (512d, depth 6) and ChemicalReconstructionHead
on 500,000 clean, diverse molecules via Masked Atom Attribute Reconstruction.
"""

import sys
import time
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flavri.featurizer import mol2graph, BatchMolGraph, ATOM_FDIM
from flavri.models import FlavriMultimodalModel
from flavri.losses import ChemicalReconstructionLoss

console = Console()


class PretrainSmilesDataset(Dataset):
    """Memory-efficient SMILES dataset for self-supervised pre-training."""
    def __init__(self, csv_path: Path, max_samples: int = 500_000):
        print(f"Loading pre-training SMILES from {csv_path}...")
        df = pd.read_csv(csv_path, nrows=max_samples)
        self.smiles_list: List[str] = df["smiles"].dropna().tolist()
        print(f"Loaded {len(self.smiles_list):,} pre-training molecules.")

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> str:
        return self.smiles_list[idx]


def pretrain_collate_fn(smiles_batch: List[str], device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
    """Featurizes a batch of SMILES into a BatchMolGraph with random atom masking."""
    batch_graph = mol2graph(smiles_batch, device=device)
    total_atoms = batch_graph.total_atoms

    # Randomly select ~15% of atoms to mask
    mask_prob = 0.15
    mask_indices = torch.rand(total_atoms, device=device) < mask_prob
    if mask_indices.sum() == 0:
        # Guarantee at least one masked atom per batch
        mask_indices[torch.randint(0, total_atoms, (1,), device=device)] = True

    true_atom_features = batch_graph.f_atoms.clone()

    # Zero out masked atom features
    batch_graph.f_atoms[mask_indices] = 0.0

    return {
        "batch_graph": batch_graph,
        "true_atom_features": true_atom_features,
        "mask_indices": mask_indices,
        "batch_size": len(smiles_batch),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="FLAVRI Stage 0: Chemical Foundation Pre-training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of pre-training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Pre-training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--hidden_dim", type=int, default=512, help="D-MPNN hidden dimension")
    parser.add_argument("--depth", type=int, default=6, help="D-MPNN message passing depth")
    parser.add_argument("--data_csv", type=str, default="data/pretrain/unlabeled_500k.csv", help="Path to pre-training SMILES CSV")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/stage0", help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 quick epoch to verify end-to-end pipeline")
    parser.add_argument("--continue", "--continue_training", "--resume", action="store_true", dest="resume", help="Continue training from latest checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = ckpt_dir / "pretrain_log.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    console.print(Panel(
        f"[bold cyan]FLAVRI Stage 0: Self-Supervised Chemical Foundation Pre-training[/bold cyan]\n"
        f"[green]Device:[/green] {device} ({gpu_name})\n"
        f"[green]Architecture:[/green] 6-Layer D-MPNN ({args.hidden_dim}d) + Multi-Head Attentive Readout\n"
        f"[green]Objective:[/green] Masked Atom Attribute Reconstruction (15% random masking)\n"
        f"[green]Epochs:[/green] {args.epochs} | [green]Batch Size:[/green] {args.batch_size} | [green]LR:[/green] {args.lr}\n"
        f"[green]Output Dir:[/green] {ckpt_dir.resolve()}",
        border_style="cyan"
    ))

    data_csv = REPO_ROOT / args.data_csv
    dataset = PretrainSmilesDataset(data_csv, max_samples=500_000)

    # In dry-run mode, take just 500 molecules
    if args.dry_run:
        dataset.smiles_list = dataset.smiles_list[:500]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: pretrain_collate_fn(b, device=device),
        num_workers=0,
    )

    # Initialize model with FLAVRI 2.0 scaled backbone & chem head
    model = FlavriMultimodalModel(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[dim]Total model parameters: {total_params:,}[/dim]\n")

    # Only train backbone and chem_head in Stage 0
    pretrain_params = list(model.backbone.parameters()) + list(model.chem_head.parameters())
    optimizer = torch.optim.AdamW(pretrain_params, lr=args.lr, weight_decay=args.weight_decay)

    epochs = 1 if args.dry_run else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Instantiate Heteroatom-Calibrated Chemical Loss
    chem_loss_fn = ChemicalReconstructionLoss(device=device)

    start_epoch = 1
    best_loss = 1e9

    if args.resume:
        latest_path = ckpt_dir / "latest_model.pt"
        if latest_path.exists():
            console.print(f"[bold green]Resuming pre-training from {latest_path}...[/bold green]")
            ckpt = torch.load(latest_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt.get("train_loss", 1e9)
            for _ in range(ckpt["epoch"]):
                scheduler.step()
            console.print(f"[green]Resuming at Epoch {start_epoch:02d} (Previous Loss: {best_loss:.5f})[/green]")
        else:
            console.print(f"[bold yellow]Warning: Checkpoint not found at {latest_path}. Starting from epoch 1.[/bold yellow]")

    csv_headers = ["epoch", "train_loss", "elem_acc", "hetero_acc", "lr", "time_sec"]
    if not log_csv_path.exists() or (not args.resume and start_epoch == 1):
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_accum = 0.0
        elem_acc_accum = 0.0
        hetero_acc_accum = 0.0
        n_batches = 0

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]Epoch {epoch:02d}/{epochs:02d}[/bold cyan]"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• Loss: {task.fields[loss]:.3f}"),
            TextColumn("• Elem: {task.fields[elem_acc]:.1%}"),
            TextColumn("• Hetero: {task.fields[hetero_acc]:.1%}"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "pretraining",
                total=len(loader),
                loss=0.0,
                elem_acc=0.0,
                hetero_acc=0.0,
            )

            for batch in loader:
                optimizer.zero_grad()
                batch_graph = batch["batch_graph"]
                true_atoms = batch["true_atom_features"]
                mask_idx = batch["mask_indices"]

                # Forward pass through D-MPNN backbone with masked atom features
                _, atom_h = model.encode(batch_graph, return_atom_h=True)

                # Reconstruct atom attributes from atom embeddings
                pred_atoms = model.predict_chemistry(atom_h)

                # Heteroatom-calibrated multi-task loss + element accuracies
                loss, elem_acc, hetero_acc = chem_loss_fn(
                    pred_atoms,
                    true_atoms,
                    mask_idx,
                    return_accuracies=True,
                )

                loss.backward()
                nn.utils.clip_grad_norm_(pretrain_params, max_norm=1.0)
                optimizer.step()

                batch_loss = loss.item()
                train_loss_accum += batch_loss
                elem_acc_accum += elem_acc
                hetero_acc_accum += hetero_acc
                n_batches += 1

                progress.update(
                    task,
                    advance=1,
                    loss=batch_loss,
                    elem_acc=elem_acc,
                    hetero_acc=hetero_acc,
                )

        avg_loss = train_loss_accum / max(1, n_batches)
        avg_elem_acc = elem_acc_accum / max(1, n_batches)
        avg_hetero_acc = hetero_acc_accum / max(1, n_batches)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Log
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_loss:.5f}",
                f"{avg_elem_acc:.4f}",
                f"{avg_hetero_acc:.4f}",
                f"{current_lr:.6f}",
                f"{epoch_time:.2f}",
            ])

        # Checkpoint
        latest_path = ckpt_dir / "latest_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_loss,
            "elem_acc": avg_elem_acc,
            "hetero_acc": avg_hetero_acc,
        }, latest_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_loss": best_loss,
                "elem_acc": avg_elem_acc,
                "hetero_acc": avg_hetero_acc,
            }, best_path)

        table = Table(title=f"Stage 0 Epoch {epoch:02d} Summary", show_header=True, header_style="bold magenta")
        table.add_column("Chemical Loss", justify="center", style="yellow")
        table.add_column("Overall Elem Acc", justify="center", style="cyan")
        table.add_column("Heteroatom (N/O/S) Acc", justify="center", style="green")
        table.add_column("Learning Rate", justify="center", style="cyan")
        table.add_column("Epoch Time", justify="center", style="dim")
        table.add_row(
            f"{avg_loss:.4f}",
            f"{avg_elem_acc:.1%}",
            f"{avg_hetero_acc:.1%}",
            f"{current_lr:.6f}",
            f"{epoch_time:.1f}s",
        )
        console.print(table)

    console.print(f"\n[bold green]Stage 0 Pre-training Completed![/bold green] Weights saved to: {ckpt_dir.resolve()}")


if __name__ == "__main__":
    main()
