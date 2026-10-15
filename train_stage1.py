"""
train_stage1.py
FLAVRI Stage 1: Supervised Sensory Anchoring with Persistent Chemistry Regularization.
Initializes from Stage 0 pre-trained foundation weights, training the SingleMoleculeHead
under Calibrated Pos-Weighted Focal BCE, Plackett-Luce Ranking, and Ontology constraints.
"""

import sys
import time
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flavri.constants import NUM_CHANNELS
from flavri.models import FlavriMultimodalModel
from flavri.losses import FlavriSinglesLoss, ChemicalReconstructionLoss
from flavri.data import FlavriSinglesDataset, SinglesCollator, ScaffoldDiverseModalitySampler
from flavri.evaluate import evaluate_singles

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="FLAVRI Stage 1: Singles Pre-training")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--hidden_dim", type=int, default=384, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=6, help="D-MPNN depth")
    parser.add_argument("--stage0_ckpt", type=str, default="checkpoints/stage0/best_model.pt", help="Stage 0 pre-trained weights")
    parser.add_argument("--aux_chem_weight", type=float, default=0.05, help="Weight for persistent chemical reconstruction loss")
    parser.add_argument("--scheduler", type=str, default="warm_restarts", choices=["warm_restarts", "cosine"], help="LR scheduler")
    parser.add_argument("--t_0", type=int, default=15, help="T_0 period for warm restarts")
    parser.add_argument("--t_mult", type=int, default=2, help="T_mult factor for warm restarts")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/stage1", help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 quick epoch to verify end-to-end pipeline")
    parser.add_argument("--continue", "--continue_training", "--resume", action="store_true", dest="resume", help="Continue training from latest checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = ckpt_dir / "training_log.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    console.print(Panel(
        f"[bold cyan]FLAVRI Stage 1: Supervised Sensory Anchoring (FLAVRI 2.0)[/bold cyan]\n"
        f"[green]Device:[/green] {device} ({gpu_name})\n"
        f"[green]Architecture:[/green] 6-Layer D-MPNN ({args.hidden_dim}d) + Multi-Head Attentive Readout\n"
        f"[green]Epochs:[/green] {args.epochs} | [green]Batch Size:[/green] {args.batch_size} | [green]LR:[/green] {args.lr}\n"
        f"[green]Stage 0 Checkpoint:[/green] {Path(args.stage0_ckpt).resolve()}\n"
        f"[green]Checkpoint Directory:[/green] {ckpt_dir.resolve()}",
        border_style="cyan"
    ))

    # Datasets
    train_csv = REPO_ROOT / "data" / "quarantining" / "singles_train.csv"
    test_csv = REPO_ROOT / "data" / "quarantining" / "singles_test.csv"

    console.print("[dim]Loading single datasets into memory...[/dim]")
    train_dataset = FlavriSinglesDataset(train_csv)
    test_dataset = FlavriSinglesDataset(test_csv)
    console.print(f"[bold green]Datasets loaded:[/bold green] {len(train_dataset):,} train molecules, {len(test_dataset):,} test holdout molecules.")

    collator = SinglesCollator(device=device)
    train_sampler = ScaffoldDiverseModalitySampler(
        train_csv,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    # Initialize model
    model = FlavriMultimodalModel(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_channels=NUM_CHANNELS,
        dropout=0.1,
    ).to(device)

    # Load Stage 0 weights if available
    stage0_path = Path(args.stage0_ckpt)
    if stage0_path.exists():
        console.print(f"[bold green]Loading pre-trained Stage 0 weights from {stage0_path}...[/bold green]")
        ckpt = torch.load(stage0_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        console.print(f"[bold yellow]Notice: Stage 0 checkpoint not found at {stage0_path}. Initializing from scratch.[/bold yellow]")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[dim]Total model parameters: {total_params:,}[/dim]\n")

    # Optimizer & Scheduler (trains backbone, fp_proj, fusion, chem_head, and single_head)
    stage1_params = (
        list(model.backbone.parameters())
        + list(model.fp_proj.parameters())
        + list(model.fusion.parameters())
        + list(model.chem_head.parameters())
        + list(model.single_head.parameters())
    )
    optimizer = torch.optim.AdamW(stage1_params, lr=args.lr, weight_decay=args.weight_decay)

    epochs = 1 if args.dry_run else args.epochs
    if args.scheduler == "warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.t_0, T_mult=args.t_mult, eta_min=args.min_lr
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.min_lr)
    loss_fn = FlavriSinglesLoss(rank_weight=0.5, hier_weight=0.1, antag_weight=0.05, device=device).to(device)
    chem_loss_fn = ChemicalReconstructionLoss(device=device)

    # Initialize CSV logger
    start_epoch = 1
    best_val_score = -1.0

    if args.resume:
        latest_path = ckpt_dir / "latest_model.pt"
        if latest_path.exists():
            console.print(f"[bold green]Resuming Stage 1 training from {latest_path}...[/bold green]")
            ckpt = torch.load(latest_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            if "val_metrics" in ckpt:
                best_val_score = ckpt["val_metrics"]["odor_macro_auroc"] + ckpt["val_metrics"]["taste_macro_auroc"]
            for _ in range(ckpt["epoch"]):
                scheduler.step()
            console.print(f"[green]Resuming at Epoch {start_epoch:02d}[/green]")
        else:
            console.print(f"[bold yellow]Warning: Checkpoint not found at {latest_path}. Starting from epoch 1.[/bold yellow]")

    csv_headers = [
        "epoch",
        "train_loss",
        "taste_macro_auroc",
        "odor_macro_auroc",
        "odor_micro_auroc",
        "odor_top5_prec",
        "lr",
        "time_sec",
    ]
    if not log_csv_path.exists() or (not args.resume and start_epoch == 1):
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_accum = 0.0
        n_batches = 0

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]Epoch {epoch:02d}/{epochs:02d}[/bold cyan]"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• Loss: {task.fields[loss]:.4f}"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("training", total=len(train_loader), loss=0.0)

            for batch in train_loader:
                optimizer.zero_grad()
                graphs = batch["graphs"]
                targets = batch["targets"]
                masks = batch["masks"]

                # Apply 10% atom masking for persistent chemical anchor
                total_atoms = graphs.total_atoms
                mask_prob = 0.10
                mask_indices = torch.rand(total_atoms, device=device) < mask_prob
                true_atoms = graphs.f_atoms.clone()
                graphs.f_atoms[mask_indices] = 0.0

                # Forward pass: sensory profiles + atom representations
                z, v, atom_h = model.predict_singles(graphs, return_atom_h=True)

                # Sensory loss (Focal BCE + Plackett-Luce Ranking + Hierarchy + Antagonism)
                sensory_loss = loss_fn(v, targets, masks)

                # Heteroatom-calibrated chemical reconstruction anchor
                pred_chem = model.predict_chemistry(atom_h)
                chem_loss = chem_loss_fn(pred_chem, true_atoms, mask_indices)

                total_loss = sensory_loss + args.aux_chem_weight * chem_loss

                total_loss.backward()
                nn.utils.clip_grad_norm_(stage1_params, max_norm=1.0)
                optimizer.step()

                batch_loss = total_loss.item()
                train_loss_accum += batch_loss
                n_batches += 1

                progress.update(task, advance=1, loss=batch_loss)

        avg_train_loss = train_loss_accum / max(1, n_batches)
        scheduler.step()

        # Validation on test set
        val_metrics = evaluate_singles(model, test_loader, device=device)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        composite_score = 0.5 * (val_metrics["taste_macro_auroc"] + val_metrics["odor_macro_auroc"])

        # Log
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_train_loss:.5f}",
                f"{val_metrics['taste_macro_auroc']:.4f}",
                f"{val_metrics['odor_macro_auroc']:.4f}",
                f"{val_metrics['odor_micro_auroc']:.4f}",
                f"{val_metrics['odor_top5_prec']:.4f}",
                f"{current_lr:.6f}",
                f"{epoch_time:.2f}",
            ])

        # Checkpoints
        latest_path = ckpt_dir / "latest_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "train_loss": avg_train_loss,
        }, latest_path)

        if composite_score > best_val_score:
            best_val_score = composite_score
            best_path = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "best_composite_score": best_val_score,
            }, best_path)

        table = Table(title=f"Epoch {epoch:02d} Summary", show_header=True, header_style="bold magenta")
        table.add_column("Train Loss", justify="center", style="yellow")
        table.add_column("Taste AUROC", justify="center", style="cyan")
        table.add_column("Odor Macro AUC", justify="center", style="cyan")
        table.add_column("Odor Micro AUC", justify="center", style="cyan")
        table.add_column("Odor Top-5", justify="center", style="green")
        table.add_column("Time", justify="center", style="dim")

        table.add_row(
            f"{avg_train_loss:.4f}",
            f"{val_metrics['taste_macro_auroc']:.4f}",
            f"{val_metrics['odor_macro_auroc']:.4f}",
            f"{val_metrics['odor_micro_auroc']:.4f}",
            f"{val_metrics['odor_top5_prec']:.4f}",
            f"{epoch_time:.1f}s",
        )
        console.print(table)

    console.print(f"\n[bold green]Stage 1 Pre-training Completed![/bold green] Best score: {best_val_score:.4f}")


if __name__ == "__main__":
    main()
