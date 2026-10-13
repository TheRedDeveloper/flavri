"""
train_stage1.py
Stage 1: Multimodal Single Molecule Pre-training.
Trains D-MPNN backbone and Single Molecule Head on singles_train.csv.
Validates on zero-leakage holdout singles_test.csv.
Saves checkpoints and logs training loss & metrics to CSV.
Displays rich, formatted progress in terminal.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import csv

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn

from flavri.constants import NUM_CHANNELS
from flavri.models import FlavriMultimodalModel
from flavri.losses import FlavriSinglesLoss
from flavri.data import FlavriSinglesDataset, SinglesCollator
from flavri.evaluate import evaluate_singles

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="FLAVRI Stage 1: Singles Pre-training")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=4, help="D-MPNN depth")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/stage1", help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 quick epoch to verify end-to-end pipeline")
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
        f"[bold cyan]FLAVRI Chemosensory Engine: Stage 1 Single Pre-training[/bold cyan]\n"
        f"[green]Device:[/green] {device} ({gpu_name})\n"
        f"[green]Epochs:[/green] {args.epochs} | [green]Batch Size:[/green] {args.batch_size} | [green]LR:[/green] {args.lr}\n"
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    # Model
    model = FlavriMultimodalModel(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_channels=NUM_CHANNELS,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[dim]Total trainable parameters: {total_params:,}[/dim]\n")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        list(model.backbone.parameters()) + list(model.single_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    epochs = 1 if args.dry_run else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = FlavriSinglesLoss()

    # Initialize CSV logger
    csv_headers = [
        "epoch",
        "train_loss",
        "val_loss",
        "taste_macro_auroc",
        "odor_macro_auroc",
        "odor_micro_auroc",
        "odor_top5_prec",
        "lr",
        "time_sec",
    ]
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    best_val_score = -1.0

    for epoch in range(1, epochs + 1):
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

                _, preds = model.predict_singles(graphs)
                loss = loss_fn(preds, targets, masks)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_loss = loss.item()
                train_loss_accum += batch_loss
                n_batches += 1

                progress.update(task, advance=1, loss=batch_loss)

        avg_train_loss = train_loss_accum / max(1, n_batches)
        scheduler.step()

        # Validation
        val_metrics = evaluate_singles(model, test_loader, device=device)
        
        # Validation loss
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                _, preds = model.predict_singles(batch["graphs"])
                val_loss_accum += loss_fn(preds, batch["targets"], batch["masks"]).item()
                val_batches += 1
        avg_val_loss = val_loss_accum / max(1, val_batches)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Composite score for best model tracking: average of taste and odor AUROC
        composite_score = 0.5 * (val_metrics["taste_macro_auroc"] + val_metrics["odor_macro_auroc"])

        # Log to CSV
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_train_loss:.5f}",
                f"{avg_val_loss:.5f}",
                f"{val_metrics['taste_macro_auroc']:.4f}",
                f"{val_metrics['odor_macro_auroc']:.4f}",
                f"{val_metrics['odor_micro_auroc']:.4f}",
                f"{val_metrics['odor_top5_prec']:.4f}",
                f"{current_lr:.6f}",
                f"{epoch_time:.2f}",
            ])

        # Save latest checkpoint
        latest_path = ckpt_dir / "latest_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }, latest_path)

        # Save best checkpoint
        is_best = composite_score > best_val_score
        if is_best:
            best_val_score = composite_score
            best_path = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "best_composite_score": best_val_score,
            }, best_path)

        # Pretty Terminal Table per Epoch
        table = Table(title=f"Epoch {epoch:02d} Summary", show_header=True, header_style="bold magenta")
        table.add_column("Train Loss", justify="center", style="yellow")
        table.add_column("Val Loss", justify="center", style="yellow")
        table.add_column("Taste AUROC", justify="center", style="cyan")
        table.add_column("Odor Macro AUC", justify="center", style="cyan")
        table.add_column("Odor Micro AUC", justify="center", style="cyan")
        table.add_column("Odor Top-5 Prec", justify="center", style="green")
        table.add_column("Best", justify="center", style="bold green" if is_best else "dim")
        table.add_column("Time", justify="center", style="dim")

        table.add_row(
            f"{avg_train_loss:.4f}",
            f"{avg_val_loss:.4f}",
            f"{val_metrics['taste_macro_auroc']:.4f}",
            f"{val_metrics['odor_macro_auroc']:.4f}",
            f"{val_metrics['odor_micro_auroc']:.4f}",
            f"{val_metrics['odor_top5_prec']:.4f}",
            "[BEST]" if is_best else "-",
            f"{epoch_time:.1f}s",
        )
        console.print(table)

    console.print(f"\n[bold green]Stage 1 Pre-training Completed Successfully![/bold green]")
    console.print(f"Checkpoints saved in: [cyan]{ckpt_dir.resolve()}[/cyan]")
    console.print(f"Metrics log saved to: [cyan]{log_csv_path.resolve()}[/cyan]")


if __name__ == "__main__":
    main()
