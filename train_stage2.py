"""
train_stage2.py
Stage 2: Joint Mixture Synergy and Blend Fine-Tuning.
Loads pre-trained D-MPNN backbone from Stage 1.
Trains Commutative Synergy Head and fine-tunes backbone on pair_train.csv.
Enforces exact commutativity and identity self-interaction neutrality.
Evaluates on zero-leakage holdout pair_test.csv across half_cold and double_cold cohorts.
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
from flavri.losses import FlavriMultimodalLoss
from flavri.data import FlavriPairsDataset, PairsCollator
from flavri.evaluate import evaluate_pairs

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="FLAVRI Stage 2: Mixture Synergy Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Pair batch size")
    parser.add_argument("--backbone_lr", type=float, default=1e-4, help="Backbone learning rate")
    parser.add_argument("--synergy_lr", type=float, default=5e-4, help="Synergy head learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--id_reg_weight", type=float, default=0.05, help="Weight for identity self-penalty")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=4, help="D-MPNN depth")
    parser.add_argument("--stage1_ckpt", type=str, default="checkpoints/stage1/best_model.pt", help="Stage 1 model path")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/stage2", help="Checkpoint directory")
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
        f"[bold cyan]FLAVRI Chemosensory Engine: Stage 2 Mixture Synergy Fine-Tuning[/bold cyan]\n"
        f"[green]Device:[/green] {device} ({gpu_name})\n"
        f"[green]Epochs:[/green] {args.epochs} | [green]Batch Size:[/green] {args.batch_size}\n"
        f"[green]Backbone LR:[/green] {args.backbone_lr} | [green]Synergy Head LR:[/green] {args.synergy_lr}\n"
        f"[green]Stage 1 Checkpoint:[/green] {Path(args.stage1_ckpt).resolve()}\n"
        f"[green]Output Checkpoint Dir:[/green] {ckpt_dir.resolve()}",
        border_style="cyan"
    ))

    # Datasets
    train_csv = REPO_ROOT / "data" / "quarantining" / "pair_train.csv"
    test_csv = REPO_ROOT / "data" / "quarantining" / "pair_test.csv"

    console.print("[dim]Loading pair datasets into memory...[/dim]")
    train_dataset = FlavriPairsDataset(train_csv)
    test_dataset = FlavriPairsDataset(test_csv)
    console.print(f"[bold green]Datasets loaded:[/bold green] {len(train_dataset):,} train pairs, {len(test_dataset):,} test holdout pairs.")

    train_collator = PairsCollator(device=device, add_identity_pairs=True, identity_ratio=0.1)
    test_collator = PairsCollator(device=device, add_identity_pairs=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_collator,
        num_workers=args.num_workers,
    )

    # Initialize model
    model = FlavriMultimodalModel(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_channels=NUM_CHANNELS,
        dropout=0.1,
    ).to(device)

    # Load Stage 1 weights if available
    stage1_path = Path(args.stage1_ckpt)
    if stage1_path.exists():
        console.print(f"[bold green]Loading pre-trained Stage 1 weights from {stage1_path}...[/bold green]")
        ckpt = torch.load(stage1_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        console.print(f"[bold yellow]Warning: Stage 1 checkpoint not found at {stage1_path}. Initializing from scratch.[/bold yellow]")

    # Differential parameter groups
    param_groups = [
        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        {"params": model.single_head.parameters(), "lr": args.backbone_lr},
        {"params": model.synergy_head.parameters(), "lr": args.synergy_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    epochs = 1 if args.dry_run else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = FlavriMultimodalLoss(mix_neg_weight=0.2)

    # CSV Logger
    csv_headers = [
        "epoch",
        "train_loss",
        "half_cold_r",
        "half_cold_rho",
        "half_cold_f1",
        "double_cold_r",
        "double_cold_mae",
        "double_cold_dir_acc",
        "time_sec",
    ]
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    best_r_score = -2.0

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
                unique_graphs = batch["unique_graphs"]
                idx_a = batch["idx_a"]
                idx_b = batch["idx_b"]
                deltas = batch["deltas"]
                masks = batch["masks"]
                batch_size = batch["batch_size"]

                # O(1) graph forward pass
                pred_deltas, y_blend, v_a, v_b = model.predict_pairs_batch(unique_graphs, idx_a, idx_b)

                # Mixture loss
                is_mix = torch.ones(batch_size, dtype=torch.bool, device=device)
                mix_loss = loss_fn(y_blend, pred_deltas, deltas, masks, is_mix)

                # Identity penalty on unique molecule embeddings
                z_unique = model.encode(unique_graphs)
                id_loss = model.synergy_head.compute_identity_penalty(z_unique)

                total_loss = mix_loss + args.id_reg_weight * id_loss

                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_loss = total_loss.item()
                train_loss_accum += batch_loss
                n_batches += 1

                progress.update(task, advance=1, loss=batch_loss)

        avg_train_loss = train_loss_accum / max(1, n_batches)
        scheduler.step()

        # Validation across cold cohorts
        val_metrics = evaluate_pairs(model, test_loader, device=device)
        hc = val_metrics["half_cold"]
        dc = val_metrics["double_cold"]

        epoch_time = time.time() - epoch_start
        composite_r = hc["pearson_r"]

        # Log to CSV
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_train_loss:.5f}",
                f"{hc['pearson_r']:.4f}",
                f"{hc['spearman_rho']:.4f}",
                f"{hc['synergy_f1']:.4f}",
                f"{dc['pearson_r']:.4f}",
                f"{dc['blend_mae']:.4f}",
                f"{dc['direction_acc']:.4f}",
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

        is_best = composite_r > best_r_score
        if is_best:
            best_r_score = composite_r
            best_path = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "best_r": best_r_score,
            }, best_path)

        # Pretty Terminal Table per Epoch
        table = Table(title=f"Epoch {epoch:02d} Summary", show_header=True, header_style="bold magenta")
        table.add_column("Train Loss", justify="center", style="yellow")
        table.add_column("HC r", justify="center", style="cyan")
        table.add_column("HC F1", justify="center", style="cyan")
        table.add_column("DC r", justify="center", style="green")
        table.add_column("DC MAE", justify="center", style="green")
        table.add_column("Dir Acc", justify="center", style="green")
        table.add_column("Best", justify="center", style="bold green" if is_best else "dim")
        table.add_column("Time", justify="center", style="dim")

        table.add_row(
            f"{avg_train_loss:.4f}",
            f"{hc['pearson_r']:.4f}",
            f"{hc['synergy_f1']:.4f}",
            f"{dc['pearson_r']:.4f}",
            f"{dc['blend_mae']:.4f}",
            f"{dc['direction_acc']:.2%}",
            "[BEST]" if is_best else "-",
            f"{epoch_time:.1f}s",
        )
        console.print(table)

    console.print(f"\n[bold green]Stage 2 Joint Fine-Tuning Completed Successfully![/bold green]")
    console.print(f"Checkpoints saved in: [cyan]{ckpt_dir.resolve()}[/cyan]")
    console.print(f"Metrics log saved to: [cyan]{log_csv_path.resolve()}[/cyan]")


if __name__ == "__main__":
    main()
