"""
train_stage3_swa.py
FLAVRI Stage 3: Stochastic Weight Averaging (SWA) Flat-Basin Polishing (FLAVRI 2.0).
Loads the converged Stage 2 checkpoint (checkpoints/stage2/best_model.pt) and executes
cyclical learning rate exploration across the multimodal loss landscape.
Averaged model weights find the center of wide flat basins, eliminating sharp valleys
and boosting test set generalization on both holdout singles and mixture pairs.
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
from torch.optim.swa_utils import AveragedModel
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
from flavri.losses import FlavriMultimodalLoss, FlavriSinglesLoss, ChemicalReconstructionLoss
from flavri.data import FlavriPairsDataset, PairsCollator, FlavriSinglesDataset, SinglesCollator
from flavri.evaluate import evaluate_pairs, evaluate_singles

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="FLAVRI Stage 3: SWA Polishing")
    parser.add_argument("--epochs", type=int, default=35, help="Number of SWA exploration epochs")
    parser.add_argument("--cycle_epochs", type=int, default=3, help="Cyclical LR restart period (epochs)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--backbone_lr", type=float, default=5e-5, help="Peak SWA backbone learning rate")
    parser.add_argument("--synergy_lr", type=float, default=2e-4, help="Peak SWA synergy head learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR at cycle trough")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--single_weight", type=float, default=0.5, help="Weight for pure singles anchoring loss")
    parser.add_argument("--aux_chem_weight", type=float, default=0.05, help="Weight for persistent chemical reconstruction loss")
    parser.add_argument("--id_reg_weight", type=float, default=0.05, help="Weight for identity self-penalty")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=6, help="D-MPNN depth")
    parser.add_argument("--stage2_ckpt", type=str, default="checkpoints/stage2/best_model.pt", help="Stage 2 input checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/stage3", help="Checkpoint directory for Stage 3 SWA")
    parser.add_argument("--output_name", type=str, default="swa_model.pt", help="SWA checkpoint output name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 quick test epoch with 5 batches")
    parser.add_argument("--continue", "--continue_training", "--resume", action="store_true", dest="resume", help="Continue training from latest checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    swa_save_path = ckpt_dir / args.output_name
    log_csv_path = ckpt_dir / "swa_training_log.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    console.print(Panel(
        f"[bold cyan]FLAVRI Stage 3: SWA Flat-Basin Polishing (FLAVRI 2.0)[/bold cyan]\n"
        f"[green]Device:[/green] {device} ({gpu_name})\n"
        f"[green]Architecture:[/green] 6-Layer D-MPNN ({args.hidden_dim}d) + Multi-Head Attentive Readout\n"
        f"[green]SWA Epochs:[/green] {args.epochs} | [green]Cycle Period:[/green] {args.cycle_epochs} epochs | [green]Batch Size:[/green] {args.batch_size}\n"
        f"[green]Backbone LR:[/green] {args.backbone_lr} -> {args.min_lr} | [green]Synergy LR:[/green] {args.synergy_lr} -> {args.min_lr}\n"
        f"[green]Input Stage 2 Model:[/green] {Path(args.stage2_ckpt).resolve()}\n"
        f"[green]Output SWA Checkpoint:[/green] {swa_save_path.resolve()}",
        border_style="cyan"
    ))

    # Datasets
    pair_train_csv = REPO_ROOT / "data" / "quarantining" / "pair_train.csv"
    pair_test_csv = REPO_ROOT / "data" / "quarantining" / "pair_test.csv"
    singles_train_csv = REPO_ROOT / "data" / "quarantining" / "singles_train.csv"
    singles_test_csv = REPO_ROOT / "data" / "quarantining" / "singles_test.csv"

    console.print("[dim]Loading datasets into memory...[/dim]")
    pair_train_dataset = FlavriPairsDataset(pair_train_csv)
    pair_test_dataset = FlavriPairsDataset(pair_test_csv)
    singles_train_dataset = FlavriSinglesDataset(singles_train_csv)
    singles_test_dataset = FlavriSinglesDataset(singles_test_csv)

    console.print(f"[bold green]Pairs loaded:[/bold green] {len(pair_train_dataset):,} train, {len(pair_test_dataset):,} test holdout.")
    console.print(f"[bold green]Singles loaded:[/bold green] {len(singles_train_dataset):,} train, {len(singles_test_dataset):,} test holdout.")

    pair_train_collator = PairsCollator(device=device, add_identity_pairs=True, identity_ratio=0.1)
    pair_test_collator = PairsCollator(device=device, add_identity_pairs=False)
    singles_collator = SinglesCollator(device=device)

    pair_train_loader = DataLoader(
        pair_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pair_train_collator,
        num_workers=args.num_workers,
    )
    pair_test_loader = DataLoader(
        pair_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pair_test_collator,
        num_workers=args.num_workers,
    )
    singles_train_loader = DataLoader(
        singles_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=singles_collator,
        num_workers=args.num_workers,
    )
    singles_test_loader = DataLoader(
        singles_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=singles_collator,
        num_workers=args.num_workers,
    )

    # Initialize model
    model = FlavriMultimodalModel(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_channels=NUM_CHANNELS,
        dropout=0.1,
    ).to(device)

    # Load Stage 2 checkpoint
    stage2_path = Path(args.stage2_ckpt)
    if stage2_path.exists():
        console.print(f"[bold green]Loading converged Stage 2 weights from {stage2_path}...[/bold green]")
        ckpt = torch.load(stage2_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        console.print(f"[bold yellow]Warning: Stage 2 checkpoint not found at {stage2_path}. Proceeding from current initialization.[/bold yellow]")

    # Initialize SWA AveragedModel
    swa_model = AveragedModel(model)

    param_groups = [
        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        {"params": model.fp_proj.parameters(), "lr": args.backbone_lr},
        {"params": model.fusion.parameters(), "lr": args.backbone_lr},
        {"params": model.chem_head.parameters(), "lr": args.backbone_lr},
        {"params": model.single_head.parameters(), "lr": args.backbone_lr},
        {"params": model.synergy_head.parameters(), "lr": args.synergy_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    epochs = 1 if args.dry_run else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.cycle_epochs,
        T_mult=1,
        eta_min=args.min_lr,
    )

    mix_loss_fn = FlavriMultimodalLoss()
    singles_loss_fn = FlavriSinglesLoss(rank_weight=0.5, hier_weight=0.1, antag_weight=0.05, device=device).to(device)
    chem_loss_fn = ChemicalReconstructionLoss(device=device)

    start_epoch = 1
    best_composite = -2.0

    if args.resume:
        latest_path = ckpt_dir / "latest_swa_model.pt"
        if not latest_path.exists():
            latest_path = swa_save_path
        if latest_path.exists():
            console.print(f"[bold green]Resuming SWA training from {latest_path}...[/bold green]")
            ckpt = torch.load(latest_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            swa_model = AveragedModel(model)
            start_epoch = ckpt.get("epoch", 1) + 1
            best_composite = ckpt.get("composite_score", -2.0)
            for _ in range(start_epoch - 1):
                scheduler.step()
            console.print(f"[green]Resuming at SWA Epoch {start_epoch:02d}[/green]")
        else:
            console.print(f"[bold yellow]Warning: Checkpoint not found at {latest_path}. Starting from epoch 1.[/bold yellow]")

    # CSV Logger
    csv_headers = [
        "epoch",
        "train_loss",
        "swa_half_cold_r",
        "swa_double_cold_r",
        "swa_double_cold_dir_acc",
        "swa_taste_macro_auroc",
        "swa_odor_macro_auroc",
        "swa_odor_micro_auroc",
        "swa_odor_top5_prec",
        "time_sec",
    ]
    if not log_csv_path.exists() or (not args.resume and start_epoch == 1):
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    batches_per_epoch = 5 if args.dry_run else len(pair_train_loader)

    console.print(f"[bold green]Starting SWA training across {epochs} epochs ({batches_per_epoch} batches/epoch)...[/bold green]")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_accum = 0.0
        n_batches = 0

        singles_iter = iter(singles_train_loader)

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]SWA Epoch {epoch:02d}/{epochs:02d}[/bold cyan]"),
            BarColumn(bar_width=35),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• Loss: {task.fields[loss]:.4f}"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("swa_training", total=batches_per_epoch, loss=0.0)

            for batch_idx, batch in enumerate(pair_train_loader):
                if args.dry_run and batch_idx >= 5:
                    break

                optimizer.zero_grad()
                unique_graphs = batch["unique_graphs"]
                idx_a = batch["idx_a"]
                idx_b = batch["idx_b"]
                deltas = batch["deltas"]
                masks = batch["masks"]
                batch_size = batch["batch_size"]

                try:
                    single_batch = next(singles_iter)
                except StopIteration:
                    singles_iter = iter(singles_train_loader)
                    single_batch = next(singles_iter)

                # 1. Mixture Pair Forward Pass (Pure Synergy Deltas - zero single_head interference)
                pred_deltas = model.predict_synergy_batch(unique_graphs, idx_a, idx_b)
                mix_loss = mix_loss_fn(pred_deltas, deltas, masks)

                # 2. Pure Singles Anchoring Forward Pass
                s_graphs = single_batch["graphs"]
                total_s_atoms = s_graphs.total_atoms
                s_mask_idx = torch.rand(total_s_atoms, device=device) < 0.10
                true_s_atoms = s_graphs.f_atoms.clone()
                s_graphs.f_atoms[s_mask_idx] = 0.0

                _, single_preds, atom_h = model.predict_singles(s_graphs, return_atom_h=True)
                single_loss = singles_loss_fn(single_preds, single_batch["targets"], single_batch["masks"])

                # Persistent chemical reconstruction anchor
                pred_chem = model.predict_chemistry(atom_h)
                chem_loss = chem_loss_fn(pred_chem, true_s_atoms, s_mask_idx)

                # 3. Identity penalty on unique molecule embeddings
                z_unique = model.encode(unique_graphs)
                id_loss = model.synergy_head.compute_identity_penalty(z_unique)

                total_loss = (
                    mix_loss
                    + args.single_weight * single_loss
                    + args.aux_chem_weight * chem_loss
                    + args.id_reg_weight * id_loss
                )

                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_loss = total_loss.item()
                train_loss_accum += batch_loss
                n_batches += 1

                progress.update(task, advance=1, loss=batch_loss)

        avg_train_loss = train_loss_accum / max(1, n_batches)
        scheduler.step()

        # Update SWA parameters with running model state
        swa_model.update_parameters(model)

        # Evaluate SWA averaged model
        console.print("[dim]Evaluating SWA Averaged Model on Holdout Sets...[/dim]")
        eval_model = swa_model.module
        eval_model.eval()

        val_pairs_swa = evaluate_pairs(eval_model, pair_test_loader, device=device)
        val_singles_swa = evaluate_singles(eval_model, singles_test_loader, device=device)

        hc = val_pairs_swa["half_cold"]
        dc = val_pairs_swa["double_cold"]
        epoch_time = time.time() - epoch_start

        composite_swa = hc["pearson_r"] + 0.5 * (val_singles_swa["taste_macro_auroc"] + val_singles_swa["odor_macro_auroc"])

        # Log SWA metrics to CSV
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_train_loss:.5f}",
                f"{hc['pearson_r']:.4f}",
                f"{dc['pearson_r']:.4f}",
                f"{dc['direction_acc']:.4f}",
                f"{val_singles_swa['taste_macro_auroc']:.4f}",
                f"{val_singles_swa['odor_macro_auroc']:.4f}",
                f"{val_singles_swa['odor_micro_auroc']:.4f}",
                f"{val_singles_swa['odor_top5_prec']:.4f}",
                f"{epoch_time:.2f}",
            ])

        # Always save current SWA model
        torch.save({
            "epoch": epoch,
            "model_state_dict": swa_model.module.state_dict(),
            "val_pairs": val_pairs_swa,
            "val_singles": val_singles_swa,
            "composite_score": composite_swa,
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "architecture": f"FLAVRI 2.0 ({args.hidden_dim}d, 6-layer D-MPNN, Multi-Head Attentive Readout)",
        }, swa_save_path)

        # Summary Table
        table = Table(title=f"SWA Epoch {epoch:02d} Averaged Model Performance", show_header=True, header_style="bold cyan")
        table.add_column("Train Loss", justify="center", style="yellow")
        table.add_column("HC Pearson r", justify="center", style="cyan")
        table.add_column("DC Dir Acc", justify="center", style="green")
        table.add_column("Taste AUROC", justify="center", style="cyan")
        table.add_column("Odor Macro AUC", justify="center", style="cyan")
        table.add_column("Odor Micro AUC", justify="center", style="cyan")
        table.add_column("Top-5 Prec", justify="center", style="green")
        table.add_column("Time (s)", justify="center", style="dim")

        table.add_row(
            f"{avg_train_loss:.4f}",
            f"{hc['pearson_r']:.4f}",
            f"{dc['direction_acc']:.2%}",
            f"{val_singles_swa['taste_macro_auroc']:.4f}",
            f"{val_singles_swa['odor_macro_auroc']:.4f}",
            f"{val_singles_swa['odor_micro_auroc']:.4f}",
            f"{val_singles_swa['odor_top5_prec']:.4f}",
            f"{epoch_time:.1f}",
        )
        console.print(table)

    console.print(f"[bold green]SWA Polishing Complete! Final model saved to {swa_save_path.resolve()}[/bold green]")


if __name__ == "__main__":
    main()
