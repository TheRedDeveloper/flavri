"""
train_stage2.py
FLAVRI Stage 2: Joint Multi-Task Mixture Synergy & Pure Singles Anchoring (FLAVRI 2.0).
Trains the Commutative Synergy Head while co-adapting the unfrozen 6-layer D-MPNN backbone (384d),
anchored by simultaneous single molecule ground-truth batches under Calibrated Pos-Weighted BCE,
Plackett-Luce Ranking, and Persistent Chemistry Regularization.
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
from flavri.losses import FlavriMultimodalLoss, FlavriSinglesLoss, ChemicalReconstructionLoss
from flavri.data import FlavriPairsDataset, PairsCollator, FlavriSinglesDataset, SinglesCollator, ScaffoldDiverseModalitySampler
from flavri.evaluate import evaluate_pairs, evaluate_singles

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="FLAVRI Stage 2: Joint Multi-Task Training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--backbone_lr", type=float, default=1e-4, help="Backbone & Single Head learning rate")
    parser.add_argument("--synergy_lr", type=float, default=5e-4, help="Synergy head learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--single_weight", type=float, default=0.5, help="Weight for pure singles anchoring loss")
    parser.add_argument("--aux_chem_weight", type=float, default=0.05, help="Weight for persistent chemical reconstruction loss")
    parser.add_argument("--id_reg_weight", type=float, default=0.05, help="Weight for identity self-penalty")
    parser.add_argument("--hidden_dim", type=int, default=384, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=6, help="D-MPNN depth")
    parser.add_argument("--stage1_ckpt", type=str, default="checkpoints/stage1/best_model.pt", help="Stage 1 model path")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/stage2", help="Checkpoint directory")
    parser.add_argument("--use_pcgrad", action="store_true", default=True, help="Enable Projecting Conflicting Gradients between mixture and singles tasks")
    parser.add_argument("--no_pcgrad", action="store_false", dest="use_pcgrad", help="Disable PCGrad")
    parser.add_argument("--scheduler", type=str, default="warm_restarts", choices=["warm_restarts", "cosine"], help="LR scheduler")
    parser.add_argument("--t_0", type=int, default=15, help="T_0 period for warm restarts")
    parser.add_argument("--t_mult", type=int, default=2, help="T_mult factor for warm restarts")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
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
        f"[bold cyan]FLAVRI Stage 2: Joint Multi-Task Mixture & Singles Training (FLAVRI 2.0)[/bold cyan]\n"
        f"[green]Device:[/green] {device} ({gpu_name})\n"
        f"[green]Architecture:[/green] 6-Layer D-MPNN ({args.hidden_dim}d) + Multi-Head Attentive Readout\n"
        f"[green]Epochs:[/green] {args.epochs} | [green]Batch Size:[/green] {args.batch_size}\n"
        f"[green]Backbone LR:[/green] {args.backbone_lr} | [green]Synergy Head LR:[/green] {args.synergy_lr}\n"
        f"[green]Singles Anchoring Weight:[/green] {args.single_weight} | [green]Chemical Anchor Weight:[/green] {args.aux_chem_weight}\n"
        f"[green]Stage 1 Checkpoint:[/green] {Path(args.stage1_ckpt).resolve()}\n"
        f"[green]Output Checkpoint Dir:[/green] {ckpt_dir.resolve()}",
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
    singles_train_sampler = ScaffoldDiverseModalitySampler(
        singles_train_csv,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    singles_train_loader = DataLoader(
        singles_train_dataset,
        batch_sampler=singles_train_sampler,
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

    # Load Stage 1 weights if available
    stage1_path = Path(args.stage1_ckpt)
    if stage1_path.exists():
        console.print(f"[bold green]Loading Stage 1 weights from {stage1_path}...[/bold green]")
        ckpt = torch.load(stage1_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        console.print(f"[bold yellow]Warning: Stage 1 checkpoint not found at {stage1_path}. Initializing from scratch.[/bold yellow]")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[dim]Total model parameters: {total_params:,}[/dim]\n")

    # Differential parameter groups
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
    if args.scheduler == "warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.t_0, T_mult=args.t_mult, eta_min=args.min_lr
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.min_lr)

    mix_loss_fn = FlavriMultimodalLoss(mix_neg_weight=0.2)
    singles_loss_fn = FlavriSinglesLoss(rank_weight=0.5, hier_weight=0.1, antag_weight=0.05, device=device).to(device)
    chem_loss_fn = ChemicalReconstructionLoss(device=device)

    start_epoch = 1
    best_composite = -2.0

    if args.resume:
        latest_path = ckpt_dir / "latest_model.pt"
        if latest_path.exists():
            console.print(f"[bold green]Resuming Stage 2 training from {latest_path}...[/bold green]")
            ckpt = torch.load(latest_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            if "val_pairs" in ckpt and "val_singles" in ckpt:
                best_composite = ckpt["val_pairs"]["half_cold"]["pearson_r"] + 0.5 * (
                    ckpt["val_singles"]["taste_macro_auroc"] + ckpt["val_singles"]["odor_macro_auroc"]
                )
            for _ in range(ckpt["epoch"]):
                scheduler.step()
            console.print(f"[green]Resuming at Epoch {start_epoch:02d}[/green]")
        else:
            console.print(f"[bold yellow]Warning: Checkpoint not found at {latest_path}. Starting from epoch 1.[/bold yellow]")

    # CSV Logger
    csv_headers = [
        "epoch",
        "train_loss",
        "half_cold_r",
        "half_cold_rho",
        "double_cold_r",
        "double_cold_dir_acc",
        "taste_macro_auroc",
        "odor_macro_auroc",
        "odor_micro_auroc",
        "odor_top5_prec",
        "time_sec",
    ]
    if not log_csv_path.exists() or (not args.resume and start_epoch == 1):
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    batches_per_epoch = 5 if args.dry_run else len(pair_train_loader)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_accum = 0.0
        n_batches = 0

        singles_iter = iter(singles_train_loader)

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]Epoch {epoch:02d}/{epochs:02d}[/bold cyan]"),
            BarColumn(bar_width=35),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• Loss: {task.fields[loss]:.4f}"),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("training", total=batches_per_epoch, loss=0.0)

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

                # Get synchronized single batch
                try:
                    single_batch = next(singles_iter)
                except StopIteration:
                    singles_iter = iter(singles_train_loader)
                    single_batch = next(singles_iter)

                # 1. Mixture Pair Forward Pass
                pred_deltas, y_blend, v_a, v_b = model.predict_pairs_batch(unique_graphs, idx_a, idx_b)
                true_blend = torch.clamp(0.5 * v_a.detach() + 0.5 * v_b.detach() + deltas, min=0.0, max=1.0)
                is_mix = torch.ones(batch_size, dtype=torch.bool, device=device)
                mix_loss = mix_loss_fn(y_blend, pred_deltas, true_blend, masks, is_mix, deltas_true=deltas)

                # 2. Pure Singles Anchoring Forward Pass (Calibrated Focal BCE + Ranking + Ontology)
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

                task_mixture = mix_loss + args.id_reg_weight * id_loss
                task_singles = args.single_weight * single_loss + args.aux_chem_weight * chem_loss
                total_loss = task_mixture + task_singles

                if args.use_pcgrad:
                    # PCGrad: project conflicting gradients between mixture & singles tasks
                    params = [p for p in model.parameters() if p.requires_grad]
                    g_mix = torch.autograd.grad(task_mixture, params, retain_graph=True, allow_unused=True)
                    g_sng = torch.autograd.grad(task_singles, params, retain_graph=False, allow_unused=True)

                    g_mix_flat = torch.cat([g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1) for g, p in zip(g_mix, params)])
                    g_sng_flat = torch.cat([g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1) for g, p in zip(g_sng, params)])

                    dot = torch.dot(g_mix_flat, g_sng_flat)
                    if dot < 0:
                        # Conflicting gradients: project onto normal planes
                        g_mix_proj = g_mix_flat - (dot / (torch.norm(g_sng_flat) ** 2 + 1e-8)) * g_sng_flat
                        g_sng_proj = g_sng_flat - (dot / (torch.norm(g_mix_flat) ** 2 + 1e-8)) * g_mix_flat
                    else:
                        g_mix_proj = g_mix_flat
                        g_sng_proj = g_sng_flat

                    g_merged = g_mix_proj + g_sng_proj

                    # Assign projected gradients directly to model parameters
                    offset = 0
                    for p in params:
                        numel = p.numel()
                        p.grad = g_merged[offset : offset + numel].view_as(p).clone()
                        offset += numel

                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                else:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                batch_loss = total_loss.item()
                train_loss_accum += batch_loss
                n_batches += 1

                progress.update(task, advance=1, loss=batch_loss)

        avg_train_loss = train_loss_accum / max(1, n_batches)
        scheduler.step()

        # Validation across mixture holdouts and singles holdout
        val_pairs = evaluate_pairs(model, pair_test_loader, device=device)
        val_singles = evaluate_singles(model, singles_test_loader, device=device)

        hc = val_pairs["half_cold"]
        dc = val_pairs["double_cold"]
        epoch_time = time.time() - epoch_start

        # Composite tracking metric
        composite = hc["pearson_r"] + 0.5 * (val_singles["taste_macro_auroc"] + val_singles["odor_macro_auroc"])

        # Log to CSV
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_train_loss:.5f}",
                f"{hc['pearson_r']:.4f}",
                f"{hc['spearman_rho']:.4f}",
                f"{dc['pearson_r']:.4f}",
                f"{dc['direction_acc']:.4f}",
                f"{val_singles['taste_macro_auroc']:.4f}",
                f"{val_singles['odor_macro_auroc']:.4f}",
                f"{val_singles['odor_micro_auroc']:.4f}",
                f"{val_singles['odor_top5_prec']:.4f}",
                f"{epoch_time:.2f}",
            ])

        # Checkpoints
        latest_path = ckpt_dir / "latest_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_pairs": val_pairs,
            "val_singles": val_singles,
            "train_loss": avg_train_loss,
        }, latest_path)

        if composite > best_composite:
            best_composite = composite
            best_path = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_pairs": val_pairs,
                "val_singles": val_singles,
                "best_composite_score": best_composite,
            }, best_path)

        # Pretty Terminal Table
        table = Table(title=f"Epoch {epoch:02d} Summary", show_header=True, header_style="bold magenta")
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
            f"{val_singles['taste_macro_auroc']:.4f}",
            f"{val_singles['odor_macro_auroc']:.4f}",
            f"{val_singles['odor_micro_auroc']:.4f}",
            f"{val_singles['odor_top5_prec']:.4f}",
            f"{epoch_time:.1f}",
        )
        console.print(table)


if __name__ == "__main__":
    main()
