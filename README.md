# FLAVRI 0.8

A revolutionary Multimodal Chemosensory Model. Predicts **148 continuous perceptual dimensions** (11 taste/trigeminal + 137 odor channels) for single molecules and non-linear binary mixtures directly from molecular graphs.

For serving FLAVRI via Model Context Protocol to Claude Desktop or other LLMs, see [flavri-mcp](https://github.com/TheRedDeveloper/flavri-mcp).

## Quickstart

```python
import torch
from flavri import FlavriMultimodalModel, mol2graph, ALL_CHANNELS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlavriMultimodalModel(hidden_dim=512, depth=6).to(device)

checkpoint = torch.load("checkpoints/flavri-0.8.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

graph_vanillin = mol2graph(["O=Cc1ccc(O)c(OC)c1"], device=device)
with torch.no_grad():
    z_vanillin, preds_vanillin = model.predict_singles(graph_vanillin)

top_notes = torch.topk(preds_vanillin[0], 5)
for idx, score in zip(top_notes.indices, top_notes.values):
    print(f"{ALL_CHANNELS[idx]}: {score.item():.1%}")

graph_caffeine = mol2graph(["Cn1cnc2c1c(=O)n(C)c(=O)n2C"], device=device)
with torch.no_grad():
    z_caffeine, preds_caffeine = model.predict_singles(graph_caffeine)
    delta = model.predict_synergy(z_vanillin, z_caffeine)
    
    w_a, w_b = torch.tensor([0.5]), torch.tensor([0.5])
    blend = model.blend_profiles(preds_vanillin, preds_caffeine, delta, w_a, w_b)
```

## Architecture Highlights

- 6-Layer D-MPNN (512d) with 4-head attentive readout
- 512-dim radius-2 fingerprints projected and fused with graph embeddings
- 3.48 Million parameters

## Benchmark Performance

Evaluated under strict quarantining:

### Single Molecule Holdout (2,570 Unseen Molecules)

| Task | Metric | Value |
|:---|:---|:---|
| **Taste Channels** | Macro AUROC | **0.916** |
| **Odor Channels** | Micro AUROC | **0.880** |
| | Macro AUROC | **0.819** |

### Pairwise Mixture Holdout (11,037 Unseen Mixtures)

| Partition | Pearson r | Accuracy | Blend MAE |
|:---|:---|:---|:---|
| **Half-Cold** (1 novel molecule) | 0.657 | 86.2% | 0.0141 |
| **Double-Cold** (both molecules novel) | 0.486 | 79.9% | 0.0167 |

## Datasets

FLAVRI is trained on **28,236 single molecules** and **49,486 pairwise mixtures**, backed by a **500k MOSES molecule** pretraining corpus:
- **Molecules (28,236)**:
  - **Taste (21,724 molecules)**: Harmonized across 50 source streams covering 11 modalities. Primary sources include MultiFlavor, ChemTastesDB, BitterDB, TastePeptidesDB, UmamiMeta, VirtuousUmami, iUmami, iBitter, and ChEMBL assays, balanced by verified negative assay sets.
  - **Olfaction (8,046 molecules)**: Integrated from The Good Scents Company (TGSC), Leffingwell (LF), Merged Molecular Dataset (MMD), and OlfactionBase.
- **Synergies (49,486)**:
  - **Odor (48,801 pairs)**: From the Blended Pairs (BP) dataset capturing interactions across 74/137 channels.
  - **Taste (685 pairs)**: TastePeptidesDB synergies & masking, TAS2R bitter inhibition, classic psychophysical benchmarks, and cross-modal trigeminal modulations.

## Training Curriculum

FLAVRI 0.8 is trained through a four-stage curriculum:

1. **Stage 0 (`train_stage0.py`)**: Self-supervised pretraining on 500k MOSES molecules via masked heteroatom attribute reconstruction.
2. **Stage 1 (`train_stage1.py`)**: Supervised sensory anchoring with cyclical cosine annealing and Plackett-Luce ranking loss.
3. **Stage 2 (`train_stage2.py`)**: Joint multi-task mixture synergy optimization with **PCGrad** (Projecting Conflicting Gradients) and continuous Huber regression on odor deltas.
4. **Stage 3 (`train_stage3_swa.py`)**: Stochastic Weight Averaging (SWA) flat-basin discovery to discover robust, generalized minima.

## Local Setup

```bash
git clone https://github.com/TheRedDeveloper/flavri
cd flavri

conda env create -f environment.yml
conda activate flavri

pip install -e .

python evaluate_flavri_complete.py --checkpoint checkpoints/flavri-0.8.pt
```