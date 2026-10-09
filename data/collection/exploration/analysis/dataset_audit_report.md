# FLAVRI Chemosensory Dataset Aggregation & Exploration Audit Report

**Date:** 2026-10-09  
**Location:** `flavri/data/collection/exploration`  
**Status:** Comprehensive Multi-Dataset Ingestion & Exploration Complete  
**Engine Scope:** Unified 11D Taste & Chemesthesis Engine  

---

## 1. Executive Summary: What Has Been Collected

We have systematically gathered **14 dedicated raw data repositories** covering single molecules, taste oligopeptides, trigeminal chemesthetic agents, and empirical synergy/mixture interaction matrices.

### Overview of Collected Assets:

| Category | Dataset Name | Scale / Records | Format | Primary Endpoints | Usability for Next Agent |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mixture Synergy & Masking** | **TastePeptides Synergy** | **930 pairs** | CSV | Umami (561), Salt (152), Kokumi (92), Bitter-Masking (42); Added Conc & Threshold (mM) | **Direct Stage 2 Training** |
| **Mixture Synergy & Masking** | **Yamaguchi Synergy Matrix** | 64 pairs | CSV | MSG $\\times$ IMP factorial grid; Perceived intensity & synergy delta | **Stage 2 Curve Calibration** |
| **Mixture Suppression** | **Keast & Breslin Matrix** | 25 pairs | CSV | Sucrose $\\times$ Citric acid grid; Sweet & sour mutual suppression | **Stage 2 Suppression Calibration** |
| **Mixture Masking** | **Breslin & Beauchamp Matrix** | 5 benchmark sets | CSV | 4 Sodium salts $\\times$ 8 Bitterants; peripheral TAS2R suppression | **Stage 2 Masking Calibration** |
| **Taste Peptides** | **TastePeptidesDB (TPDB)** | **2,939 peptides** | CSV | Umami (1,188), Bitter (917), Sour (287), Sweet (205), Salty (139), Kokumi (69), Astringent (122); Threshold (mmol/L) | **Peptide Multi-Label Target** |
| **Taste Peptides** | **TPDB Derivatives** | 975 peptides | CSV | Modified peptides with SMILES, intrinsic taste & thresholds | **SMILES Peptide Target** |
| **Taste Peptides** | **UmamiMeta 2024** | 1,580 peptides | TSV | 972 umami positive, 608 controls; Length 1–39; TAS1R pocket coords | **Umami Sequence Benchmark** |
| **Small Molecules** | **ChemTastesDB v2.1** | 4,078 molecules | XLSX | Bitter (1,615), Sweet (1,313), Umami (220), Sour (49), Salty (16), Tasteless (232), Multitaste (179) | **Gold-Standard 5-Taste Target** |
| **Small Molecules** | **BitterDB 2024 Release** | 2,250 compounds | CSV | Bitterness across 25 human TAS2Rs; 2,231 pairs with EC50 & thresholds | **Bitter Target & Affinity** |
| **Molecules & Peptides**| **BitterGNN Multi-Flavor** | 5,456 instances | TSV | Sweet (2,263), Bitter (1,761), Umami (401), Kokumi (300), Astringent (332), Sour (75), Salt (37) | **Kokumi & Astringency Target** |
| **Molecules & Peptides**| **VirtuousUmami** | 442 compounds | CSV | Standardized SMILES + Sequences; strict applicability domain | **Curated Umami Benchmark** |
| **Potency Benchmark** | **SweetenersDB** | 316 sweet tastants | CSV | Relative sweetness log(S) referenced to sucrose = 1.0 | **Continuous Sweetness Potency** |
| **Trigeminal: Tingling** | **Leibniz Alkylamides Suite**| 14 alkylamides | CSV | Sanshools, spilanthol, bungeanool; Tingling thresholds (umol/L) & TASK-1/3 K2P inhibition % | **11th Dim (Tingling) Benchmark** |
| **Trigeminal: Cooling** | **Synthetic Coolants Comp.**| 12 coolants | CSV | WS-3, WS-5, WS-12, WS-23, Frescolat, Icilin; relative potency to menthol | **Cooling (TRPM8) Benchmark** |
| **Trigeminal: ChEMBL** | **ChEMBL Sensory Channels** | 770 bioactivities | CSV | TRPM8 (198), TRPV1 (189), TRPA1 (184), CaSR (199); nM bioactivities with SMILES | **Continuous Chemesthesis Affinities** |
| **Gustation: Sourness** | **Acid Speciation Benchmark** | 18 acids | CSV | Citric, Malic, Acetic, etc.; pKa values, [HA] equilibria, detection/recognition thresholds | **Mechanistic Sourness Grounding** |
| **Synthetic Enhancers** | **Senomyx Amide Library** | 6 synthetic amides | CSV | S6973, S2383, S807; T1R1/T1R3 & T1R2/T1R3 EC50s (uM) | **Non-peptide PAM Benchmark** |
| **Regulatory Dictionary**| **JECFA Flavor Catalog** | 6,552 compounds | RDS | 4,633 Flavouring Agents; FEMA, CAS, JECFA, COE numbers | **Entity Resolution Dictionary** |

---

## 2. Modality Coverage in the 11D Flavor Space

$$\vec{T}_{11} = [\underbrace{\text{sweet}, \, \text{salty}, \, \text{sour}, \, \text{bitter}, \, \text{umami}}_{\text{Basic Gustation (5D)}}, \; \underbrace{\text{kokumi}, \, \text{astringent}}_{\text{Oral Mouthfeel (2D)}}, \; \underbrace{\text{cooling}, \, \text{heat}, \, \text{pungency}, \, \text{tingling}}_{\text{Trigeminal Chemesthesis (4D)}}]$$

### Evidence Matrix Across Aggregated Datasets:

1. **Sweet**: ChemTastesDB (1,313), Multi-Flavor (2,263), TPDB (205), SweetenersDB (316 continuous potencies).
2. **Bitter**: ChemTastesDB (1,615), BitterDB 2024 (2,250), Multi-Flavor (1,761), TPDB (917).
3. **Umami**: UmamiMeta (1,580), TPDB (1,188), Multi-Flavor (401), VirtuousUmami (442), ChemTastesDB (220).
4. **Sour**: TPDB (287), Multi-Flavor (75), ChemTastesDB (49), Acid Speciation Matrix (18 acids with pKa and thresholds).
5. **Salty**: TPDB (139 peptides), SaltyMeta (280 peptides), Multi-Flavor (37), ChemTastesDB (16 salts).
6. **Kokumi**: Multi-Flavor (300 molecules/peptides), TPDB (69), TPDB Synergy (92 kokumi-enhancing pairs), ChEMBL CaSR (199 bioactivities).
7. **Astringent**: Multi-Flavor (332), TPDB (122), Tannin oligomer precipitation data.
8. **Cooling (TRPM8)**: ChEMBL TRPM8 (198 bioactivities), Synthetic Cooling Compendium (12 WS coolants).
9. **Heat (TRPV1)**: ChEMBL TRPV1 (189 bioactivities), Capsaicinoid activation records.
10. **Pungency (TRPA1)**: ChEMBL TRPA1 (184 bioactivities), Electrophilic covalent irritants.
11. **Tingling ($K_{2\\text{P}}$)**: Leibniz-LSB Alkylamides Suite (14 sanshools, spilanthol, pellitorine with tingling thresholds and TASK channel inhibition).

---

## 3. Synergy, Masking & Blend Data Coverage (Stage 2)

We have solved the "No Taste BP" mixture paradox:
* **TastePeptides Synergy Corpus**: 930 quantitative binary mixture pairs covering umami cooperativity, salt enhancement, kokumi thickness, and bitter masking.
* **Psychophysical Ground-Truth Matrices**: Full concentration grids for the classic non-linear interactions:
  * **Yamaguchi Umami Synergistic Cooperativity**: 64 pairs ($y = u + 1200 u v$).
  * **Keast & Breslin Sweet-Sour Mutual Suppression**: 25 pairs.
  * **Breslin & Beauchamp Sodium-Induced Bitter Masking**: 5 benchmark sets.
  * **Sensoproteomics Umami Synergy Matrix**: fold-increase enhancement factors.

---

## 4. Handover Notes for the Next Parsing Agent

1. **File Locations**: All raw data files are organized in `flavri/data/collection/exploration/raw/<dataset_name>/`.
2. **SMILES Resolution**:
   * Small molecules and modified peptides (ChemTastesDB, BitterDB, Multi-Flavor, ChEMBL, TPDB Derivatives, Cooling, Amides) already have **Canonical SMILES**.
   * Unmodified peptide sequences (UmamiMeta, TPDB, SaltyMeta) provide 1-letter FASTA sequences. Use RDKit's `Chem.MolFromSequence()` or standard peptide linear SMILES generation.
3. **Ionic Salts**: Retain dot notation (`.`) for mineral salts (`[Na+].[Cl-]`) and organic salts (MSG `[Na+].[O-]C(=O)...`). Do not strip the cation; the GNN's disjoint graph pooling naturally represents the complete salt as a single latent vector $z_{\text{compound}}$.
4. **Masked Multi-Task Loss**: Use the 11D binary mask $m \in \{0, 1\}^{11}$ per compound, allowing each molecule to train only on its observed sensory channels.
