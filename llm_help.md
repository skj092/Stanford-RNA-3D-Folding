# Stanford RNA 3D Folding

## Solve RNA structure prediction, one of biology's remaining grand challenges

## Overview

If you sat down to complete a puzzle without knowing what it should look like, you’d have to rely on patterns and logic to piece it together. In the same way, predicting Ribonucleic acid (RNA)’s 3D structure involves using only its sequence to figure out how it folds into the structures that define its function.

In this competition, you’ll develop machine learning models to predict an RNA molecule’s 3D structure from its sequence. The goal is to improve our understanding of biological processes and drive new advancements in medicine and biotechnology.

## Description

RNA is vital to life’s most essential processes, but despite its significance, predicting its 3D structure is still difficult. Deep learning breakthroughs like AlphaFold have transformed protein structure prediction, but progress with RNA has been much slower due to limited data and evaluation methods.
This competition builds on recent advances, like the deep learning foundation model RibonanzaNet, which emerged from a prior Kaggle competition. Now, you’ll take on the next challenge—predicting RNA’s full 3D structure.
Your work could push RNA-based medicine forward, making treatments like cancer immunotherapies and CRISPR gene editing more accessible and effective. More fundamentally, your work may be the key step in illuminating the folds and functions of natural RNA molecules, which have been called the 'dark matter of biology'.
This competition is made possible through a worldwide collaborative effort including the organizers, experimental RNA structural biologists, and predictors of the CASP16 and RNA-Puzzles competitions; Howard Hughes Medical Institute; the Institute of Protein Design; and Stanford University School of Medicine.

## Evaluation

Submissions are scored using TM-score ("template modeling" score), which goes from 0.0 to 1.0 (higher is better):

TM-score=max⎛⎝⎜⎜1Lref∑i=1Lalign11+(did0)2⎞⎠⎟⎟

where:

Lref is the number of residues solved in the experimental reference structure ("ground truth").
Lalign is the number of aligned residues.
di is the distance between the ith pair of aligned residues, in Angstroms.
d0 is a distance scaling factor in Angstroms, defined as:
d0=0.6(Lref−0.5)1/2−2.5
for Lref ≥ 30; and d0 = 0.3, 0.4, 0.5, 0.6, or 0.7 for Lref <15, 12-15, 16-19, 20-23, or 24-29, respectively.
The rotation and translation of predicted structures to align with experimental reference structures are carried out by US-align. To match default settings, as used in the CASP competitions, the alignment will be sequence-independent.
For each target RNA sequence, you will submit 5 predictions and your final score will be the average of best-of-5 TM-scores of all targets. For a few targets, multiple slightly different structures have been captured experimentally; your predictions' scores will be based on the best TM-score compared to each of these reference structures.


### Submission File
For each sequence in the test set, you can predict five structures. Your notebook should look for a file test_sequences.csv and output submission.csv. This file should contain x, y, z coordinates of the C1' atom in each residue across your predicted structures 1 to 5:

```csv
ID,resname,resid,x_1,y_1,z_1,... x_5,y_5,z_5
R1107_1,G,1,-7.561,9.392,9.361,... -7.301,9.023,8.932
R1107_2,G,1,-8.02,11.014,14.606,... -7.953,10.02,12.127
etc.
```


## Dataset Description
In this competition you will predict five 3D structures for each RNA sequence.

## Competition Phases and Updates
This is a code competition that will proceed in three phases.

- Initial model training phase: At launch, expect approximately 25 sequences in the hidden test set. Some of those sequences are used for a private leaderboard to allow the host to track progress on wholly unseen data. During this phase the public test set sequences includes–but is not limited to–targets from the 2024 CASP16 competition whose structures have not yet been publicly released in the PDB database.
- Model training phase 2: On April 23rd we will update the hidden test set and reset the leaderboard. Sequences in the current public test set will be added to the train data, all sequences currently in the private set will be rolled into the new public set, and new sequences will be added to the public test set.
- Future data phase: Your selected submissions will be run against a completely new private test set generated after the end of the model training phases. There will be up to 40 sequences in the test set, all of them used for the private leaderboard.

## Files
### [train/validation/test]_sequences.csv - the target sequences of the RNA molecules.

target_id - (string) An arbitrary identifier. In train_sequences.csv, this is formatted as pdb_id_chain_id, where pdb_id is the id of the entry in the Protein Data Bank and chain_id is the chain id of the monomer in the pdb file.
sequence - (string) The RNA sequence. For test_sequences.csv, this is guaranteed to be a string of A, C, G, and U. For some train_sequences.csv, other characters may appear.
temporal_cutoff - (string) The date in yyyy-mm-dd format that the sequence was published. See Additional Notes.
description - (string) Details of the origins of the sequence. For a few targets, additional information on small molecule ligands bound to the RNA is included. You don't need to make predictions for these ligand coordinates.
all_sequences - (string) FASTA-formatted sequences of all molecular chains present in the experimentally solved structure. In a few cases this may include multiple copies of the target RNA (look for the word "Chains" in the header) and/or partners like other RNAs or proteins or DNA. You don't need to make predictions for all these molecules; if you do, just submit predictions for sequence. Some entries are blank.

### [train/validation]_labels.csv - experimental structures.

- ID - (string) that identifies the target_id and residue number, separated by _. Note: residue numbers use one-based indexing.
- resname - (character) The RNA nucleotide ( A, C, G, or U) for the residue.
- resid - (integer) residue number.
- x_1,y_1,z_1,x_2,y_2,z_2,… - (float) Coordinates (in Angstroms) of the C1' atom for each experimental RNA structure. There is typically one structure for the RNA sequence, and train_labels.csv curates one structure for each training sequence. However, in some targets the experimental method has captured more than one conformation, and each will be used as a potential reference for scoring your predictions. validation_labels.csv has examples of targets with multiple reference structures (x_2,y_2,z_2, etc.).

### sample_submission.csv
- Same format as train_labels.csv but with five sets of coordinates for each of your five predicted structures (x_1,y_1,z_1,x_2,y_2,z_2,…x_5,y_5,z_5).
- You must submit five sets of coordinates.

## Additional notes
- The validation_sequences.csv and test_sequences.csv publicly provided here comprise 12 targets from the 2022 CASP15 competition which have been a widely used test set in the RNA modeling field.
- If you choose to use the provided 12 CASP15 targets in validation_sequences.csv for validation, make sure that you train only on train_sequences.csv that have temporal_cutoff before the test_sequences (2022-05-27 is a safe date). If you wish, you can use train_sequences.csv with temporal_cutoff after this date as an additional validation set.
- Once you begin hill climbing on the competition's actual Public Leaderboard, you can use all the train_sequences.csv and indeed all 3D structural information that you can find in the PDB database, since the competition's actual leaderboard targets are not released in the PDB database. However, note that the 12 CASP15 targets provided here in validation_sequences.csv will be 'burned' since they will be in your training set.
- RNA chains from the same or different PDB entries that share sequence are given as different entries in train_sequences.csv. You may consider deduplicating these entries and merging the various available structures into additional x_2,y_2,z_2, etc. labels, as has been done with validation_sequences.csv
- If you use RibonanzaNet (as in the competition starting notebook) it does not use information from the PDB before CASP15 and so is expected to be valid for use for all test sets. If you are using other neural networks, make sure to check their temporal cutoffs for training data.
- If you are prompting a large language model you should request information that is available before the temporal_cutoff for each target. Otherwise, information from preprints or blog posts on CASP16 targets that were released after CASP16 competition end (2024-09-18) may leak into your submissions, and you will get a Public Leaderboard score that may be deceptively inflated compared to the CASP16 expert baseline or your eventual Private Leaderboard score. Only notebooks that beat the CASP16 expert baseline while also paying close attention to temporal_cutoff will be eligible for the Early Sharing prizes!

------------------

# ChatGPT Help: https://chatgpt.com/c/67c295ee-1aa8-800f-b5ab-180a8115327a
Since you don't have a background in RNA, let's break this problem down step by step:

## **1. Understanding the Problem**
RNA (Ribonucleic Acid) is a molecule that plays a crucial role in biological processes. Its function is largely determined by its 3D structure. However, predicting this structure from a sequence of nucleotides (A, C, G, U) is a major unsolved problem in computational biology.

### **Key Challenge**
- Given an RNA sequence, predict the **3D coordinates of the C1' atom** for each nucleotide.
- Each test sequence requires **five different predictions** (diverse folding possibilities).
- Evaluation is based on **TM-score**, a measure of structural similarity between the predicted and experimental structures.

---

## **2. Literature Study**
To understand the problem and existing approaches, you'll need to cover:
### **A. Basics of RNA Structure**
- RNA is made of **four nucleotides (A, C, G, U)**.
- It folds into **secondary structures** (like loops and stems) and **tertiary structures** (3D configurations).
- The folding is driven by **hydrogen bonding and stacking interactions**.

**Useful Reading:**
- [RNA Secondary Structure Wikipedia](https://en.wikipedia.org/wiki/Nucleic_acid_secondary_structure)
- [RNA 3D Structure Prediction Review](https://doi.org/10.1146/annurev-biophys-070816-033639)

### **B. Methods for RNA 3D Structure Prediction**
- **Physics-based methods**: Molecular dynamics simulations (expensive but accurate).
- **Template-based methods**: Use known RNA structures as templates.
- **Machine learning methods**: Predict 3D coordinates directly from sequences.

**Relevant Papers:**
- [Rosetta for RNA 3D structure](https://doi.org/10.1038/s41592-018-0180-2)
- [Deep learning for RNA structure prediction](https://doi.org/10.1038/s41592-022-01576-w)

### **C. Related Models**
- **AlphaFold** (for proteins) inspired many RNA models.
- **RibonanzaNet** is a recent deep-learning-based RNA structure predictor.

---

## **3. Building a Baseline Model**
We need a simple model that generates RNA 3D structures. Here’s a structured approach:

### **Step 1: Explore the Data**
- Load `train_sequences.csv` and `train_labels.csv`.
- Visualize a few RNA 3D structures using libraries like **PyMOL** or **PyMOL-RNA**.

### **Step 2: Feature Engineering**
- Convert RNA sequences into numerical representations (e.g., one-hot encoding, embeddings).
- Use **secondary structure predictors** (like RNAfold) to get intermediate structures.

### **Step 3: Model Selection**
A few approaches to start:
1. **Graph Neural Networks (GNNs)** – RNA structure can be modeled as a graph.
2. **Transformer-based models** – Adapt large protein models for RNA sequences.
3. **Diffusion models** – Generative models can be used for RNA 3D conformation sampling.

A simple baseline:
- **Train a 3D coordinate regression model** using RibonanzaNet or a lightweight CNN+LSTM.
- **Use geometric loss functions** to improve structure accuracy.

### **Step 4: Generate Predictions**
- Predict 3D coordinates for test sequences.
- Generate **five diverse structures** per sequence (e.g., via dropout in a neural network).

### **Step 5: Evaluate using TM-score**
- Use **US-align** to compare predictions to ground truth.

---

## **4. Next Steps**
1. **Explore the dataset** – Write a script to load and visualize RNA structures.
2. **Try an existing model** – Start with RibonanzaNet to generate baseline predictions.
3. **Improve predictions** – Use secondary structure information, augment training data.
4. **Experiment with ML architectures** – Implement GNNs or transformers for better modeling.
