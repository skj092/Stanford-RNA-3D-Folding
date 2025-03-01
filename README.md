
1. Understanding: https://www.kaggle.com/code/ayushs9020/understanding-the-competition-standford-ribonaza
2. KNOT Theory: https://www.youtube.com/watch?v=8DBhTXM_Br4
    1. How to verify whether two know are same?



# Gemini Help To Understanding Data:


## Understanding the Problem: RNA 3D Structure Prediction
- Like DNA, RNA is made up of a sequence of nucleotides: Adenine (A), Cytosine (C), Guanine (G), and Uracil (U).
- The sequence of these nucleotides determines how the RNA molecule folds into a specific 3D shape.
- The sequence of these nucleotides determines how the RNA molecule folds into a specific 3D shape.

### Challenge:
- Predicting the 3D structure of an RNA molecule solely from its sequence is a complex problem.
- Develop a machine learning model that can accurately predict the 3D coordinates of the RNA molecule's atoms (specifically the C1' atom) based on its sequence.

## Understanding the Dataset:
1. train_sequences.csv:
    1. target_id:
    2. sequence:
    3. temporalcutoff:
    4. description:
    5. all_sequences:


set(train_sequences.target_id.to_list()) == set(train_labels.pdb_id.to_list()) = True



## Key Data Concepts:
Residue: A single nucleotide in the RNA sequence.
C1' Atom: A specific atom within each residue used as a reference point for 3D structure prediction.
3D Coordinates (x, y, z): Represent the position of the C1' atom in 3D space.
TM-score: The evaluation metric. A higher TM-score indicates a better prediction of the RNA's 3D structure.


