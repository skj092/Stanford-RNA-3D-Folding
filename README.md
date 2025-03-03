
1. Understanding: https://www.kaggle.com/code/ayushs9020/understanding-the-competition-standford-ribonaza
2. KNOT Theory: https://www.youtube.com/watch?v=8DBhTXM_Br4


# Solution:
project/
├── main.py                # Main script to run the entire pipeline
├── config.py              # Configuration and hyperparameters
├── requirements.txt       # Package dependencies
├── README.md              # Project documentation
├── data/
│   ├── __init__.py
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── dataset.py         # Custom dataset and dataloader classes
│   └── feature_engineering.py  # Feature extraction and engineering
├── models/
│   ├── __init__.py
│   ├── model.py           # Neural network architecture
│   └── ensemble.py        # Ensemble modeling
├── training/
│   ├── __init__.py
│   ├── train.py           # Training loop and validation
│   └── loss.py            # Custom loss functions
├── utils/
│   ├── __init__.py
│   ├── metrics.py         # Evaluation metrics (TM-score)
│   ├── secondary_structure.py  # RNA secondary structure prediction
│   └── visualization.py   # Visualization utilities
└── inference/
    ├── __init__.py
    ├── predict.py         # Model inference
    └── submission.py      # Submission file generation
