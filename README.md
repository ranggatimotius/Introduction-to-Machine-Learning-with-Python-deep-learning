# Introduction to Machine Learning with Python

> Code reproduction and theoretical deep-dive based on *Introduction to Machine Learning with Python: A Guide for Data Scientists* by Andreas C. Müller and Sarah Guido (O'Reilly Media, 2016).

This repository reproduces the core workflows from all eight chapters using modern Python and scikit-learn. Every notebook combines a chapter summary, an original theoretical explanation, executable examples, visualizations, and key takeaways. All notebook outputs are committed so the results can be reviewed directly on GitHub.

## Repository Structure

```text
.
|-- Chapter_01_Introduction/
|   `-- notebook.ipynb
|-- Chapter_02_Supervised_Learning/
|   `-- notebook.ipynb
|-- Chapter_03_Unsupervised_Learning_and_Preprocessing/
|   `-- notebook.ipynb
|-- Chapter_04_Representing_Data_and_Engineering_Features/
|   `-- notebook.ipynb
|-- Chapter_05_Model_Evaluation_and_Improvement/
|   `-- notebook.ipynb
|-- Chapter_06_Algorithm_Chains_and_Pipelines/
|   `-- notebook.ipynb
|-- Chapter_07_Working_with_Text_Data/
|   `-- notebook.ipynb
|-- Chapter_08_Wrapping_Up/
|   `-- notebook.ipynb
|-- scripts/
|   `-- build_notebooks.py
|-- tests/
|   `-- test_notebooks.py
|-- requirements.txt
`-- README.md
```

## Chapter Summaries

### Chapter 1 - Introduction

Introduces supervised and unsupervised learning, the scientific Python ecosystem, the sample-feature-target vocabulary, train/test separation, exploratory visualization, and the first end-to-end classifier using k-nearest neighbors on the Iris dataset.

### Chapter 2 - Supervised Learning

Explains classification, regression, generalization, underfitting, and overfitting. The notebook compares k-NN, logistic regression, Naive Bayes, decision trees, random forests, gradient boosting, kernel SVMs, and multilayer perceptrons. It also demonstrates regularized regression, feature importance, and predicted probabilities.

### Chapter 3 - Unsupervised Learning and Preprocessing

Covers StandardScaler, MinMaxScaler, RobustScaler, PCA, NMF, and t-SNE. Clustering experiments compare k-means, agglomerative clustering, and DBSCAN using both internal and external evaluation metrics.

### Chapter 4 - Representing Data and Engineering Features

Shows how numeric representation changes what a model can learn. Topics include one-hot encoding, column-wise preprocessing, discretization, polynomial interactions, nonlinear transformations, feature selection, and domain-informed features.

### Chapter 5 - Model Evaluation and Improvement

Develops a reliable evaluation workflow using stratified cross-validation, grid search, a held-out test set, confusion matrices, classification reports, ROC and precision-recall curves, multiclass F1, and regression metrics. The notebook emphasizes choosing metrics from the real cost of errors.

### Chapter 6 - Algorithm Chains and Pipelines

Demonstrates how pipelines prevent leakage by fitting transformations only within training folds. It covers named steps, joint tuning of preprocessing and model hyperparameters, and grid-searching across different estimator families.

### Chapter 7 - Working with Text Data

Builds text classifiers with bag-of-words, stop-word filtering, TF-IDF, and word n-grams. It inspects learned coefficients and finishes with Latent Dirichlet Allocation for exploratory topic modeling.

### Chapter 8 - Wrapping Up

Connects modeling to deployment: problem formulation, data contracts, custom scikit-learn transformers, human review for uncertain cases, drift monitoring, reproducibility, and responsible maintenance.

## Setup

Python 3.12 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open any chapter directory and run `notebook.ipynb` from top to bottom.

## Rebuild and Verify

The notebooks use deterministic random seeds and datasets included with scikit-learn or generated locally. No external dataset download is required.

```powershell
python scripts/build_notebooks.py
python -m unittest discover -s tests -v
```

The build script recreates and executes all notebooks. The validation suite checks that every chapter exists, contains the required explanatory sections and code, has saved execution counts, and contains no execution-error output.

## Compatibility Notes

The original book used older versions of Python and scikit-learn. This reproduction updates deprecated interfaces and uses current pipeline and preprocessing APIs. The custom `mglearn` helper package is intentionally not required; equivalent plots are created directly with pandas, matplotlib, and scikit-learn so the repository remains easier to install and reproduce.

## Reference

- Andreas C. Müller and Sarah Guido. *Introduction to Machine Learning with Python: A Guide for Data Scientists*. O'Reilly Media, 2016.
- [Official book page](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/)
- [scikit-learn documentation](https://scikit-learn.org/stable/)

This repository was prepared for educational coursework. Explanations are paraphrased and the examples are adapted for a modern, self-contained environment.
