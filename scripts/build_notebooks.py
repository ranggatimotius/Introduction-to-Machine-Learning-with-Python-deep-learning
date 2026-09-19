"""Build and execute the eight chapter notebooks for the course assignment."""

import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
MATPLOTLIB_CACHE = ROOT / ".matplotlib"
MATPLOTLIB_CACHE.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE)


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def chapter_1():
    return [
        md(r"""
# Chapter 1: Introduction

## Chapter Summary
This chapter frames machine learning as learning decision rules from examples rather than hand-coding every rule. It introduces supervised and unsupervised learning, the scientific Python stack, the sample-feature-target vocabulary, and a complete first classification workflow using the Iris dataset.

## Theoretical Deep-Dive

### Learning from data
In supervised learning, observations are pairs $(x_i, y_i)$. A model learns a function $f(x)$ that should generalize to unseen observations. Classification predicts a discrete label; regression predicts a continuous value. Unsupervised learning receives only $x_i$ and searches for structure such as clusters or lower-dimensional representations.

### Generalization and the holdout principle
Training accuracy measures memory as well as learning. A separate test set estimates performance on unseen data. The test set must not influence model selection or preprocessing.

### k-nearest neighbors
For a new sample, k-NN finds the $k$ closest training samples and predicts the majority class. Small $k$ creates a flexible boundary; larger $k$ smooths the boundary. Distance-based models are sensitive to feature scale.

**Book coverage:** scientific Python tools, Iris inspection, train/test split, k-NN prediction, and evaluation (Chapter 1, printed pp. 1-23).
"""),
        code("""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from scipy import sparse

print(f"Python: {sys.version.split()[0]}")
print(f"NumPy: {np.__version__} | pandas: {pd.__version__} | scikit-learn: {sklearn.__version__}")

eye = np.eye(4)
print("Dense identity matrix:")
print(eye)
print("Sparse CSR representation:")
print(sparse.csr_matrix(eye))
"""),
        code("""
from sklearn.datasets import load_iris

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
print("Dataset shape:", iris_df.shape)
display(iris_df.head())
display(iris_df.groupby("species", observed=True).mean().round(2))
"""),
        code("""
axes = pd.plotting.scatter_matrix(
    iris_df[iris.feature_names], c=iris.target, figsize=(9, 9),
    marker="o", hist_kwds={"bins": 15}, s=35, alpha=0.75
)
plt.suptitle("Iris feature relationships", y=0.9)
plt.show()
"""),
        code("""
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, stratify=iris.target, random_state=42
)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Predicted species for first test row:", iris.target_names[y_pred[0]])
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names)
plt.title("Iris k-NN confusion matrix")
plt.show()
"""),
        md("""
## Key Takeaways
- Start with the question and the data, not an algorithm.
- Rows are samples, columns are features, and the target is what supervised learning predicts.
- Keep test data isolated until final evaluation.
- Inspect distributions and relationships before modeling.
- A compact scikit-learn workflow is `split -> fit -> predict -> evaluate`.
"""),
    ]


def chapter_2():
    return [
        md(r"""
# Chapter 2: Supervised Learning

## Chapter Summary
Supervised learning maps features to known targets. This chapter compares major classifiers and regressors, connects model complexity to underfitting and overfitting, and demonstrates why preprocessing and uncertainty estimates matter.

## Theoretical Deep-Dive

### Bias, variance, and generalization
An overly simple model has high bias and underfits. An overly flexible model has high variance and may memorize noise. Validation performance, not training performance alone, guides model complexity.

### Model families
- **k-NN:** local, nonparametric, and scale-sensitive.
- **Linear models:** predictions are weighted sums; regularization constrains coefficients.
- **Naive Bayes:** fast probabilistic models with conditional-independence assumptions.
- **Trees and ensembles:** nonlinear decision rules; ensembles reduce the instability of one tree.
- **Kernel SVM:** nonlinear boundaries through similarity features.
- **Neural networks:** compositions of linear transformations and nonlinear activations.

For binary logistic regression, $P(y=1|x)=\sigma(w^Tx+b)$, where $\sigma(z)=1/(1+e^{-z})$. L2 regularization penalizes large weights and trades training fit for stability.

**Book coverage:** classification/regression, generalization, k-NN, linear models, Naive Bayes, trees, ensembles, SVM, neural networks, and classifier uncertainty (Chapter 2, printed pp. 25-127).
"""),
        code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42
)
print("Cancer data:", cancer.data.shape)
print("Class counts:", dict(zip(cancer.target_names, np.bincount(cancer.target))))
"""),
        code("""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rows = []
for k in [1, 3, 7, 15]:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    knn.fit(X_train, y_train)
    rows.append((f"k-NN (k={k})", knn.score(X_train, y_train), knn.score(X_test, y_test)))

for c in [0.01, 1, 100]:
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
    logistic.fit(X_train, y_train)
    rows.append((f"Logistic (C={c})", logistic.score(X_train, y_train), logistic.score(X_test, y_test)))

display(pd.DataFrame(rows, columns=["model", "train_accuracy", "test_accuracy"]).round(3))
"""),
        code("""
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

models = {
    "Gaussian NB": GaussianNB(),
    "Decision tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "Gradient boosting": GradientBoostingClassifier(random_state=42),
    "RBF SVM": make_pipeline(StandardScaler(), SVC(C=2, random_state=42)),
    "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(40,), max_iter=1500, random_state=42)),
}
comparison = []
for name, estimator in models.items():
    estimator.fit(X_train, y_train)
    comparison.append((name, estimator.score(X_train, y_train), estimator.score(X_test, y_test)))
display(pd.DataFrame(comparison, columns=["model", "train_accuracy", "test_accuracy"]).round(3))
"""),
        code("""
X_reg, y_reg = make_regression(n_samples=180, n_features=30, n_informative=8, noise=18, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, random_state=42)
regression_results = []
for name, estimator in {
    "Ridge alpha=1": Ridge(alpha=1),
    "Ridge alpha=10": Ridge(alpha=10),
    "Lasso alpha=1": Lasso(alpha=1, max_iter=5000),
}.items():
    estimator.fit(Xr_train, yr_train)
    prediction = estimator.predict(Xr_test)
    regression_results.append((name, mean_squared_error(yr_test, prediction) ** 0.5, estimator.score(Xr_test, yr_test)))
display(pd.DataFrame(regression_results, columns=["model", "RMSE", "R2"]).round(3))
"""),
        code("""
best = models["Random forest"]
probabilities = best.predict_proba(X_test[:5])
uncertainty = pd.DataFrame(probabilities, columns=cancer.target_names)
uncertainty["predicted"] = cancer.target_names[best.predict(X_test[:5])]
display(uncertainty.round(3))

forest = models["Random forest"]
top = np.argsort(forest.feature_importances_)[-10:]
plt.barh(cancer.feature_names[top], forest.feature_importances_[top])
plt.xlabel("Feature importance")
plt.title("Random forest: ten most important features")
plt.tight_layout(); plt.show()
"""),
        md("""
## Key Takeaways
- Training score alone cannot diagnose generalization.
- Hyperparameters such as `k`, `C`, depth, and regularization strength control complexity.
- Scaling is essential for distance-, margin-, and gradient-based models.
- Ensembles usually generalize better than a single unrestricted tree.
- Probabilities and decision scores express confidence, but they are not automatically calibrated truth.
"""),
    ]


def chapter_3():
    return [
        md(r"""
# Chapter 3: Unsupervised Learning and Preprocessing

## Chapter Summary
Unsupervised learning works without target labels. This chapter covers scaling, PCA, NMF, t-SNE, and clustering with k-means, agglomerative clustering, and DBSCAN. Labels are used only after fitting to evaluate how discovered structure relates to known classes.

## Theoretical Deep-Dive

Scaling prevents large-unit features from dominating distance and variance. Standardization applies $z=(x-\mu)/\sigma$, with parameters learned from training data only.

PCA finds orthogonal directions of maximum variance and projects data onto them. NMF approximates a nonnegative matrix as $X \approx WH$, producing additive parts. t-SNE preserves local neighborhoods for visualization but should not be treated as a predictive transform.

Clustering methods encode different assumptions: k-means favors compact spherical clusters, hierarchical clustering builds nested merges, and DBSCAN identifies dense regions plus noise without specifying the number of clusters.

**Book coverage:** preprocessing, PCA, NMF, t-SNE, clustering, and cluster evaluation (Chapter 3, printed pp. 131-208).
"""),
        code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42
)
results = []
for scaler in [StandardScaler(), MinMaxScaler(), RobustScaler()]:
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = SVC(C=2).fit(Xtr, y_train)
    results.append((scaler.__class__.__name__, model.score(Xte, y_test)))
display(pd.DataFrame(results, columns=["preprocessing", "test_accuracy"]).round(3))
"""),
        code("""
from sklearn.decomposition import PCA, NMF

digits = load_digits()
X_scaled = StandardScaler().fit_transform(digits.data)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, cmap="tab10", s=12, alpha=0.7)
ax.set(title="Digits projected to two PCA components", xlabel="PC1", ylabel="PC2")
plt.colorbar(scatter, ax=ax, label="digit"); plt.show()
print("Explained variance ratio:", pca.explained_variance_ratio_.round(3))

nmf = NMF(n_components=10, init="nndsvda", max_iter=800, random_state=42)
W = nmf.fit_transform(digits.data)
print("NMF factor shapes:", W.shape, nmf.components_.shape)
"""),
        code("""
from sklearn.manifold import TSNE

subset = np.random.default_rng(42).choice(len(digits.data), size=500, replace=False)
embedding = TSNE(n_components=2, init="pca", learning_rate="auto", max_iter=750, random_state=42).fit_transform(digits.data[subset])
plt.figure(figsize=(8, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target[subset], cmap="tab10", s=16)
plt.title("t-SNE visualization of 500 digit samples")
plt.xticks([]); plt.yticks([]); plt.show()
"""),
        code("""
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score

X_blob, y_blob = make_blobs(n_samples=450, centers=4, cluster_std=[0.7, 1.1, 0.6, 0.9], random_state=42)
clusterers = {
    "k-means": KMeans(n_clusters=4, n_init=20, random_state=42),
    "agglomerative": AgglomerativeClustering(n_clusters=4),
    "DBSCAN": DBSCAN(eps=0.6, min_samples=6),
}
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
scores = []
for ax, (name, clusterer) in zip(axes, clusterers.items()):
    labels = clusterer.fit_predict(X_blob)
    valid = len(set(labels)) > 1
    scores.append((name, adjusted_rand_score(y_blob, labels), silhouette_score(X_blob, labels) if valid else np.nan, int((labels == -1).sum())))
    ax.scatter(X_blob[:, 0], X_blob[:, 1], c=labels, cmap="tab10", s=18)
    ax.set_title(name)
plt.tight_layout(); plt.show()
display(pd.DataFrame(scores, columns=["method", "ARI", "silhouette", "noise_points"]).round(3))
"""),
        md("""
## Key Takeaways
- Fit preprocessing only on training data, then reuse it on validation/test data.
- PCA is a linear variance-preserving projection; NMF learns additive nonnegative parts.
- t-SNE is primarily an exploratory visualization tool.
- Clustering has no universally correct algorithm; the geometry and density assumptions matter.
- Evaluate clusters with internal metrics and, when legitimate labels exist, external metrics such as ARI.
"""),
    ]


def chapter_4():
    return [
        md(r"""
# Chapter 4: Representing Data and Engineering Features

## Chapter Summary
Models operate on numeric representations, so feature design can matter more than model choice. This chapter covers categorical encoding, discretization, interactions, polynomial expansion, nonlinear transformations, feature selection, and the use of domain knowledge.

## Theoretical Deep-Dive

One-hot encoding creates one indicator per category without inventing an order. Binning converts continuous values into intervals and can give linear models nonlinear stepwise behavior. Polynomial features add powers and interactions, allowing a linear estimator to fit curved surfaces while remaining linear in its coefficients.

Transformations such as $\log(1+x)$ can make skewed distributions easier to model. Feature selection reduces noise and complexity: univariate methods score features independently, model-based methods use learned importance, and iterative methods repeatedly fit a model.

All learned transformations belong inside the train/validation workflow to prevent leakage.

**Book coverage:** categorical variables, binning, polynomial/interactions, transformations, selection, and expert features (Chapter 4, printed pp. 211-250).
"""),
        code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

customers = pd.DataFrame({
    "age": [22, 45, 36, 52, 29, 41, 63, 34, 27, 58, 47, 31],
    "income": [32, 85, 58, 95, 45, 72, 110, 54, 39, 99, 78, 49],
    "city": ["Jakarta", "Bandung", "Jakarta", "Surabaya", "Bandung", "Jakarta", "Surabaya", "Bandung", "Jakarta", "Surabaya", "Bandung", "Jakarta"],
    "premium": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
})
X = customers.drop(columns="premium"); y = customers["premium"]
preprocess = ColumnTransformer([
    ("numeric", StandardScaler(), ["age", "income"]),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])
encoded = preprocess.fit_transform(X)
print("Original shape:", X.shape, "Encoded shape:", encoded.shape)
print("Features:", preprocess.get_feature_names_out())
"""),
        code("""
from sklearn.datasets import make_regression
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
x = np.linspace(-3, 3, 180).reshape(-1, 1)
y_curve = x[:, 0] ** 3 - 2 * x[:, 0] ** 2 + rng.normal(0, 2, len(x))
Xtr, Xte, ytr, yte = train_test_split(x, y_curve, random_state=42)

models = {
    "linear": LinearRegression(),
    "binned": make_pipeline(KBinsDiscretizer(n_bins=8, encode="onehot-dense", strategy="uniform"), LinearRegression()),
    "polynomial degree 3": make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression()),
}
grid = np.linspace(-3, 3, 300).reshape(-1, 1)
for name, model in models.items():
    model.fit(Xtr, ytr)
    print(f"{name:20s} RMSE={mean_squared_error(yte, model.predict(Xte)) ** 0.5:.3f}")
    plt.plot(grid, model.predict(grid), label=name)
plt.scatter(Xtr, ytr, s=12, alpha=0.45, label="train")
plt.legend(); plt.title("Representation changes model capacity"); plt.show()
"""),
        code("""
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile, SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, stratify=data.target, random_state=42)
selectors = {
    "univariate 40%": SelectPercentile(percentile=40),
    "model-based": SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median"),
    "recursive (12)": RFE(LogisticRegression(max_iter=3000), n_features_to_select=12, step=0.2),
}
selection_rows = []
for name, selector in selectors.items():
    pipe = make_pipeline(StandardScaler(), selector, LogisticRegression(max_iter=3000))
    pipe.fit(X_train, y_train)
    selection_rows.append((name, selector.fit(X_train, y_train).get_support().sum(), pipe.score(X_test, y_test)))
display(pd.DataFrame(selection_rows, columns=["method", "selected_features", "test_accuracy"]).round(3))
"""),
        code("""
counts = rng.poisson(lam=3, size=1000)
skewed = np.exp(counts / 2)
transformed = np.log1p(skewed)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].hist(skewed, bins=30); axes[0].set_title("Skewed count-like feature")
axes[1].hist(transformed, bins=30); axes[1].set_title("After log1p transformation")
plt.tight_layout(); plt.show()

customers["income_per_age"] = customers["income"] / customers["age"]
display(customers[["age", "income", "income_per_age"]].head())
"""),
        md("""
## Key Takeaways
- Categorical codes are not numbers with meaningful distances; one-hot encode nominal categories.
- Binning and polynomial interactions change what a linear model can express.
- Transform highly skewed features when the model benefits from a more regular scale.
- Feature selection must be fitted inside each training fold.
- Domain-informed features can expose useful relationships unavailable to a generic algorithm.
"""),
    ]


def chapter_5():
    return [
        md(r"""
# Chapter 5: Model Evaluation and Improvement

## Chapter Summary
Reliable evaluation separates model development from final assessment. This chapter covers cross-validation, stratification, grid search, nested validation logic, and metrics for imbalanced binary, multiclass, and regression tasks.

## Theoretical Deep-Dive

k-fold cross-validation rotates the validation fold and summarizes variability across splits. Stratification preserves class proportions. Hyperparameters must be chosen inside cross-validation; reporting the best validation result as a final test estimate is optimistically biased.

For binary classification, precision is $TP/(TP+FP)$ and recall is $TP/(TP+FN)$. F1 is their harmonic mean. ROC-AUC ranks positives above negatives across thresholds; average precision is often more informative when positives are rare. The correct metric follows the real cost of mistakes.

**Book coverage:** cross-validation, grid search, binary/multiclass metrics, regression metrics, and metric-aware selection (Chapter 5, printed pp. 251-302).
"""),
        code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, stratify=data.target, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
display(pd.DataFrame(scores)[["test_accuracy", "test_f1", "test_roc_auc"]].round(3))
print("Mean accuracy +/- SD:", f"{scores['test_accuracy'].mean():.3f} +/- {scores['test_accuracy'].std():.3f}")
"""),
        code("""
from sklearn.model_selection import GridSearchCV

param_grid = {
    "logisticregression__C": np.logspace(-3, 2, 6),
    "logisticregression__class_weight": [None, "balanced"],
}
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=1, return_train_score=True)
search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)
print(f"Best cross-validation ROC-AUC: {search.best_score_:.3f}")
print(f"Held-out test ROC-AUC: {search.score(X_test, y_test):.3f}")
results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
display(results[["params", "mean_train_score", "mean_test_score", "std_test_score"]].head().round(3))
"""),
        code("""
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, RocCurveDisplay,
    PrecisionRecallDisplay, roc_auc_score, average_precision_score
)

y_pred = search.predict(X_test)
y_score = search.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred, target_names=data.target_names))
print(f"ROC-AUC={roc_auc_score(y_test, y_score):.3f} | Average precision={average_precision_score(y_test, y_score):.3f}")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=data.target_names, ax=axes[0])
RocCurveDisplay.from_predictions(y_test, y_score, ax=axes[1])
PrecisionRecallDisplay.from_predictions(y_test, y_score, ax=axes[2])
plt.tight_layout(); plt.show()
"""),
        code("""
from sklearn.datasets import load_digits, make_regression
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

digits = load_digits()
Xd_train, Xd_test, yd_train, yd_test = train_test_split(digits.data, digits.target, stratify=digits.target, random_state=42)
multi = RandomForestClassifier(n_estimators=120, random_state=42).fit(Xd_train, yd_train)
multi_pred = multi.predict(Xd_test)
print("Multiclass macro-F1:", round(f1_score(yd_test, multi_pred, average="macro"), 3))

Xr, yr = make_regression(n_samples=400, n_features=12, noise=20, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, random_state=42)
reg = RandomForestRegressor(n_estimators=120, random_state=42).fit(Xr_train, yr_train)
rp = reg.predict(Xr_test)
print(f"Regression MAE={mean_absolute_error(yr_test, rp):.2f}, RMSE={mean_squared_error(yr_test, rp) ** 0.5:.2f}, R2={r2_score(yr_test, rp):.3f}")
"""),
        md("""
## Key Takeaways
- Cross-validation estimates variability; a single split can be misleading.
- Tune hyperparameters with training folds and reserve a final test set.
- Stratify classification splits, especially with imbalance.
- Accuracy hides error types; inspect confusion matrices, precision, recall, F1, and threshold curves.
- Select metrics from operational goals, not convenience.
"""),
    ]


def chapter_6():
    return [
        md(r"""
# Chapter 6: Algorithm Chains and Pipelines

## Chapter Summary
Pipelines bind preprocessing and prediction into one estimator. They prevent leakage, make cross-validation correct by construction, and allow joint tuning of transformations and models.

## Theoretical Deep-Dive

If scaling or feature selection sees all observations before cross-validation, validation data influences training and scores become optimistic. A pipeline fits each transformation only on the current training fold, then applies the learned transformation to that fold's validation data.

A pipeline is compositional: each intermediate step implements `fit` and `transform`, while the final step implements `fit` and `predict`. Parameters are addressed with `step__parameter`, enabling a single search over preprocessing and model choices.

**Book coverage:** preprocessing with parameter selection, pipeline construction, grid search, step inspection, and model selection (Chapter 6, printed pp. 305-320).
"""),
        code("""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, stratify=data.target, random_state=42)
pipeline = Pipeline([("scale", StandardScaler()), ("svc", SVC(random_state=42))])
pipeline.fit(X_train, y_train)
print(f"Pipeline test accuracy: {pipeline.score(X_test, y_test):.3f}")
print("Named steps:", list(pipeline.named_steps))
print("First three learned means:", pipeline.named_steps["scale"].mean_[:3].round(2))
"""),
        code("""
from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": [0.001, 0.01, 0.1, "scale"],
}
search = GridSearchCV(pipeline, grid, cv=cv, scoring="roc_auc", n_jobs=1)
search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)
print(f"CV ROC-AUC: {search.best_score_:.3f}; test ROC-AUC: {search.score(X_test, y_test):.3f}")
"""),
        code("""
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegression

flexible = Pipeline([
    ("scale", StandardScaler()),
    ("reduce", "passthrough"),
    ("model", LogisticRegression(max_iter=3000)),
])
parameter_options = [
    {"reduce": [PCA(random_state=42)], "reduce__n_components": [5, 10, 15], "model__C": [0.1, 1, 10]},
    {"reduce": [SelectPercentile()], "reduce__percentile": [30, 60, 100], "model__C": [0.1, 1, 10]},
]
joint_search = GridSearchCV(flexible, parameter_options, cv=cv, scoring="accuracy", n_jobs=1)
joint_search.fit(X_train, y_train)
print("Best joint configuration:", joint_search.best_params_)
print(f"Test accuracy: {joint_search.score(X_test, y_test):.3f}")
"""),
        code("""
from sklearn.ensemble import RandomForestClassifier

model_selector = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])
model_grid = [
    {"scale": [StandardScaler()], "clf": [LogisticRegression(max_iter=3000)], "clf__C": [0.1, 1, 10]},
    {"scale": ["passthrough"], "clf": [RandomForestClassifier(random_state=42)], "clf__n_estimators": [80, 150], "clf__max_depth": [None, 5]},
]
model_search = GridSearchCV(model_selector, model_grid, cv=cv, scoring="accuracy", n_jobs=1).fit(X_train, y_train)
print("Selected estimator:", model_search.best_estimator_.named_steps["clf"])
print(f"Held-out accuracy: {model_search.score(X_test, y_test):.3f}")
"""),
        md("""
## Key Takeaways
- A pipeline is the default safe unit for preprocessing plus modeling.
- Cross-validation refits every pipeline step within each training fold.
- Use `step__parameter` names for grid search.
- `named_steps` exposes fitted transformers and models for interpretation.
- Searches can compare preprocessing strategies and estimator families without leakage.
"""),
    ]


def chapter_7():
    return [
        md(r"""
# Chapter 7: Working with Text Data

## Chapter Summary
Text must be converted into numeric features. This chapter builds bag-of-words and TF-IDF representations, compares stop-word and n-gram choices, inspects learned coefficients, and demonstrates topic modeling with Latent Dirichlet Allocation (LDA).

## Theoretical Deep-Dive

A document-term matrix records token counts. TF-IDF downweights terms common across many documents: $tfidf(t,d)=tf(t,d)\log((N+1)/(df(t)+1))$. Logistic regression then learns which terms push a prediction toward each class.

n-grams preserve short word order at the cost of a larger, sparser vocabulary. Tokenization defines the model's evidence, so casing, punctuation, stemming, and domain-specific stop words are modeling choices rather than neutral cleaning.

LDA treats each document as a mixture of latent topics and each topic as a distribution over words. Topics require human interpretation and stability checks.

**Book coverage:** sentiment analysis, bag-of-words, stop words, TF-IDF, coefficients, n-grams, tokenization, and LDA (Chapter 7, printed pp. 323-355).
"""),
        code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

positive = [
    "a thoughtful and moving film", "excellent acting and a wonderful story", "funny warm and beautifully directed",
    "the performances were brilliant", "an exciting film with a satisfying ending", "smart writing and memorable characters",
    "I loved every minute of this movie", "a charming and uplifting experience", "visually stunning and emotionally rich",
    "the plot was engaging and original", "great pacing with strong performances", "an impressive and enjoyable production",
]
negative = [
    "a dull and predictable film", "terrible acting and a weak story", "boring cold and poorly directed",
    "the performances were painful", "a confusing film with a disappointing ending", "lazy writing and forgettable characters",
    "I regretted watching this movie", "a tedious and frustrating experience", "visually messy and emotionally empty",
    "the plot was repetitive and unoriginal", "awful pacing with flat performances", "a disappointing and unpleasant production",
]
texts = np.array(positive + negative); labels = np.array([1] * len(positive) + [0] * len(negative))
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.33, stratify=labels, random_state=42)
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(X_train)
print("Document-term shape:", counts.shape)
print("Vocabulary sample:", vectorizer.get_feature_names_out()[:15])
"""),
        code("""
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
representations = {
    "counts": CountVectorizer(),
    "counts + stopwords": CountVectorizer(stop_words="english"),
    "tfidf": TfidfVectorizer(stop_words="english"),
    "tfidf + 1-2 grams": TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1),
}
rows = []
for name, transformer in representations.items():
    pipe = make_pipeline(transformer, LogisticRegression(max_iter=2000))
    score = cross_val_score(pipe, texts, labels, cv=cv, scoring="accuracy")
    pipe.fit(X_train, y_train)
    rows.append((name, score.mean(), pipe.score(X_test, y_test), len(pipe.named_steps[transformer.__class__.__name__.lower()].get_feature_names_out())))
display(pd.DataFrame(rows, columns=["representation", "cv_accuracy", "test_accuracy", "vocabulary_size"]).round(3))
"""),
        code("""
sentiment = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=2000))
sentiment.fit(texts, labels)
tfidf = sentiment.named_steps["tfidfvectorizer"]
clf = sentiment.named_steps["logisticregression"]
terms = tfidf.get_feature_names_out()
weights = clf.coef_[0]
top_negative = terms[np.argsort(weights)[:8]]
top_positive = terms[np.argsort(weights)[-8:][::-1]]
print("Strong negative evidence:", top_negative)
print("Strong positive evidence:", top_positive)
for review, probability in zip(["wonderful acting and smart writing", "boring plot and terrible ending"], sentiment.predict_proba(["wonderful acting and smart writing", "boring plot and terrible ending"])[:, 1]):
    print(f"P(positive)={probability:.3f} | {review}")
"""),
        code("""
from sklearn.decomposition import LatentDirichletAllocation

topic_documents = [
    "team won the football match with two goals", "coach prepared the team for the league game", "players trained for the championship match",
    "new software improves computer security", "python code trains a machine learning model", "database servers process cloud applications",
    "fresh vegetables make a healthy meal", "the chef prepared pasta with tomato sauce", "baking bread requires flour and yeast",
    "election voters debated government policy", "parliament approved a new public law", "the minister announced an economic policy",
]
topic_vectorizer = CountVectorizer(stop_words="english", min_df=1)
topic_matrix = topic_vectorizer.fit_transform(topic_documents)
lda = LatentDirichletAllocation(n_components=4, max_iter=40, learning_method="batch", random_state=42)
document_topics = lda.fit_transform(topic_matrix)
feature_names = topic_vectorizer.get_feature_names_out()
for index, topic in enumerate(lda.components_):
    print(f"Topic {index}:", ", ".join(feature_names[np.argsort(topic)[-6:][::-1]]))
print("Dominant topic per document:", document_topics.argmax(axis=1))
"""),
        md("""
## Key Takeaways
- Text models learn from a representation, not raw language.
- Bag-of-words discards order; n-grams recover short local context.
- TF-IDF reduces the influence of globally common terms.
- Coefficient inspection provides a useful model audit for linear text classifiers.
- Topic labels are human interpretations of word distributions, not objective ground truth.
"""),
    ]


def chapter_8():
    return [
        md(r"""
# Chapter 8: Wrapping Up

## Chapter Summary
The final chapter connects modeling to problem formulation and deployment. A successful system needs a measurable goal, representative data, reproducible preprocessing, monitoring, tests, human oversight, and a plan for change after deployment.

## Theoretical Deep-Dive

An offline score is only a proxy for real-world value. Define the prediction target, decision threshold, cost of errors, and feedback loop before optimizing an algorithm. Distribution shift can make yesterday's validation result irrelevant, so production systems monitor inputs, predictions, outcomes, latency, and failures.

scikit-learn's estimator interface enables custom components. Estimators expose `fit`; predictors expose `predict`; transformers expose `transform`. Following this interface makes custom logic compatible with pipelines and cross-validation.

Human-in-the-loop systems route uncertain or high-impact cases to review. Automation should include an abstention policy rather than pretending every prediction is equally reliable.

**Book coverage:** approaching ML problems, humans in the loop, prototype-to-production concerns, testing, custom estimators, and next steps (Chapter 8, printed pp. 357-366).
"""),
        code("""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, stratify=data.target, random_state=42)
production_candidate = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
production_candidate.fit(X_train, y_train)
print(classification_report(y_test, production_candidate.predict(X_test), target_names=data.target_names))
"""),
        code("""
from sklearn.base import BaseEstimator, TransformerMixin

class PercentileClipper(TransformerMixin, BaseEstimator):
    # Clip every feature to bounds learned from the training set.
    def __init__(self, lower=1.0, upper=99.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.lower_bounds_ = np.percentile(X, self.lower, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X), self.lower_bounds_, self.upper_bounds_)

custom_pipeline = make_pipeline(PercentileClipper(), StandardScaler(), LogisticRegression(max_iter=3000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(custom_pipeline, data.data, data.target, cv=cv)
print("Custom pipeline CV scores:", scores.round(3))
print(f"Mean +/- SD: {scores.mean():.3f} +/- {scores.std():.3f}")
"""),
        code("""
probability = production_candidate.predict_proba(X_test)[:, 1]
confidence = np.maximum(probability, 1 - probability)
review_threshold = 0.80
automatic = confidence >= review_threshold

review_table = pd.DataFrame({
    "probability_benign": probability,
    "confidence": confidence,
    "decision": np.where(automatic, np.where(probability >= 0.5, "auto-benign", "auto-malignant"), "human-review"),
    "actual": data.target_names[y_test],
})
display(review_table.head(10).round(3))
print(f"Automatically handled: {automatic.mean():.1%}")
print(f"Sent to human review: {(~automatic).mean():.1%}")
"""),
        code("""
reference_mean = X_train.mean(axis=0)
incoming_mean = X_test.mean(axis=0)
reference_std = X_train.std(axis=0)
standardized_shift = np.abs(incoming_mean - reference_mean) / np.maximum(reference_std, 1e-12)
largest = np.argsort(standardized_shift)[-5:][::-1]
shift_report = pd.DataFrame({
    "feature": data.feature_names[largest],
    "standardized_mean_shift": standardized_shift[largest],
})
display(shift_report.round(3))
print("Illustrative alert rule: investigate features with standardized mean shift > 0.5")
"""),
        md("""
## Key Takeaways
- Define the decision, success metric, and error costs before modeling.
- Package every learned preprocessing step with the model.
- Test data contracts and custom components, not only model accuracy.
- Monitor drift and real outcomes after deployment.
- Route uncertain or consequential predictions to people.
- Reproducibility, documentation, privacy, and maintenance are part of model quality.
"""),
    ]


CHAPTERS = {
    "Chapter_01_Introduction": chapter_1,
    "Chapter_02_Supervised_Learning": chapter_2,
    "Chapter_03_Unsupervised_Learning_and_Preprocessing": chapter_3,
    "Chapter_04_Representing_Data_and_Engineering_Features": chapter_4,
    "Chapter_05_Model_Evaluation_and_Improvement": chapter_5,
    "Chapter_06_Algorithm_Chains_and_Pipelines": chapter_6,
    "Chapter_07_Working_with_Text_Data": chapter_7,
    "Chapter_08_Wrapping_Up": chapter_8,
}


def build_and_execute(folder, cell_factory):
    output_dir = ROOT / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook(
        cells=cell_factory(),
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    target = output_dir / "notebook.ipynb"
    nbf.write(notebook, target)
    print(f"Built {target.relative_to(ROOT)}")


if __name__ == "__main__":
    for chapter_folder, factory in CHAPTERS.items():
        build_and_execute(chapter_folder, factory)
