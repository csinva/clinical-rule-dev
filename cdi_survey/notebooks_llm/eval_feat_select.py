from sklearn.model_selection import train_test_split

# import sklearn tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from imodels import HSTreeClassifierCV
import numpy as np
from sklearn.model_selection import cross_val_score
import warnings
from collections import defaultdict
from sklearn import metrics as skm
from sklearn.model_selection import train_test_split, StratifiedKFold


def evaluate_features(
    X, y, seed=42, class_weight="balanced", return_pr_curve=False, n_splits=3
):
    if class_weight == "balanced":
        class_weight = y.size / y.sum()
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size, random_state=seed, stratify=y
    # )
    # print(y_train.mean(), y_test.mean(), y_test.sum())

    # compute cross validation scores
    met_scores = defaultdict(list)
    # for scoring in ["roc_auc", "accuracy", "f1", "precision", "recall"]:
    # with warnings.catch_warnings():
    # warnings.filterwarnings("ignore", message=".*Precision is ill-defined*")

    # generate stratified kfolds split

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # initialize model
        # m = DecisionTreeClassifier(
        # random_state=seed, class_weight={0: 1, 1: class_weight}
        # )
        m = LogisticRegression(random_state=seed, class_weight={0: 1, 1: class_weight})
        # m = LogisticRegressionCV(
        # random_state=seed, class_weight={0: 1, 1: class_weight}
        # )

        # fit model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*lbfgs failed to converge*")
            m.fit(X_train, y_train)

        # predict
        y_pred_proba = m.predict_proba(X_test)[:, 1]
        y_pred = m.predict(X_test)
        # y_pred = (
        # y_pred_proba > y_pred_proba.min()
        # )  # threshold at non-minimum probability

        # calculate metrics
        met_scores["accuracy"].append(skm.accuracy_score(y_test, y_pred))
        met_scores["f1"].append(skm.f1_score(y_test, y_pred, zero_division=0))
        met_scores["precision"].append(
            skm.precision_score(y_test, y_pred, zero_division=0)
        )
        met_scores["recall"].append(skm.recall_score(y_test, y_pred, zero_division=0))
        met_scores["roc_auc"].append(skm.roc_auc_score(y_test, y_pred_proba))

    met_scores = {
        k: np.mean(v) for k, v in met_scores.items() if not k.endswith("_curve")
    }
    # met_scores["roc_auc_curve"] = met_scores["roc_auc_curve"][0]
    if return_pr_curve:
        met_scores["roc_auc_curve"] = skm.precision_recall_curve(y_test, y_pred_proba)

    # met_scores[scoring] = np.mean(
    # cross_val_score(m, X, y, cv=5, scoring=scoring)
    # )
    # print(f"{scoring}: {scores.mean():.3f} +/- {scores.std():.3f}")

    # m.fit(X_train, y_train)

    # y_pred = m.predict(X_test)
    # y_pred_proba = m.predict_proba(X_test)[:, 1]

    # # calculate metrics
    # met_scores = {
    #     "accuracy": accuracy_score(y_test, y_pred),
    #     "f1": f1_score(y_test, y_pred),
    #     "precision": precision_score(y_test, y_pred),
    #     "recall": recall_score(y_test, y_pred),
    #     "roc_auc": roc_auc_score(y_test, y_pred_proba),
    # }
    return met_scores
