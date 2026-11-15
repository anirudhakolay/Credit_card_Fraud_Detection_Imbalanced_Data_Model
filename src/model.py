"""
Model Training Module
Supports various ML models configured for imbalanced data.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_logistic_regression(balanced: bool = True, random_state: int = 42):
    """
    Returns a Logistic Regression pipeline with StandardScaler.
    If balanced=True, uses cost-sensitive weighting to penalize fraud misclassifications.
    """
    class_weight = "balanced" if balanced else None
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state
        ))
    ])


def get_decision_tree(balanced: bool = True, max_depth: int = 8, random_state: int = 42):
    """
    Returns a Decision Tree Classifier.
    """
    class_weight = "balanced" if balanced else None
    return DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state
    )


def get_random_forest(n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
    """
    Returns a Random Forest Classifier with balanced subsample weighting.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state
    )


def get_hist_gradient_boosting(max_iter: int = 100, random_state: int = 42):
    """
    Returns a HistGradientBoostingClassifier (fast, gradient boosted trees)
    with balanced class weighting.
    """
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=random_state
    )
