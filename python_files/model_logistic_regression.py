# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Midwest Survey — Logistic Regression
#
# Predict the census region from survey responses using a logistic regression
# pipeline with `skrub.TableVectorizer` for automatic feature encoding.

# %%
import joblib
import skrub
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# %%
bunch = skrub.datasets.fetch_midwest_survey()
X, y = bunch.X, bunch.y
print(f"X shape: {X.shape}, target classes: {y.nunique()}")

# %%
model = make_pipeline(
    skrub.TableVectorizer(),
    LogisticRegression(max_iter=1000),
)
model

# %%
scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# %%
model.fit(X, y)
joblib.dump(model, "model_logistic_regression.pkl")
print("Model saved to model_logistic_regression.pkl")
