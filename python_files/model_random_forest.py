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
# # Midwest Survey — Random Forest
#
# Predict the census region from survey responses using a
# `RandomForestClassifier` pipeline with `skrub.TableVectorizer` and a
# numerical-stability preprocessing step.

# %%
import joblib
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# %%
bunch = skrub.datasets.fetch_midwest_survey()
X, y = bunch.X, bunch.y
print(f"X shape: {X.shape}, target classes: {y.nunique()}")

# %%
from midwest_survey_models.transformers import NumericalStabilizer



# %%
model = make_pipeline(
    skrub.TableVectorizer(),
    NumericalStabilizer(),
    RandomForestClassifier(n_estimators=200, random_state=42),
)
model

# %%
scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# %%
model.fit(X, y)
joblib.dump(model, "model_random_forest.pkl")
print("Model saved to model_random_forest.pkl")
