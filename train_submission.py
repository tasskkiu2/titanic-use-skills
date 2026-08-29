import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Title"] = (
        result["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    )
    result.loc[
        ~result["Title"].isin(["Mr", "Miss", "Mrs", "Master"]), "Title"
    ] = "Rare"
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    result["IsAlone"] = (result["FamilySize"] == 1).astype(int)
    result["Deck"] = result["Cabin"].str[0].fillna("U")
    return result.drop(
        columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"],
        errors="ignore",
    )


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
X = make_features(train)
y = train["Survived"]
X_test = make_features(test)

numeric = ["Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
categorical = ["Pclass", "Sex", "Embarked", "Title", "Deck"]
preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric,
        ),
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical,
        ),
    ]
)

candidates = {
    "logistic_regression": LogisticRegression(max_iter=1_000, random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=1,
    ),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
evaluations = []
for name, estimator in candidates.items():
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "roc_auc", "f1"],
        n_jobs=1,
    )
    evaluations.append(
        {
            "model": name,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "f1_mean": scores["test_f1"].mean(),
        }
    )

evaluation = pd.DataFrame(evaluations).sort_values("roc_auc_mean", ascending=False)
best_name = evaluation.iloc[0]["model"]
best_pipeline = Pipeline(
    [("preprocess", preprocessor), ("model", candidates[best_name])]
)
best_pipeline.fit(X, y)
predictions = best_pipeline.predict(X_test).astype(int)

submission = pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": predictions}
)
submission.to_csv("submission.csv", index=False)
evaluation.to_csv("model_evaluation.csv", index=False)

print(evaluation.round(4).to_string(index=False))
print(f"selected={best_name}")
print(f"submission_rows={len(submission)} survived_rate={predictions.mean():.3f}")
