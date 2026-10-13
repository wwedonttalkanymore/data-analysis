import numpy as np

try:
    from sklearn.datasets import load_boston
    DATASET_NAME = "Boston"
    X, y = load_boston(return_X_y=True)
except Exception:
    from sklearn.datasets import load_diabetes
    DATASET_NAME = "Diabetes"
    X, y = load_diabetes(return_X_y=True)

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score


def compare_regularized_models():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Ridge": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge())
            ]),
            {"model__alpha": np.logspace(-3, 3, 10)}
        ),
        "Lasso": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(max_iter=10000, random_state=42))
            ]),
            {"model__alpha": np.logspace(-4, 1, 10)}
        ),
        "ElasticNet": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", ElasticNet(max_iter=10000, random_state=42))
            ]),
            {
                "model__alpha": np.logspace(-4, 1, 10),
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
        ),
    }

    results = []

    for name, (pipe, params) in models.items():
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=params,
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        model = best_model.named_steps["model"]
        coef = model.coef_
        non_zero_count = np.count_nonzero(np.abs(coef) > 1e-8)

        results.append({
            "name": name,
            "best_params": grid.best_params_,
            "best_cv_rmse": -grid.best_score_,
            "test_rmse": rmse,
            "test_r2": r2,
            "non_zero_count": non_zero_count,
        })

        print(f"{name:10s} | best params: {grid.best_params_}")
        print(f"{name:10s} | CV RMSE: { -grid.best_score_:.4f}")
        print(f"{name:10s} | Test RMSE: {rmse:.4f}, R2: {r2:.4f}")
        print(f"{name:10s} | non-zero coefficient count: {non_zero_count}")
        print("-" * 60)

    # Choose the model with smallest test RMSE
    best_model = min(results, key=lambda x: x["test_rmse"])
    print(f"\nConclusion: on {DATASET_NAME}, {best_model['name']} has the lowest test RMSE and performs best.")

    if best_model["name"] == "Lasso":
        print("Explanation: Lasso penalizes coefficients so that some become zero, producing a sparser model and supporting feature selection.")
    elif best_model["name"] == "Ridge":
        print("Explanation: Ridge is more stable for collinear features and is useful when many features are highly correlated.")
    else:
        print("Explanation: Elastic Net balances the stability of Ridge and the sparsity of Lasso, often performing well when features are highly correlated.")

    print("\nModel comparison results:")
    for item in results:
        print(
            f"{item['name']:10s} | CV RMSE={item['best_cv_rmse']:.4f} | "
            f"Test RMSE={item['test_rmse']:.4f} | "
            f"R2={item['test_r2']:.4f} | non-zero coefficients={item['non_zero_count']}"
        )


if __name__ == "__main__":
    print(f"Dataset: {DATASET_NAME}")
    print("Compare Ridge, Lasso, and Elastic Net in a regression task\n")
    compare_regularized_models()
