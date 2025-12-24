import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="house_prices_preprocessing")
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    X_train = pd.read_csv(data_dir / "X_train_processed.csv")
    X_valid = pd.read_csv(data_dir / "X_valid_processed.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
    y_valid = pd.read_csv(data_dir / "y_valid.csv").values.ravel()

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))
    r2 = float(r2_score(y_valid, y_pred))

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"rmse": rmse, "r2": r2}, f, indent=2)

    mlflow.log_params({"n_estimators": args.n_estimators, "random_state": args.random_state})
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.head(5))

    print("RMSE:", rmse)
    print("R2:", r2)
    print("Saved artifacts ->", out_dir.resolve())


if __name__ == "__main__":
    main()
