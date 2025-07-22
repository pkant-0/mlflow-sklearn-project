import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main(data_path, penalty, max_iter):
    mlflow.start_run()
    df = pd.read_csv(data_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(penalty=penalty, max_iter=max_iter)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("penalty", penalty)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--penalty", type=str)
    parser.add_argument("--max_iter", type=int)
    args = parser.parse_args()

    main(args.data_path, args.penalty, args.max_iter)
