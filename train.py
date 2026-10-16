from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).parent / "data" / "penguins.csv"
FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
TARGET = "species"


def build_model() -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def main() -> None:
    data = pd.read_csv(DATA_PATH).dropna(subset=[TARGET])
    features = data[FEATURES]
    labels = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"Dataset rows: {len(data)}")
    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.1%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()
