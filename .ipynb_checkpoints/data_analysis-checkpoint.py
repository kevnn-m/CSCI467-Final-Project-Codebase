import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def check_class_imbalance(y):
    class_counts = y.value_counts()
    total_count = len(y)
    print("Class Distribution:")
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count} samples ({count / total_count * 100:.2f}%)")

def train_and_evaluate(df, feature_cols=None, train_size=0.7, val_size=0.1, test_size=0.2):
    assert train_size + val_size + test_size == 1.0, "Split ratios must add up to 1.0"

    if feature_cols is None:
        feature_cols = ['SIZE', 'FUEL', 'DISTANCE', 'DESIBEL', 'AIRFLOW', 'FREQUENCY']

    X = df[feature_cols]
    y = df['STATUS']

    check_class_imbalance(y)

    if 'FUEL' in feature_cols:
        label_encoder = LabelEncoder()
        X['FUEL'] = label_encoder.fit_transform(X['FUEL'])

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_conf_matrix = confusion_matrix(y_val, y_val_pred)
    print("Validation Set Evaluation:")
    print(f"Accuracy: {val_accuracy}")
    print("Confusion Matrix:")
    print(val_conf_matrix)
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred))

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred)
    print("\nTest Set Evaluation:")
    print(f"Accuracy: {test_accuracy}")
    print("Confusion Matrix:")
    print(test_conf_matrix)
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))

    return model, {"val_accuracy": val_accuracy, "test_accuracy": test_accuracy}

df = pd.read_excel("Acoustic_Extinguisher_Fire_Dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx")
model, metrics = train_and_evaluate(df)