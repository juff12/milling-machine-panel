import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

class_labels = {
    "0": "Heat Dissipation Failure",
    "1": "No Failure",
    "2": "Overstrain Failure",
    "3": "Power Failure",
    "4": "Random Failures",
    "5": "Tool Wear Failure"
}

def run_models(csv_path="predictive_maintenance_balanced.csv"):
    df = pd.read_csv(csv_path)
    df_temp = df.copy()

    for col in ["UDI", "Product ID"]:
        if col in df_temp.columns:
            df_temp = df_temp.drop(col, axis=1)

    X = df_temp.drop("Failure Type", axis=1)
    y = df_temp["Failure Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7, stratify=y
    )

    one_hot_encoder = OneHotEncoder()
    transformer = ColumnTransformer(
        [("one_hot_encoder", one_hot_encoder, ["Product Type", "Target"])],
        remainder="passthrough",
    )
    transformer.fit(X_train)
    encoded_X_train = transformer.transform(X_train)
    encoded_X_test = transformer.transform(X_test)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    encoded_y_train = label_encoder.transform(y_train)
    encoded_y_test = label_encoder.transform(y_test)

    classification_reports = {}

    models = {
        "Random Forest": RandomForestClassifier(random_state=7),
        "Logistic Regression": LogisticRegression(random_state=7, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=7),
        "Multi-Layer Perceptron": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=7),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(random_state=7),
    }

    for model_name, model in models.items():
        model.fit(encoded_X_train, encoded_y_train)
        predictions = model.predict(encoded_X_test)
        cr = classification_report(encoded_y_test, predictions, output_dict=True)

        formatted_cr = {class_labels[k]: v for k, v in cr.items() if k in class_labels}
        formatted_cr["accuracy"] = cr["accuracy"]
        classification_reports[model_name] = formatted_cr

    return classification_reports


def get_classification_report_heatmap(cr, model_name=""):
    colors = ["#845EC2", "#D65DB1", "#FF6F91", "#FF9671",
              "#FFC75F", "#008F7A", "#F9F871"]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f"{model_name} Classification Report Heat Map")

    df_cr = pd.DataFrame(cr).transpose()

    if "support" in df_cr.columns:
        df_cr = df_cr.drop("support", axis=1)

    df_cr.index = [class_labels.get(str(idx), idx) for idx in df_cr.index]

    sns.heatmap(df_cr, annot=True, cmap=colors, vmin=0, vmax=1, ax=ax)
    fig.tight_layout()
    return fig
