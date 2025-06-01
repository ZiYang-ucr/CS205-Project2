import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from Part1 import forward_selection, baseline_accuracy

def preprocess_rice_data(filename):
    df = pd.read_csv(filename)


    #make sure the last column is "Class"
    if df.columns[-1] != "Class":
        class_col = df.pop("Class")
        df.insert(0, "Class", class_col)

    # covert class labels to string and remove unwanted characters
    df["Class"] = df["Class"].astype(str).str.replace("b'", "").str.replace("'", "")

    # map class labels to integers
    label_map = {"Cammeo": 1, "Osmancik": 2}
    df["Class"] = df["Class"].map(label_map)

    if df["Class"].isnull().any():
        raise ValueError("Class column mapping failed. Check for unexpected labels.")

    print("Class label mapping successful.")

    # Extract features and labels
    labels = df["Class"].values
    features = df.drop(columns=["Class"]).astype(np.float32).values

    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, labels

if __name__ == "__main__":
    file_name = "Rice_Cammeo_Osmancik.csv"
    features, labels = preprocess_rice_data(file_name)

    print(f"\nThis dataset has {features.shape[1]} features (not including the class attribute), with {len(labels)} instances.\n")

    baseline_accuracy(features, labels)
    forward_selection(features, labels)