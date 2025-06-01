

import numpy as np


def nearest_neighbor(features,labels, feature_indices):
    correct = 0
    n = len(features)
    for i in range(n):
      features_train = np.delete(features, i, axis=0)[:, feature_indices]
      labels_train = np.delete(labels, i)
      features_test = features[i,feature_indices]
      labels_test = labels[i]
      distances = np.linalg.norm(features_train - features_test, axis=1)
      nearest_index = np.argmin(distances)
      if labels_train[nearest_index] == labels_test:
          correct += 1
    return correct / n

def load_data(file_name):
    data = np.loadtxt(file_name)
    features = data[:, :-1]
    labels = data[:, -1]
    return features, labels


def forward_selection(features, labels):
    num_features = features.shape[1]
    current_features = []
    best_overall_accuracy = 0
    best_features = []
    history = []

    for _ in range(num_features):
        best_accuracy = 0
        best_features = None
        for feature in range(num_features):
            if feature not in current_features:
                new_features = current_features + [feature]
                accuracy = nearest_neighbor(features, labels, new_features)
                history.append((new_features, accuracy))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_features = new_features
        if best_features is not None:
            current_features.append(best_features)  
            if best_accuracy > best_overall_accuracy:
                best_overall_accuracy = best_accuracy
                best_overall_features = current_features.copy()
        else:
            break
    for features, acc in history:
        print(f"Using feature(s) {[i+1 for i in features]} accuracy is {acc * 100:.1f}%")
    print(f"\nFinished search!! The best feature subset is {[i+1 for i in best_overall_features]}, which has an accuracy of {best_overall_accuracy * 100:.1f}%.")

def backward_elimination(features, labels):
    current_features = list(range(features.shape[1]))
    best_overall_accuracy = nearest_neighbor(features, labels, current_features)
    best_overall_features = current_features.copy()
    history = [(current_features.copy(), best_overall_accuracy)]

    while len(current_features) > 1:
        best_accuracy = 0
        feature_to_remove = None
        for feature in current_features:
            new_features = [f for f in current_features if f != feature]
            accuracy = nearest_neighbor(features, labels, new_features)
            history.append((new_features, accuracy))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                feature_to_remove = feature
        if best_accuracy > best_overall_accuracy:
            current_features.remove(feature_to_remove)
            best_overall_accuracy = best_accuracy
            best_overall_features = current_features.copy()
        else:
            break

    for features, acc in history:
        print(f"Using feature(s) {[i+1 for i in features]} accuracy is {acc * 100:.1f}%")
    print(f"\nFinished search!! The best feature subset is {[i+1 for i in best_overall_features]}, which has an accuracy of {best_overall_accuracy * 100:.1f}%.")


def baseline_accuracy(features, labels):
    all_features = list(range(features.shape[1]))
    acc = nearest_neighbor(features, labels, all_features)
    print(f"Running nearest nieghhbor with all{len(all_features)} features,using leaving-one-out evaluation, I get an accuracy of {acc * 100:.2f}%")


if __name__ == "__main__":
    print("Welcome to Zi Yang's Feature Selection Algorithm.")
    file_name = input("Type in the name of the file to test:").strip()
    #CS205_small_Data__9.txt or CS205_large_Data__26.txt
    features, labels = load_data(file_name)
    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection\n 2) Backward Elimination\n")
    choice = int(input().strip())   
    if choice == 1:
        baseline_accuracy(features, labels)
        forward_selection(features, labels)
    elif choice == 2:
        baseline_accuracy(features, labels)
        backward_elimination(features, labels)
    else:
        print("Invalid choice. Please run the program again.")

    


