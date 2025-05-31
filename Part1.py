



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




if __name__ == "__main__":
    print("Welcome to Zi Yang's Feature Selection Algorithm.")
    file_name = input("Type in the name of the file to test:").strip()
    #CS205_small_Data__9.txt or CS205_large_Data__26.txt
    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection\n 2) Backward Elimination\n")
    choice = int(input().strip())   

    

