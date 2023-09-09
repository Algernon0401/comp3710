from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import torch

# Get the data
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape

X = lfw_people.data
n_features = X.shape[1]

y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print(lfw_people.images.shape, lfw_people.data.shape)
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# Splite data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# convert to torch tensor
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

n_components = 150

# Center data
mean = torch.mean(X_train, dim=0, keepdim=True) # <TORCH>
X_train -= mean
X_test -= mean

# Eigen-decomposition
U, S, V = torch.linalg.svd(X_train, full_matrices=False) # <TORCH>
components = V[:n_components]

#project into PCA subspace
X_transformed = torch.matmul(X_train,components.T)
X_test_transformed = torch.matmul(X_test,components.T)


from sklearn.ensemble import RandomForestClassifier

# build random forest
estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
estimator.fit(X_transformed, y_train) # expects X as [n_samples, n_features]
predictions = torch.from_numpy(estimator.predict(X_test_transformed))
correct = (predictions==y_test)
total_test = len(X_test_transformed)
# print("Gnd Truth:", y_test)
print("Total Testing", total_test)
# print("Predictions", predictions)
# print("Which Correct:", correct)
print("Total Correct:", torch.sum(correct))
print("Accuracy:", torch.sum(correct)/total_test)
print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))

