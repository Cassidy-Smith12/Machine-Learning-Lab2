import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


digits = load_digits()
X, y = digits.data, digits.target
print(f'Shape X: {X.shape}')
print(f'Shape y: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
random_state=42) 

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print('Model accuracy score: ', accuracy_score(y_test, pred))

conf_matrix = confusion_matrix(y_test, pred)
print(f'\nConfusion Matrix: \n{conf_matrix}')

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for ax, idx in zip(axes, range(5)):
    ax.imshow(digits.images[idx], cmap='gray')
    ax.set_title(f'Label: {digits.target[idx]}')
    ax.axis('off')
plt.show()
