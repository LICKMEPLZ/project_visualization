import pandas as pd # from python we can use this to help us get data
import matplotlib.pyplot as plt # library that helps us with plotting

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("heart.csv")

x = data.values[:, :data.shape[1] -1]
y = data.values[:, data.shape[1]-1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5, shuffle=True)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

val_acc = []
train_acc = []
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
    knn.fit(X_train_std, y_train)
    val_acc.append(knn.score(X_test_std, y_test))
    train_acc.append(knn.score(X_train_std, y_train))
    print(f"heart disease knn machine learning of distance -> {i} score -> {knn.score(X_test_std, y_test)*100}")

plt.plot(val_acc, "*--", label="test_acc")
plt.plot(train_acc, "^--", label="train_acc")
plt.xlabel("distance")
plt.ylabel("acc")
plt.legend()
plt.show()
