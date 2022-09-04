import pandas as pd  # from python we can use this to help us get data
import numpy as np  # a library that help us with vector and matrix with lists
import matplotlib.pyplot as plt  # library that helps us with plotting
import seaborn as sns  # from matplotlib that helps us visualizing data

# setting of data visualization
sns.set(style="white", context="notebook", palette="deep")

data = pd.read_csv("heart.csv")
print("what is data columns of heart disease csv file? --> \n{}".format(data.columns))
# len is used to find the total index number of a list
print("data index columns of heart disease --> {}".format(len(data.columns)))
print("data count --> \n{}".format(data.count()))

disease_count = data["target"].value_counts()
print("heart disease counting -->\n{}".format(disease_count))

# heart disease target order by age
f, ax = plt.subplots(1, 3, figsize=(18, 8))
# ax is order by which it comes out
disease_count.plot.pie(ax=ax[0], shadow=True, explode=[0, 0.1], autopct="%1.1f%%")
ax[0].set_title("heart disease -- Pie plot")
ax[0].set_ylabel("")

sns.countplot("target", data=data, ax=ax[1])
ax[1].set_title("target -- count")
ax[1].set_xlabel("target")
ax[1].set_ylabel("")

disease_target = data[["age", "target"]].groupby(["age"], as_index=True).count()  # used to find the relations between age and target
disease_target.plot(ax=ax[2])
ax[2].set_xlabel("age")
ax[2].set_ylabel("count")
plt.show()

sns.countplot("age", hue="target", data=data)
plt.show()
# -------------------------------------------------------------------------------------

f, ax = plt.subplots(1, 2, figsize=(18, 8))
a = data["cp"].value_counts()
a.plot.pie(ax=ax[0], shadow=True, autopct='%1.1f%%')
sns.countplot("cp", data=data, hue="target", ax=ax[1])
plt.show()

sns.countplot("target", hue="sex", data=data)
plt.xlabel("")
plt.show()

f, ax = plt.subplots(1, 2, figsize=(18, 8))
data[["cp", "target"]].groupby(["cp"], as_index=True).mean().plot.bar(ax=ax[0])
data[["cp", "target"]].groupby(["cp"], as_index=True).count().plot.bar(ax=ax[1])
plt.show()

age_target = data["age"][data["target"] == 1]
age_target_na = data["age"][data["target"] == 0]
max_target = data["thalach"][data["target"] == 1]
max_target_na = data["thalach"][data["target"] == 0]

plt.scatter(x=age_target, y=max_target, c="red")
plt.scatter(x=age_target_na, y=max_target_na)
plt.legend(["Disease", "Not Disease"])
plt.xlabel("age")
plt.ylabel("Heart rate")
plt.show()


age_f = data["age"][data["fbs"] == 1]
sns.jointplot(data=data, x= 'trestbps', y='cp',cmap='PuBu')
plt.show()

age_fbs = data[["age", "fbs"]].groupby("age", as_index=True).count()
age_fbs.plot()
plt.title("age of fbs")
plt.xlabel("age")
plt.ylabel("fbs")
plt.show()