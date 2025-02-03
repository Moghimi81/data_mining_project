#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv('modified_diabetes_prediction_dataset.csv')
data.head()


# In[2]:


data = data[data['gender'].isin(['Male', 'Female'])]
data = data[data['age'] >= 0]

data.drop_duplicates(inplace=True)

age_mean = data[data['age'] != 0]['age'].mean()
data['age'].fillna(age_mean, inplace=True)

data['age'] = data['age'].astype(int)

data = data[(data['bmi'] >= 10) & (data['bmi'] <= 60)]

HbA1c_level_mean = data[data['HbA1c_level'] != 0]['HbA1c_level'].mean()
data['HbA1c_level'].fillna(HbA1c_level_mean, inplace=True)

data = data[data['blood_glucose_level'] != 9999]

data.replace({"smoking_history": {"yes": "current"}}, inplace=True)

ordinal_mapping = {'never': 0, 'former': 0.25, 'not current': 0.5, 'ever': 0.75, 'current': 1}

data['smoking_history'] = data['smoking_history'].map(ordinal_mapping)

smoking_history_mode_value = data['smoking_history'].mode()[0]
data['smoking_history'].fillna(smoking_history_mode_value, inplace=True)

data.head()


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})

# ویژگی‌ها و هدف را جدا کنید
X = data[['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = data['diabetes']

# داده‌ها را به مجموعه‌های آموزشی و تست تقسیم کنید
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# مدل SVM را آموزش دهید
svm_model = SVC()
svm_model.fit(X_train, y_train)

# پیش‌بینی‌ها با مدل SVM
svm_predictions = svm_model.predict(X_test)

# مدل KNN را آموزش دهید
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# پیش‌بینی‌ها با مدل KNN
knn_predictions = knn_model.predict(X_test)

# محاسبه معیارهای مورد نظر برای مدل SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions)
svm_recall = recall_score(y_test, svm_predictions)
svm_f1 = f1_score(y_test, svm_predictions)
svm_conf_matrix = confusion_matrix(y_test, svm_predictions)

print('SVM Accuracy:', svm_accuracy)
print('SVM Precision:', svm_precision)
print('SVM Recall:', svm_recall)
print('SVM F1 Score:', svm_f1)
print('SVM Confusion Matrix:\n', svm_conf_matrix)

# محاسبه معیارهای مورد نظر برای مدل KNN
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions)
knn_conf_matrix = confusion_matrix(y_test, knn_predictions)

print('KNN Accuracy:', knn_accuracy)
print('KNN Precision:', knn_precision)
print('KNN Recall:', knn_recall)
print('KNN F1 Score:', knn_f1)
print('KNN Confusion Matrix:\n', knn_conf_matrix)


# In[11]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


sample_data = data.sample(frac=0.1, random_state=42)


# نمایش برخی از داده‌ها
print(sample_data.head())

# داده‌های ویژگی را به صورت استاندارد درمی‌آوریم
features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
X = sample_data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تعیین تعداد خوشه‌ها با استفاده از امتیاز سیلوئت
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# انتخاب بهترین تعداد خوشه‌ها بر اساس امتیاز سیلوئت
best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f'The best number of clusters is: {best_n_clusters}')

# اجرای K-Means با بهترین تعداد خوشه‌ها
kmeans = KMeans(n_clusters=best_n_clusters, init='k-means++', random_state=42)
sample_data['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

# نمایش برخی از داده‌ها با خوشه‌بندی
print(sample_data.head())


# In[5]:


from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import seaborn as sns

# محاسبه ماتریس پیوند برای خوشه‌بندی سلسله‌مراتبی
Z = linkage(X_scaled, method='ward')

# رسم دندروگرام
plt.figure(figsize=(10, 6))
dendrogram(Z, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

best_n_clusters_hierarchical = best_n_clusters
sample_data['Cluster_Hierarchical'] = fcluster(Z, best_n_clusters_hierarchical, criterion='maxclust')


data_mean_hierarchical = sample_data[features + ['Cluster_Hierarchical']].groupby('Cluster_Hierarchical').mean()

# چاپ ویژگی‌های برجسته هر خوشه برای Hierarchical Clustering
print("\nHierarchical Clustering:")
for cluster in range(1, best_n_clusters_hierarchical + 1):
    print(f"\nCluster {cluster}:")
    print(data_mean_hierarchical.loc[cluster].sort_values(ascending=False))
    
# نمایش برخی از داده‌ها با خوشه‌بندی سلسله‌مراتبی
print(sample_data.head())


# In[6]:


# انتخاب ویژگی‌های برجسته برای تحلیل
selected_features = ['age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# تحلیل ویژگی‌های برجسته در هر خوشه (K-Means)
cluster_centers_kmeans = pd.DataFrame(kmeans.cluster_centers_, columns=features)[selected_features]
data_mean_kmeans = sample_data[selected_features + ['Cluster_KMeans']].groupby('Cluster_KMeans').mean()

# تحلیل ویژگی‌های برجسته در هر خوشه (Hierarchical)
data_mean_hierarchical = sample_data[selected_features + ['Cluster_Hierarchical']].groupby('Cluster_Hierarchical').mean()

# چاپ ویژگی‌های برجسته هر خوشه برای K-Means
print("\nK-Means Clustering:")
for cluster in range(best_n_clusters):
    print(f"\nCluster {cluster}:")
    print(data_mean_kmeans.loc[cluster].sort_values(ascending=False))

# چاپ ویژگی‌های برجسته هر خوشه برای Hierarchical Clustering
print("\nHierarchical Clustering:")
for cluster in range(1, best_n_clusters_hierarchical + 1):
    print(f"\nCluster {cluster}:")
    print(data_mean_hierarchical.loc[cluster].sort_values(ascending=False))

