from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
raw_iris = datasets.load_iris()


#피처 , 타깃 데이터 지정
X = raw_iris.data
y = raw_iris.target

#트레이닝 / 테스트 데이터 분할
X_tn, X_te , y_tn , y_te = train_test_split(X,y,random_state= 0)

#데이터 표준화
std_scale = StandardScaler()
std_scale.fit(X_tn)
X_tn_std = std_scale.transform(X_tn)
X_te_std = std_scale.transform(X_te)

#데이터 학습
clf_knn = KNeighborsClassifier(n_neighbors= 2)
clf_knn.fit(X_tn_std, y_tn)

#데이터 예측
knn_pred = clf_knn.predict(X_tn_std)
print(knn_pred)

knn_pred = clf_knn.predict(X_te_std)
print(knn_pred)

# 정확도
accuracy = accuracy_score(y_te, knn_pred)
print(accuracy)

#confusion matrix 확인
conf_matrix = confusion_matrix(y_te, knn_pred)
print(conf_matrix)

#분류 리포트 확인
class_report = classification_report(y_te, knn_pred)
print(class_report)





