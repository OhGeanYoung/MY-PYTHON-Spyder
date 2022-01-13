# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:52:00 2022

@author: KITCOOP
test0112-풀이.py
"""
'''
1. https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
   다운받은 데이터를 이용하여 KNN 알고리즘으로 데이터를 분석하여
   Confusion Matrix 와 classification_report를 출력하기
'''
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split

uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/\
breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df = pd.read_csv(uci_path, header=None)

df.columns = ['id', 'clump', 'cell_size', 'cell_shape', 'adhesion',\
   'epithlial','bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses',\
       'class']
df['bare_nuclei'].replace('?', np.nan, inplace=True)
df.dropna(subset=['bare_nuclei'], axis=0, inplace=True)
df['bare_nuclei'] = df['bare_nuclei'].astype('int')
# 속성(설명)변수
X = df[['clump', 'cell_size', 'cell_shape', 'adhesion',
   'epithlial','bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses']]
# 예측변수
Y = df['class']
#속성변수를 정규화
X = preprocessing.StandardScaler().fit(X).transform(X)
# 데이터 셋 분리.
X_train, X_test, y_train, y_test = train_test_split(
                  X, Y, test_size=0.3, random_state=10)
print("훈련데이터 갯수:",X_train.shape)
print("검증데이터 갯수:",X_test.shape)

from sklearn.neighbors import KNeighborsClassifier
# n_neighbors=5 : k개의 최근접 이웃.
#                 최근접 데이터를 5개 선택.  
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)   
y_hat = knn.predict(X_test)
print(y_hat[0:10])
print(y_test.values[0:10])

#모형의 성능평가하기
from sklearn import metrics 
knn_matrix = metrics.confusion_matrix(y_test, y_hat)  
print(knn_matrix)
knn_report = metrics.classification_report(y_test, y_hat)            
print(knn_report)



'''
2. 1번의 breast-cancer-wisconsin.data 데이터를, 
   SVM알고리즘으로 데이터를 분석하여, 
   Confusion Matrix 와 classification_report를 출력하기
'''
from sklearn import svm
svm_model = svm.SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
y_hat = svm_model.predict(X_test)
# 모형 성능 평가 - Confusion Matrix 계산
from sklearn import metrics 
svm_matrix = metrics.confusion_matrix(y_test, y_hat)  
print(svm_matrix)
# 모형 성능 평가 - 평가지표 계산
svm_report = metrics.classification_report(y_test, y_hat)            
print(svm_report)

'''
 3. 2016_middle_shcool_graduates_report.xlsx를 이용하여 
    DBSCAN 알고리즘을 이용하여 [과학고  외고_국제고    자사고] 컬럼으로
    군집화하고 지도에 색상으로 표시하기
'''
import pandas as pd
import folium
from sklearn import cluster
file_path = 'data/2016_middle_shcool_graduates_report.xlsx'
df = pd.read_excel(file_path,  header=0)
df.info()
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
label_code = label_encoder.fit_transform(df["코드"])
label_loc = label_encoder.fit_transform(df["지역"])
label_type = label_encoder.fit_transform(df["유형"])
label_type
label_day = label_encoder.fit_transform(df["주야"])
label_day
#label_encoder 된 데이터를 df의 컬럼으로 추가
df["code"]=label_code
df["location"]=label_loc
df["type"]=label_type
df["day"]=label_day
X=df.iloc[:,[9,10,13]]
X = preprocessing.StandardScaler().fit(X).transform(X)
dbm = cluster.DBSCAN(eps=0.2,min_samples=5)
dbm.fit(X)
cluster_label = dbm.labels_
df["cluster"] = cluster_label

colors = {-1:'gray', 0:'coral', 1:'blue', 2:'green', 
           3:'red',  4:'purple',5:'orange', 6:'brown' }

cluster_map=folium.Map(location=[37.55,126.98],zoom_start=12)

for name, lat, lng, clus in \
                zip(df.학교명, df.위도, df.경도, df.cluster):  
    folium.CircleMarker([lat, lng],
                        radius=5,            
                        color=colors[clus],  
                        fill=True,
                        fill_color=colors[clus],
                        fill_opacity=0.7,  
                        popup=name,
                        tooltip=name
    ).add_to(cluster_map)
cluster_map.save('./seoul_mschool_cluster2.html')

