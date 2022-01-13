# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:00:43 2022

@author: KITCOOP
20220112.py
"""
# 지도학습 : 분류
# KNN 알고리즘
#    데이터정리(전처리) -> 훈련/검증 분리 -> 알고리즘 선택.
# -> 모형학습 -> 예측 -> 평가 -> 활용

#titanic데이터 로드 : df
import seaborn as sns
import pandas as pd
df = sns.load_dataset("titanic")
df.info()
# deck 컬럼 삭제 : 결측값이 너무 많음 :rdf
# embark_town 컬럼 삭제 : embarked 중복의미 
rdf = df.drop(['deck','embark_town'],axis=1)
rdf.info()
#age컬럼이 없는 행을 제거 : rdf
rdf = rdf.dropna(subset=['age'],axis=0)
rdf.info()
#embarked 열의 NaN값을 
#승선도시 중에서 가장 많이 출현한 값으로 치환하기
rdf['embarked'].value_counts(dropna=False)
most_freq = rdf['embarked'].value_counts().idxmax()
most_freq
rdf["embarked"].fillna(most_freq, inplace=True)
rdf['embarked'].value_counts(dropna=False)

#분석에 필요한 열선택 : ndf
ndf = rdf[['survived','pclass','sex','age','sibsp','parch','embarked']]
ndf.info()
'''
   원핫인코딩 : 문자열형 범주데이터를 모형이 인식할 수 있도록 숫자형으로 변환이 필요
   pandas.get_dummies()
'''
#sex 컬럼을 원핫인코딩 하기
oneh_sex = pd.get_dummies(ndf["sex"])
oneh_sex
#ndf 데이터와 oneh_sex 데이터를 합하기
ndf = pd.concat([ndf,oneh_sex], axis=1) 
ndf.info()
#embarked 컬럼을 원핫인코딩 하기
onehot_embarked = pd.get_dummies(ndf["embarked"],prefix="town")
onehot_embarked
#ndf 데이터와 onehot_embarked 데이터를 합하기
ndf = pd.concat([ndf,onehot_embarked], axis=1) 
ndf.info()

#sex,embarked컬럼 제거하기
del ndf["sex"],ndf["embarked"]
ndf.info()

#설명변수,목표변수 결정
# 설명변수 : survived 컬럼 제외한 변수들
# 목표변수 : survived 컬럼
X = ndf[ndf.columns.difference(["survived"])]
X.info()
Y = ndf["survived"]
#훈련데이터,검증데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.3, random_state=10)
X_train.shape
X_test.shape

# KNN 분류 관련 알고리즘을 이용
# KNN :최근접이웃알고리즘

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5) #KNN알고리즘 객체
knn.fit(X_train,y_train) #학습
y_hat = knn.predict(X_test) #예측


#모형의 성능 평가하기
#혼동행렬
from sklearn import metrics 
knn_matrix = metrics.confusion_matrix(y_test, y_hat)  
print(knn_matrix)
'''
혼동행렬 : 분류 결과 데이터 

실제/예측  0    1  
  0     [111  14]
  1     [ 29  61]

0:F, 1:T

 TN =  111 : 실제 : F, 예측 : F
 FP =  14  : 실제 : F, 예측 : T 
 FN =  29  : 실제 : T, 예측 : F
 TP =  61  : 실제 : T, 예측 : T
 
 정답정확하게 예측 : TN,TP
 정답틀리게 예측 :   FN,FP

'''
#모형 성능 평가 : 평가지표 
knn_report = metrics.classification_report(y_test, y_hat)            
print(knn_report)
y_test.value_counts()

'''
support : 실제 데이터 갯수
   accuracy(정확도) : 정답율
         정확한예측 /전체데이터 => (TN+TP)/(TN+FN+TP+FP)
                                 (172/215) = 0.8
  precision(정밀도) : True로 예측한것 중 실제 True인 비율
         실제 True/True 예측 => TP/TP+FP
                               61/(61+14) = 0.8133333333333333
  recall(재현율,민감도) : 실제 True인 경우 True로 예측한 비율 
           예측True /실제 True => TP/TP+FN 
                               61/(61+29) = 0.677777777
  f1-score(조화평균) : 2 * (정밀도 * 재현율) / (정밀도+재현율)        
  macro avg : 평균의 평균
 weighted avg : 가중 평균        
'''
### SVM 분류 알고리즘으로 모델 구현하기
# SVM : Support Vector Machine : 공간을 (선/면)으로 분류하는 방식
from sklearn import svm
#kernel='rbf' : 공간을 분리하는데 사용되는 방식 결정하는 함수 지정
#             rbf(기본값), linear,poly
svm_model = svm.SVC(kernel='rbf')
svm_model.fit(X_train, y_train) #학습하기 
y_hat = svm_model.predict(X_test) #검증하기 
y_test.values[:10]
y_hat[:10]
#혼동행렬, 평가값을 출력하기 
svm_matrix = metrics.confusion_matrix(y_test, y_hat)  
svm_matrix
svm_report = metrics.classification_report(y_test, y_hat)            
print(svm_report)
# 함수들을이용하여 정확도,정밀도,재현율,F1-Score 값 출력하기
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
print("정확도(accuracy): %.2f" % accuracy_score(y_test, y_hat))
print("정밀도(Precision) : %.3f" % precision_score(y_test, y_hat))
print("재현율(Recall) : %.3f" % recall_score(y_test, y_hat))
print("F1-score : %.3f" % f1_score(y_test, y_hat))

####
# Decision Tree (의사결정나무)
# UCI 데이터 : 암세포 진단 데이터 
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np

uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/\
breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df = pd.read_csv(uci_path,header=None)
df.columns=['id','clump','cell_size','cell_shape','adhesion',\
            'epithlial','bare_nuclei','chromatin','normal_nucleoli',\
            'mitoses','class']
df.info()    
df.head()  
'''
컬럼설명
1.id : ID번호 
2.clump : 덩어리 두께
3.cell_size : 암세포 크기
4.cell_shape:세포모양
5.adhesion : 한계
6.epithlial: 상피세포 크기
7.bare_nuclei : 베어핵
8.chromatin : 염색질 
9.normal_nucleoli : 정상세포
10.mitoses : 유사분열
11.class : 2 (양성), 4(악성)
'''    
df['class'].value_counts()
df['bare_nuclei'].value_counts()
df.info()
# bare_nuclei 데이터가 ? 데이터를 조회하기.
df.loc[df["bare_nuclei"]=='?']
df.loc[df["bare_nuclei"]=='?',['id',"bare_nuclei",'class']]
# bare_nuclei 데이터에서 ? 데이터를 포함한 행을 삭제하고,
# 자료형을 정수형으로 변경하기
df['bare_nuclei'].replace('?', np.nan, inplace=True) #?결측값변경
df.dropna(subset=['bare_nuclei'], axis=0, inplace=True)#결측값제거
df['bare_nuclei'] = df['bare_nuclei'].astype('int') #자료형변경
df.info()
#설명(독립,속성)변수 : id,class 컬럼을 제외한 컬럼
X = df[df.columns.difference(['id','class'])]
X.info()
#목표(종속,예측)변수 : class 컬럼
Y = df["class"] 

#설명변수 정규화하기.
X = preprocessing.StandardScaler().fit(X).transform(X)
type(X)
X.shape
X[:10]

#훈련/검증 분리
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=10)
X_train.shape
X_test.shape

#Decision Tree 알고리즘
# entropy : 불순도를 수치화시키는 데이터 
#criterion='entropy' : 불순도 계산하기위한 함수 지정.
# max_depth=5 : 트리의 깊이 지정. 
# 트리의 깊이가 너무 긴경우는 과대적합이 생길수 있다. 
# 과대적합 : 학습이 지나침. 단점 : 검증 점수가 낮다.
tree_model = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5) #알고리즘
tree_model.fit(X_train,Y_train) # 지도학습
y_hat = tree_model.predict(X_test)
y_hat[:10]
Y_test.values[:10]
tree_matrix=metrics.confusion_matrix(Y_test, y_hat)
tree_matrix
tree_report = metrics.classification_report(Y_test,y_hat)
tree_report

'''
   지도학습 : 정답 제공. 정답 존재.
     회귀분석 : 
           단순회귀분석 : 독립,종속변수가 1개씩. 회귀선 1차원. 직선형태
           다항회귀분석 : 회귀선 다차원. 곡선형태
           다중회귀분석 : 독립변수가 여러개존재,종속변수가 1개.
       분류 : 결과를 정확하게 예측. 
           KNN(최근접이웃알고리즘)
           SVM(Support Vector Machine)
           Decision Tree(의사결정나무)
   비지도학습 : 정답 제공 없음.
       군집 : 데이터의 유사성으로 구분         
           K-means : 임의의 중심점에서 가까운 점들을 하나의 그룹
'''
# 군집 
import pandas as pd
import matplotlib.pyplot as plt
#고객의 연간 구매금액을 상품 종류별로 구분한 데이터
uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/\
00292/Wholesale%20customers%20data.csv'
df = pd.read_csv(uci_path, header=0)
df.info()
df.head()
X = df.iloc[:,:] #모든 데이터 

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X) #정규화
X[:10]
#Kmeans 알고리즘
from sklearn import cluster
# init="k-means++" : 중심점 랜덤하게 설정 
# n_init=10 : 10개로 중심점 시작.
# n_clusters=5 : 결과를 5개의 그룹 분리. 클러스터=그룹. 
#                클러스터의 갯수 설정 
kmeans = cluster.KMeans(init="k-means++", n_clusters=5,n_init=10) #kmeans 알고리즘
kmeans.fit(X) #학습.
cluster_label = kmeans.labels_
len(cluster_label) #각 레코들의 클러스터 값
len(X)
#df 데이터에 클러스터 컬럼 추가하기
df['cluster'] = cluster_label
#cluster 컬럼의 값 조회하기
df['cluster'].unique() #5가지 값. 5가지 종류로 구분함. 
#cluster 별 평균조회하기
df.groupby('cluster').mean()
#산점도 출려하기
# Grocery, Frozen 의 산점도 출력하기
# cluster컬럼을 색상으로 표현
df.plot(kind='scatter',x='Grocery',y='Frozen',c='cluster',cmap='Set1',\
        colorbar=True,figsize=(10,10))

# Milk,Delicassen 의 산점도 출력하기
# cluster컬럼을 색상으로 표현
df.plot(kind='scatter',x='Milk',y='Delicassen',c='cluster',cmap='Set1',\
        colorbar=True,figsize=(10,10))

# 군집 : DBSCAN 알고리즘. => 공간의 밀집도로 클러스터 구분 
import pandas as pd
import folium
file_path = 'data/2016_middle_shcool_graduates_report.xlsx'
df = pd.read_excel(file_path,  header=0)
df.info()
# df데이터에서 각 중학교의 정보를 지도로 표시하기
mschool_map = folium.Map(location=[37.55,126.98], zoom_start=12)
for name,lat,lng in zip(df.학교명,df.위도,df.경도) :
    folium.CircleMarker([lat,lng],
                        radius=5,color='brown', fill=True,fill_color='coral',
                  fill_opacity=0.7, popup=name,tooltip=name).add_to(mschool_map)
mschool_map.save('seoul_mscshool.location.html')    
df.info()
df.지역.unique()
df.코드.unique()
df.유형.unique()
# 원핫인코딩 : 문자열 -숫자형. 컬럼 구분.
#            preprocessing.OneHotEncoder()
# label인코더 : 크기정보가 의미 없이, 단순한 종류인 경우.
#               문자/숫자 -> 숫자형
#            preprocessing.LabelEncoder()
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
onehot_encoder=preprocessing.OneHotEncoder()
label_code = label_encoder.fit_transform(df["코드"])
label_code
df["코드"].values[:10]
label_code[:10]
df["코드"].unique()
label_loc = label_encoder.fit_transform(df["지역"])
label_loc
label_type = label_encoder.fit_transform(df["유형"])
label_type
label_day = label_encoder.fit_transform(df["주야"])
label_day
#label_encoder 된 데이터를 df의 컬럼으로 추가
df["code"]=label_code
df["location"]=label_loc
df["type"]=label_type
df["day"]=label_day
df.info()
df["code"].unique()
df["location"].unique()
df["type"].unique()
df["day"].unique()
df.info()
#속성변수 설정. 과학고,외고국제고,자사고 진학률로 분리하기
X=df.iloc[:,[9,10,13]]
X.info()
#X 데이터 정규화
X = preprocessing.StandardScaler().fit(X).transform(X)
X[:5]
# DBSCAN 알고리즘
# eps=0.2 : 반지름 크기. 
# min_samples=5 : 최소점의 갯수.
#                 eps 영역내에 최소 5개의 점이 있으면 클러스터로 인정
# cluster : 그룹
# core point : 그룹화를 위한 중심점.
# noise point : 그룹화 하지 못한 데이터. -1 설정

dbm = cluster.DBSCAN(eps=0.2,min_samples=5)
dbm.fit(X)
cluster_label = dbm.labels_
df["cluster"] = cluster_label
df["cluster"].unique()
df["cluster"].value_counts()

#cluster로 그룹화 하여 레코드 조회하기
for key,group in df.groupby("cluster") :
    print("* cluster:", key)
    print("* number :",len(group))
    print(group.iloc[:,[0,1,3,9,10,13]].head())
    print("\n")





