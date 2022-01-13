# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 08:57:31 2022

@author: KITCOOP
20220111.py
"""
###################
# 머신러닝 : 기계학습. 예측.
#           변수(컬럼,피처)들의 관계를 찾아가는 과정.
#    지도학습   : 기계 학습시 정답을 지정. 
#              회귀분석 : 가격, 매출,주가 등등의 연속성 있는 데이터의 예측에 사용
#              분류 : 데이터 선택. 
#    비지도학습 : 기계 학습시 정답을 주지 않음. 비슷한 데이터들끼리 그룹화
#              군집 : 데이터그룹화
#    강화학습 
#
# 머신러닝 프로세스
#     데이터정리 -> 데이터분리(훈련데이터/검증데이터) -> 알고리즘준비
#   ->모형학습(훈련데이터) -> 예측(검증데이터) -> 모형평가 -> 모형활용
##########################
#   회귀분석(regression) 
#     단순회귀분석 : 독립변수, 종속변수가 각각 한개씩.
#     
#          독립변수(설명변수) : 예측에 사용되는 변수 CCTV 갯수.
#          종속변수(예측변수) : 예측해야 하는 데이터 
#      단항회귀분석 : 독립변수1, 종속변수1
#      다항회귀분석 : 독립변수여러개, 종속변수1
#####################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/auto-mpg.csv",header=None)
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name'] 
df.head()
df.info()
#horsepower 컬럼을 숫자형으로 변경하기
df.horsepower.unique()
#1. ?를 결측값으로 변경
df['horsepower'].replace('?', np.nan, inplace=True)
df['horsepower'].isnull().sum() #결측값의 갯수 
#2. horsepower 컬럼의 결측값을 가진 행을 제거.
df.dropna(subset=["horsepower"],axis=0,inplace=True)
#3. horsepower 컬럼을 숫자형으로 변경 
df['horsepower'] = df['horsepower'].astype('float')
df.info()

# 머신러닝에 필요한 속성(열,컬럼,변수)을 선택하기
ndf = df[['mpg','cylinders','horsepower','weight']]
ndf.info()
ndf[['weight','mpg']].corr()
# mpg,weight변수의 산점도 출력하기
#1. matplot 모듈이용
plt.figure()
ndf.plot(kind='scatter',x='weight',y='mpg',c='coral',s=10,figsize=(10,5))
plt.show()
#2. seaborn.regplot 모듈이용
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
sns.regplot(x='weight',y='mpg',data=ndf,ax=ax1)
sns.regplot(x='weight',y='mpg',data=ndf,ax=ax2,fit_reg=False)
plt.show()
#3. seaborn.jointplot 모듈이용
sns.jointplot(x='weight',y='mpg',data=ndf)
sns.jointplot(x='weight',y='mpg',data=ndf,kind='reg')
plt.show()
#4. seaborn.pairplot 모듈이용
sns.pairplot(ndf[['weight','mpg']],kind='reg')
plt.show()

# 단순회귀분석 : 단항 회귀분석
#독립변수, 종속변수
X = ndf[['weight']]  #독립변수
Y = ndf['mpg']     #종속변수 
X
Y

#데이터분리(훈련데이터/검증데이터)
from sklearn.model_selection import train_test_split
'''
train_test_split : 훈련/검증데이터로 분리. 임의의순서 데이터분리.
train_test_split(독립변수,종속변수,[검증데이터의비율(0.25),seed값])
X_train : 훈련데이터. 독립변수
X_test :  검증데이터. 독립변수
Y_train : 훈련데이터. 종속변수
Y_test :  검증데이터. 종속변수
'''
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=10)
len(X)   #392
len(Y)   #392
len(X_train) #274. 훈련데이터 
len(X_test)  #118. 검증데이터
len(Y_train) #274. 훈련데이터 
len(Y_test)  #118. 검증데이터

#알고리즘준비 : LinearRegression(선형회귀분석)
from sklearn.linear_model import LinearRegression
lr = LinearRegression() #알고리즘 선택 
# 모형 학습
lr.fit(X_train,Y_train)
# 모형 예측. 검증데이터.
y_hat = lr.predict(X_test)
y_hat[:10]  #예측데이터
Y_test[:10] #실제데이터

# 모형 평가
r_square = lr.score(X_test,Y_test)
r_square #결정계수. 값이 작을수록 성능 좋다.
r_square = lr.score(X,Y)
r_square #결정계수. 값이 작을수록 성능 좋다.

# 실제데이터와 예측데이터 시각화
y_hat = lr.predict(X)
y_hat
#y_hat = pd.DataFrame(y_hat)
#y_hat
#Y값과 y_hat예측값을 그래프로 작성
plt.figure(figsize=(10,5))
ax1 = sns.kdeplot(Y, label="Y") #실제 mpg데이터 그래프 
ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
plt.legend()
plt.show()

# 알고리즘 선택 : PolynomialFeatures . 다항 회귀분석
#  단항회귀분석 : ax+b
#  다항회귀분석 : ax**2+bx + c
from sklearn.preprocessing import PolynomialFeatures 
poly = PolynomialFeatures(2) #2차항 선택.
X_train.shape
X_train[:5]
#X_train 데이터를 다항식의 값으로 변환 
X_train_poly=poly.fit_transform(X_train)
X_train_poly.shape
X_train_poly[:5]
pr = LinearRegression() #선형회귀분석. 곡선의형태로 분석 
pr.fit(X_train_poly, Y_train) #학습하기 
#X_train 데이터를 다항식의 값으로 변환 
X_poly = poly.fit_transform(X) #전체 데이터를 곡선의 형태로 변형 
y_hat = pr.predict(X_poly) #예측하기 

plt.figure(figsize=(10, 5))
ax1 = sns.kdeplot(Y, label="Y") #실제 mpg데이터 그래프 
ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
plt.legend()
plt.show()

'''
   단순회귀분석 : 독립변수,종속변수 한개인 경우
      단항 : 직선의 형태로 분석
      다항 : 곡선의 형태로 분석
   다중회귀분석 : 독립변수가 여러개,종속변수 한개인 경우
             Y = b+a1X1 + a2X2 + .... anXn
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/auto-mpg.csv",header=None)
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name'] 
df.head()
df.info()
#horsepower 컬럼을 숫자형으로 변경하기
df.horsepower.unique()
df['horsepower'].replace('?', np.nan, inplace=True)
df['horsepower'].isnull().sum() #결측값의 갯수 
df.dropna(subset=["horsepower"],axis=0,inplace=True)
df['horsepower'] = df['horsepower'].astype('float')
df.info()
ndf = df[['mpg','cylinders','horsepower','weight']]
X=ndf[['cylinders','horsepower','weight']] #독립변수 들.
Y=ndf["mpg"]                               #종속변수

#데이터 분리 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = \
           train_test_split(X,Y,test_size=0.3,random_state=10)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

#선형회귀분석을 이용하여 학습시키기
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)

# 모형 예측. 검증데이터.
y_hat = lr.predict(X)
#예측된 데이터와 실데이터를 kdeplot으로 출력하기
plt.figure(figsize=(10, 5))
ax1 = sns.kdeplot(Y, label="Y")
ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
plt.legend()
plt.show()

#단순 회귀분석의 간단한 예
x=[[10],[5],[9],[7]] #공부시간.
y=[100,50,90,77]     #시험점수
model = LinearRegression()
model.fit(x,y)
result = model.predict([[7],[8]])
result

#다중 회귀분석의 간단한 예
x=[[10,3],[5,2],[9,3],[7,3],[8,2]] #공부시간,학년.
y=[100,50,90,77,85]     #시험점수
#7시간 공부한 2학년학생의 예측 점수 출력하기
#5시간 공부한 3학년학생의 예측 점수 출력하기
#학습되지 않은 model.
model = LinearRegression()
model = model.fit(x, y) #학습하기 
#model : 학습된  model
result =model.predict([[7,2],[5,3]])   #7시간공부 2학년,5시간 공부한 3학년
print(result)
 
####################
# 분류 :질병진단, 스팸메일 
#       설명변수 -> 목표변수 
#    
# KNN(k-Nearset-Neighbors)  

#titanic데이터 로드
import seaborn as sns
import pandas as pd
df = sns.load_dataset("titanic")
df.info()
df.embarked.unique()
df.embark_town.unique()
# deck 컬럼 삭제 : 결측값이 너무 많음
# embark_town 컬럼 삭제 : embarked 중복의미 
rdf = df.drop(['deck','embark_town'],axis=1)
rdf.info()
#age컬럼이 없는 행을 제거
rdf = rdf.dropna(subset=['age'], axis=0)
rdf.info()

#embarked 열의 NaN값을 
#승선도시 중에서 가장 많이 출현한 값으로 치환하기
rdf["embarked"].value_counts(dropna=False)
most_freq = rdf["embarked"].value_counts().idxmax()
most_freq
rdf["embarked"].fillna(most_freq,inplace=True)
rdf.info()
rdf["embarked"].value_counts(dropna=False)

#분석에 필요한 열선택
ndf = rdf[['survived','pclass','sex','age','sibsp','parch','embarked']]
ndf.info()
ndf.describe(include='all')
'''
   원핫인코딩 : 문자열형 범주데이터를 모형이 인식할 수 있도록 숫자형으로 변환이 필요
   pandas.get_dummies()
'''
#sex 컬럼을 원핫인코딩 하기
onehot_sex = pd.get_dummies(ndf["sex"])
onehot_sex
#ndf 데이터와 onehot_sex 데이터를 합하기
ndf = pd.concat([ndf,onehot_sex],axis=1)
ndf.info()
#embarked 컬럼을 원핫인코딩 하기
onehot_embarked = pd.get_dummies(ndf["embarked"],prefix="town")
onehot_embarked
#ndf 데이터와 onehot_embarked 데이터를 합하기
ndf = pd.concat([ndf,onehot_embarked],axis=1)
ndf.info()
#sex,embarked컬럼 제거하기
del ndf["sex"]
del ndf["embarked"]
ndf.info()
#설명변수,목표변수 결정
# 목표변수 : survived 컬럼
# 설명변수 : survived 컬럼 제외한 변수들
X = ndf[['pclass','age','sibsp','parch','female','male',\
         'town_C','town_Q','town_S']]
ndf.columns.difference(['survived'])    
X = ndf[ndf.columns.difference(['survived'])]
Y = ndf['survived']
X[:5]
Y[:5] 
'''
   설명변수의 정규화 필요
   - 분석시에 사용되는 설명변수의 크기에 따라 분석에 영향을 미침.
   - age 컬럼의 범위가 큼.
   - 정규화를 통해서 모든 설명 변수의 값을 기준 단위로 변경함.
'''
from sklearn import preprocessing
import numpy as np
X = preprocessing.StandardScaler().fit(X).transform(X)
X[:5]

#훈련데이터,검증데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.3, random_state=10)
X_train.shape
X_test.shape

# KNN 분류 관련 알고리즘을 이용
# KNN :최근접이웃알고리즘
from sklearn.neighbors import KNeighborsClassifier
#n_neighbors=5 : 5개의 최근접이웃 확인. 
knn = KNeighborsClassifier(n_neighbors=5) #KNN알고리즘 객체
knn.fit(X_train,y_train) #학습
y_hat = knn.predict(X_test) #예측
y_hat[:20] #예측데이터
y_test.values[:20] #실제데이터

#모형의 성능 평가하기
from sklearn import metrics 
#혼동행렬
knn_matrix = metrics.confusion_matrix(y_test, y_hat)  
print(knn_matrix)
'''
confusion_matrix : 혼동행렬
[[109  16]
 [ 25  65]]
'''
#모형 성능 평가 : 평가지표 
knn_report = metrics.classification_report(y_test, y_hat)            
print(knn_report)
