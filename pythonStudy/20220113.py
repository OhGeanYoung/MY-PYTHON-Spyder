# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 09:18:11 2022

@author: KITCOOP
20220113.py
"""
#  iris 데이터를 이용하여 군집 알고리즘 연습하기
#sklearn : 사이킷런 모듈
from sklearn import datasets
iris = datasets.load_iris() #iris 데이터 로드
type(iris)
iris #딕셔너리 형태로 저장 
iris.target
iris.data
import pandas as pd
labels = pd.DataFrame(iris.target)
labels.columns=['labels']
data = pd.DataFrame(iris.data)
data.columns=\
    ["Sepal length","Sepal width","Petal length","Petal width"]
data = pd.concat([data,labels],axis=1)    
data.info()

#꽃받침 정보로만 그룹화하기
feature = data[ ['Sepal length','Sepal width']]

#Kmeans 알고리즘으로 군집화하기
from sklearn import cluster
model = cluster.KMeans(init="k-means++", n_clusters=3)
model.fit(feature)
predict = pd.DataFrame(model.predict(feature))
predict
predict.columns=['predict'] #컬럼명 설정
predict
# feature, predict 데이터 컬럼 기준 연결
r = pd.concat([feature,predict],axis=1)
r.head()

#예측된 데이터를 그래프로 표시하기
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
fig = plt.figure()
plt.scatter(r['Sepal length'],r['Sepal width'],c=r['predict'],alpha=0.5)
plt.title("예측 cluster")
plt.show()
#실제 데이터를 그래프로 표시하기
fig = plt.figure()
plt.scatter(data['Sepal length'],data['Sepal width'],c=data['labels'],alpha=0.5)
plt.title("예측 cluster")
plt.show()

#군집화 평가하기 => 의미없다.
# 예측된 클러스터값과 실제 labels 데이터 혼동 행렬 출력하기
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix\
    (data["labels"].values, r['predict'].values)
print(confmat)
#academy1.csv 파일을 읽어서 3개로 군집화하기
# KMeans 알고리즘을 이용
# 국어점수:100,영어:80 그룹 ?
# 국어점수:60,영어:50 그룹 ?

import pandas as pd
data = pd.read_csv("data/academy1.csv")
data.info()
from sklearn.cluster import KMeans
model = KMeans(init="k-means++", n_clusters=3)
model.fit(data.iloc[:,1:]) #학번컬럼을 제외
model.predict([[100,80]])
model.predict([[60,50]])

#예측하기
result = model.predict(data.iloc[:,1:])
result
data["group"]=result
data
fig = plt.figure()
plt.scatter(data['국어점수'],data['영어점수'],c=data['group'],alpha=0.5)
plt.colorbar()
plt.show()

#보스톤 주택가격 정보 예측. 
'''
TOWN : 도시명
LON : 경도
LAT : 위도
CMEDV : 주택가격 ($1000)
CRIM : 1인당 범죄율
ZN : 25000 평방피트를 초과하는 거주지역의 비율
INDUS: 비상업지역 토지 비율
CHAS : 찰스강 경계
NOX : 10ppm 당 일산화질소
RM : 주택 1가구당 평균 방의 개수
AGE : 1940년 이전의 건축된 주택의 비율
DIS : 5개의 보스턴직업센터까지의 접근성 지수
RAD : 방사형 도로까지의 접근성 지수
TAX : 10000 달러당 재산세율
PTRATIO : 타운별 학생/교사 비율
B : 흑인의 비율
LSTAT : 모집단의 하위계층의 비율 (%)
'''
#1. 데이터 로드
import pandas as pd
housing = pd.read_csv("data/BostonHousing2.csv")
housing.info()

# CMEDV,NOX,RM,LSTAT 컬럼을 산점도로 출력하기 
cols= ['CMEDV','NOX','RM','LSTAT']
import seaborn as sns
sns.pairplot(housing[cols])
plt.show()
X = housing[['NOX','RM','LSTAT','TAX']]
Y = housing['CMEDV']
#정규화하기
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[:10]
# 훈련:검증데이터 8:2비율로 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split\
    (X, Y, test_size=0.2, random_state=10)

#회귀모델 생성
from sklearn import linear_model
lr = linear_model.LinearRegression()
#학습하기
model = lr.fit(X_train, y_train) #학습하기
'''
 R-squared (결정계수)
  - 회귀모델 독립변수가 종속변수의 관계를 수치 표시함.
  - 독립변수가 종속변수를 얼마나 설명할 수 있는지를 수치 나타냄
  - 0 ~ 1 사이의 값
  - 독립변수의 갯수가 많아지면 값은 높아짐.
 상관계수 : -1 ~ 1 사이의 값  
'''
print(model.score(X_train, y_train)) #0.6534415218879266
print(model.score(X_test, y_test))   #0.5674059746656421

from sklearn.metrics import mean_squared_error
from math import sqrt
'''
 mse : mean_squared_error(평균제곱오차)
     (실제데이터-예측데이터)**2 의 평균 
     - 작은값일 수록 정확성이 높아짐. 
 rmse : mse의 제곱근.      
     - 작은값일 수록 정확성이 높아짐. 
'''
y_pre = lr.predict(X_train)
rmse = sqrt(mean_squared_error(y_train,y_pre))
print(rmse)

#검증 데이터의 RMSE 점수 출력하기
y_pre_test = lr.predict(X_test)
rmse = sqrt(mean_squared_error(y_test,y_pre_test))
print(rmse)

# 투수들의 연봉 예측하기
import pandas as pd
picher = pd.read_csv("data/picher_stats_2017.csv")
picher.info()
#1. 2018년도연봉 컬럼을 y로 변경하기
picher = picher.rename(columns={'연봉(2018)':'y'})
picher.info()
picher['팀명'].unique()
#2. 팀명을 one-hot 인코딩하기. picher데이터셋에 추가하기
#   pandas.get_dummies() 함수이용
onehot_team = pd.get_dummies(picher['팀명'])
onehot_team
picher = pd.concat([picher,onehot_team],axis=1)
picher.info()
#3. 팀명 컬럼 제거하기
del picher['팀명']
picher.info()

#4. 선수명,y 을 제외한 모든 컬럼을 독립변수 X로 설정하기
X = picher[picher.columns.difference(['선수명','y'])]
Y= picher['y']
X.info()
Y[:10]

#정규화 함수로 정규화 하기 
def standard_scaling(df, scale_columns):
   for col in scale_columns:
      s_mean = df[col].mean()
      s_std = df[col].std()
      df[col] = df[col].apply(lambda x: (x - s_mean)/s_std)
   return df

scale_columns = X.columns.difference(onehot_team.columns)
scale_columns
picher_df = standard_scaling(X, scale_columns)
picher_df.head()
#훈련 검증 데이터 분리
X_train, X_test, y_train, y_test = \
   train_test_split(X, Y, test_size=0.2, random_state=19)

import statsmodels.api as sm
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit() #회귀분석
model.summary()
'''
R-squared: 결정계수. 독립변수의 갯수가 많아지면 값이 높아진다  
Adj. R-squared : 수정 결정계수.
P>|t| : t(검정통계량) : 컬럼의 내용이 유의미 여부를 나타내는 값.
     0.05미만인 피처가 회귀분석에서는 유의미한 피처임.
coef : 회귀계수. 독립변수별 종속변수에 미치는 영향 수치로 표현
       상관계수가 낮은경우는 신뢰할 수 없다.
'''
#그래프로 출력
plt.rcParams['figure.figsize'] = [20, 16]
plt.rc('font', family='Malgun Gothic')
coefs = model.params.tolist() #데이터값
coefs 
coefs_series = pd.Series(coefs)
coefs_series.head()
x_labels = model.params.index.tolist()
ax = coefs_series.plot(kind='bar') #막대그래프 
ax.set_title('feature_coef_graph')
ax.set_xlabel('x_features')
ax.set_ylabel('coef')
ax.set_xticklabels(x_labels)
x_labels
'''
VIF : 분산팽창요인.10~15 정도가 넘으면 다중공선성 문제 발생의 가능성이 있다.
다중공선성 : 독립변수들은 서로 독립적이어야 함. 서로 연관성이 없는게 좋다.
            독립변수들 사이의 연관성이 높은 경우 가중치발생.
            연관성이 높은 독립변수는 하나만 선택해야함 
FIP 컬럼과 KFIP 변수는 한개만 선택해야 함.            
'''
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) \
                     for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif.round(1))

#회귀 분석
X_train, X_test, y_train, y_test = \
   train_test_split(X, Y, test_size=0.2, random_state=19)

from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
#r2-score 값 출력하기
# 훈련데이터 r2score > 테스트데이터 r2score : 과대적합 가능성
model.score(X_train, y_train)
model.score(X_test, y_test)
#mse : 평균제곱오차
#rmse : sqrt(평균제곱오차) : 값이 작을수록 좋은 모델
#rmse 값 출력하기
from math import sqrt
from sklearn.metrics import mean_squared_error
y_predictions = lr.predict(X_train)
print(sqrt(mean_squared_error(y_train, y_predictions))) # 7282.718684746373
y_predictions = lr.predict(X_test)
print(sqrt(mean_squared_error(y_test, y_predictions))) # 14310.69643688916
# 훈련 rmse > 테스트 rmse : 과대적합 됨.
y_pred = lr.predict(X)
y_pred[:5]
picher["y"].head()
picher = pd.read_csv("data/picher_stats_2017.csv")
result_df = picher[['선수명','연봉(2018)','연봉(2017)']]
result_df.head()
pred_df = pd.DataFrame(y_pred)
pred_df.columns=['예측연봉']
pred_df.head()
result_df = pd.concat([result_df,pred_df],axis=1)
result_df.info()
#2018년도 연봉의 내림차순 정렬
result_df = result_df.sort_values(by=['연봉(2018)'], ascending=False)

# 2018년도 급여 많은선수 10명만 그래프로 출력하기
result_df = result_df.iloc[:10, :]
plt.rc('font', family='Malgun Gothic')
result_df.plot(x='선수명',\
       y=['연봉(2017)', '연봉(2018)','예측연봉'], kind="bar")
    
### 시계열 분석
### 시계열 데이터 : 연속적인 시간에 따라 다르게 측정되는 데이터. 
###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# https://www.blockchain.com/ko/charts/market-price 다운
file_path = 'data/market-price.csv'
bitcoin_df = pd.read_csv(file_path, names=['day', 'price'],header=0)
bitcoin_df.head()
bitcoin_df.info()
#day 컬럼을 날짜형 변환
bitcoin_df['day'] = pd.to_datetime(bitcoin_df["day"])
bitcoin_df.info()
bitcoin_df.head()
#day 컬럼을 index로 변환
bitcoin_df.set_index('day',inplace=True)
bitcoin_df.info()
bitcoin_df.head()
#시각화
bitcoin_df.plot()
plt.show()
'''
  ARIMA : 시계열 분석 알고리즘
  
  AR : 과거 정보를 이용하여 현재 정보 계산. AutoRegression
  MA : 이전항의 오차를 이용하여 현재항을 추론.
       Moving Average
  ARMA = AR + MA
  ARIMA : ARMA + 추세 변동 경향을 반영.
         ARMA의 개선된 알고리즘
         AutoRegression Integrated Moving Average
'''
from statsmodels.tsa.arima_model import ARIMA
#알고리즘 선택 
'''
   order=(2,1,2)
   2 : AR관련 데이터. 2번째 과거
   1 : 차분(Difference). 현재상태 - 이전상태의 차이.
       시계열데이터의 불규칙성을 보정
   2 : MA. 2번째 과거 정보의 오차를 이용해서 현재 추론.    
'''
model = ARIMA(bitcoin_df.price.values,order=(2,1,2))
#학습
model_fit = model.fit(trend='c',full_output=True,disp=True)
fig = model_fit.plot_predict() #예측한 결과 시각화
#resid : 잔차 
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
#5일 예측데이터 
forecast_data = model_fit.forecast(steps=5)
'''
  3개의 배열 리턴
  1번배열 : 예측값. 5일치 예측값
  2번배열 : 표준오차  
  3번배열 : [예측하한값, 예측상한값]
'''
forecast_data

