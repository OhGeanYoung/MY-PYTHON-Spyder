# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:50:46 2022

@author: KITCOOP
test0111-풀이.py
"""
'''
1. seoul_5.csv 데이터를 이용하여 2022년 1월 11일의 평균기온을 예측해보기.
=> 결과는 틀릴수 있음
결과
단순회귀분석 : [-2.65642349]
다항회귀분석 : [-1.70084221]
'''

#단순 회귀 분석
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family ='Malgun Gothic')
seoul = pd.read_csv("data/seoul_5.csv", encoding="euc-kr")
seoul.info()
seoul.head()
#년도컬럼 생성 
seoul['년도'] = seoul['날짜'].apply(lambda x: x[:4])
seoul['년도'].head()
#seoul0111 : 매년 01-11일자 정보 저장 
seoul0111 = seoul[seoul['날짜'].apply(lambda x: x[5:])=='01-11']
seoul0111.head()
seoul0111.info()
#결측값가진 행 제거
seoul0111.dropna(subset=["평균기온(℃)"],axis=0, inplace=True)
seoul0111['년도']
seoul0111["평균기온(℃)"]

from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = seoul0111[["년도"]] #훈련데이터
Y = seoul0111['평균기온(℃)']
model.fit(X, Y)
result = model.predict([['2022']]) #2022년도의 평균기온 예측
print('단순회귀분석 : ',result)

#다항 회귀 분석
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family ='Malgun Gothic')
seoul = pd.read_csv("data/seoul_5.csv", encoding="cp949")
seoul['년도'] = seoul['날짜'].apply(lambda x: x[:4])
seoul0111 = seoul[seoul['날짜'].apply(lambda x: x[5:])=='01-11']
seoul0111.dropna(subset=["평균기온(℃)"],axis=0, inplace=True)
seoul0111.info()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
from sklearn.preprocessing import PolynomialFeatures   #다항식 변환
poly = PolynomialFeatures(degree=2) # 2차항 적용 
X = seoul0111[["년도"]]
Y = seoul0111['평균기온(℃)']
X_poly=poly.fit_transform(X) # 다항식으로 변형.
model.fit(X_poly, Y)
X_test_poly = poly.fit_transform([['2022']]) #다항식으로 변형
result = model.predict(X_test_poly)
print('다항회귀분석 : ',result)

'''
2. seoul_5.csv 데이터를 이용하여 2022년 1월 11일의 최고기온을 
    예측해보기.

   [결과] 
    다중회귀분석 :  [-2.31790043]
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family ='Malgun Gothic')
seoul = pd.read_csv("data/seoul_5.csv", encoding="cp949")
seoul['년도'] = seoul['날짜'].apply(lambda x: x[:4])
seoul0111 = seoul[seoul['날짜'].apply(lambda x: x[5:])=='01-11']
seoul0111.dropna(subset=["최고기온(℃)"],axis=0, inplace=True)
seoul0111.info()
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = seoul0111[["년도","최저기온(℃)"]]
Y = seoul0111['최고기온(℃)']
model.fit(X, Y)
result = model.predict([['2021',-10]])
print('다중회귀분석 : ',result)


'''
3. 01월11일에 해당하는 평균온도를 산점도,회귀선 출력하기
'''
seoul0111.info()
seoul0111['년도'] =seoul0111['년도'].astype("int64")
seoul0111.info()
fp1 = np.polyfit(seoul0111['년도'], seoul0111['평균기온(℃)'], 2)
f1 = np.poly1d(fp1)
fx = np.linspace(1907, 2017, 108)
seoul0111.plot(kind='scatter', x='년도', y='평균기온(℃)',  \
         c='coral', s=10, figsize=(10, 5))
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
plt.show()

fig = plt.figure(figsize=(10, 5))   
ax1 = fig.add_subplot(1, 1, 1)
sns.regplot(x='년도', y='평균기온(℃)', data=seoul0111, ax=ax1) 
plt.show()

sns.jointplot(x='년도', y='평균기온(℃)', data=seoul0111, kind='reg')
plt.show()
#seaborn pariplot 
sns.pairplot(seoul0111,kind='reg')
plt.show()
