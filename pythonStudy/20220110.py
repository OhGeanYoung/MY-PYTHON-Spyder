# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 09:18:12 2022

@author: Zion_1956
20220110.py
"""

'''
서울시 각 구별 cctv수를 파악하고, 인구대비 cctv 비율을 파악해서 순위 비교
서울시 각 구별 cctv수 : 01. cctv_in_seoul.csv
서울시 인구 현황 :      01. population_in_Seoul.xls
'''
import pandas as pd
CCTV_Seoul = pd.read_csv("data/01. CCTV_in_Seoul.csv")
CCTV_Seoul.info()
Pop_Seoul = pd.read_excel("data/01. population_in_Seoul.xls")
Pop_Seoul.info()

'''
    header 정보 : 3번째행, 2번 인덱스.
    셀 : B,D,G,J,N
'''
Pop_Seoul = pd.read_excel("data/01. population_in_Seoul.xls",header=2,usecols="B,D,G,J,N")
Pop_Seoul.info()
Pop_Seoul.head()

#컬럼명 변경하기
# CCTV_Seoul : 기관명 -> 구별
CCTV_Seoul.rename(columns={"기관명":"구별"},inplace=True)
CCTV_Seoul.info()

#pop_Seoul : 자치구 => 구별, 계=>인구수, 계.1=>한국인, 계2=>외국인
#                           65세이상 고령자 => 고령자
Pop_Seoul.columns = ["구별","인구수","한국인","외국인","고령자"]
Pop_Seoul.info()
Pop_Seoul.head()
#drop 함수 : 첫번째 행 제거하기
Pop_Seoul.drop([0],inplace=True,axis=0)
Pop_Seoul.head()

#CCTV 최근증가율이 높은 구 5개를 조회하기
#1. 최근증가율 컬럼 생성 :
#   2014년부터 2016년 까지 최근 3년간 CCTV수의 합을
#   2013년 이전 CCTV수로 나눠 *100을 해서 % 단위로 계산
#2. 최근증가율 컬럼으로 정렬하여 상위 5개만 조회

CCTV_Seoul["최근증가율"] = \
(CCTV_Seoul["2014년"]+CCTV_Seoul["2015년"]+CCTV_Seoul["2016년"])\
    /CCTV_Seoul["2013년도 이전"] * 100
CCTV_Seoul.info()
CCTV_Seoul.head()
CCTV_Seoul.sort_values(by="최근증가율",ascending=False).head()

#외국인비율, 고령자비율이 높은 구 5개 조회하기
#1. 인구 데이터의 외국인비율, 고령자비율 컬럼 추가하기
#   외국인비율 : 외국인/인구수 * 100
#   고령자비율 : 고령자/인구수 * 100
#2. 외국인비율 순으로 내림차순 정렬하여 상위 5개 조회
#3. 고령자비율 순으로 내림차순 정렬하여 상위 5개 조회
Pop_Seoul["외국인비율"] = Pop_Seoul["외국인"]/Pop_Seoul["인구수"] * 100
Pop_Seoul["고령자비율"] = Pop_Seoul["고령자"]/Pop_Seoul["인구수"] * 100
Pop_Seoul.info()
Pop_Seoul.sort_values(by="외국인비율",ascending=False).head()
Pop_Seoul.sort_values(by="고령자비율",ascending=False).head()

#구별컬럼을 연결 컬럼으로 CCTV 데이터와 인구 데이터 합치기.(data_result)
#CCTV 데이터중 2013년도 이전, 2014년,2015년,2016년 컬럼 제거하기
data_result = pd.merge(CCTV_Seoul,Pop_Seoul, on='구별')
data_result.head()
data_result.info()
#del 명령으로 컬럼 제거
#del data_result["2013년도 이전"],data_result["2014년"],\
#    data_result["2015년",data_result["2016년"]
#drop 함수로 컬럼 제거
data_result.drop(["2013년도 이전","2014년","2015년","2016년"],axis=1,inplace=True)
data_result.info()

#구별 컬럼을 인덱스로 변경하기
data_result.set_index("구별",inplace=True)
data_result.info()
data_result.head()
#인구수과 소계컬럼 두개의 피처간의 상관계수 구하기
data_result[["인구수","소계"]].corr()
import seaborn as sns
sns.pairplot(data_result[["인구수","소계"]])
'''
    CCTV 비율 : 인구수대비 CCTV 개수 CCTV 개수 / 인구수 * 100
    CCTV 비율 컬럼 추가하기
    CCTV 비율이 높은 순으로 수평막대 그래프 출력하기.
'''
data_result['CCTV비율'] = data_result['소계']/data_result['인구수'] * 100
data_result.head()
data_result['CCTV비율'].sort_values().plot(kind='barh', grid=True, figsize=(10,10))

#인구수와 소계의 산점도를 matplot 모듈을 이용하여 출력하기
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(data_result['인구수'],data_result["소계"], s=50)
plt.xlabel("인구수")
plt.ylabel("CCTV 개수")
plt.grid()
plt.show()
#인구수와 소계 데이터의 산점도, 회귀선 출력하기
import numpy as np
# 상수값 (기울기,y절편) : 모든점과 차이가 가장 적은 직선의 기울기, y절편 값 리턴
# 1 : 직선, 1차함수
# 2 : 곡선, 2차함수

fp1 = np.polyfit(data_result['인구수'], data_result['소계'],1)
fp1
f1 = np.poly1d(fp1) #fp1 함수
fx = np.linspace(100000,700000, 100) #10만 70만까지 값을 100개로 균등분할
f1(100000) # y=f1(x), 예측 cctv 개수 = f1(인구수)
plt.figure(figsize=(6,6))
plt.scatter(data_result['인구수'], data_result["소계"], s=50)
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g') #회귀선
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()

#오차 컬럼 생성하기
data_result["오차"] = data_result["소계"] - f1(data_result["인구수"])
data_result["오차"].head()
#오차 : |실제 CCTV의 개수 - 예상 CCTV 개수|
#오차가 가장 큰 순으로 정렬하기
df_sort = data_result.sort_values(ny="오차",ascending=False)
df_sort.오차.head()
#오차 값에 따라 색상 표시하기
plt.figure(figsize=(14,10))
plt.scatter(data_result['인구수'],data_result['소계'],
            c=data_result['오차'], s=50)
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g') #회귀선
#점에 해당하는 구의이름 오차가 많은 구 10개 정보를 표시하기

for n in range(10):
    plt.text(df_sort['인구수'][n]*1.02,df_sort['소계'][n]*0.98,
             df_sort.index[n], fontsize=15)
    
plt.xlabel('인구수')
plt.ylabel('CCTV 개수')
plt.colorbar()
plt.grid()
plt.show()

#서울시 경찰서별 범죄율 데이터 읽기. 경찰서 명을 제외한 데이터는 숫자형으로 읽기
import numpy as np
import pandas as pd
crime_Seoul = pd.read_csv('data/02. crime_in_seoul.csv',
                          thousands=',',encoding='euc-kr')
crime_Seoul.info()
crime_Seoul.head()
#경찰서 위치 데이터 읽기
police_state = pd.read_csv('data/경찰관서 위치.csv',encoding='euc-kr')
police_state.info()
police_state["지방청"].unique()
#police_Seoul 데이터에 서울청 데이터만 저장하기
police_Seoul = police_state[police_state["지방청"]=='서울청']
police_Seoul.head()
police_Seoul.head()
police_Seoul["경찰서"].unique()
len(police_Seoul["경찰서"].unique())

#police_Seoul 데이터의 경찰서 컬럼의 내용을 'xx서' 이름으로 저장하기
# crime_Seoul 데이터의 관서명과 같은 형식으로 변경하기
police_Seoul["관서명"] = police_Seoul["경찰서"].apply((lambda x : str(x[2:]+'서')))
police_Seoul["관서명"].unique()
police_Seoul.info()
crime_Seoul["관서명"].unique()
police_Seoul.head()

#police_Seoul 데이터에 지방청, 경찰서, 구분 컬럼 제거하기
del police_Seoul["지방청"],police_Seoul["경찰서"],police_Seoul["구분"]
#drop_duplicates 함수 : 중복행 제거, 관서명이 중복된 데이터 제거하기.
police_Seoul = police_Seoul.drop_duplicates(subset=['관서명'])
police_Seoul
police_Seoul.info()

#police_Seoul 데이터의 주소 컬럼을 이용하여 구별 컬럼을 생성하기
#police_Seoul 데이터의 주소 컬럼 제거하기
police_Seoul["구별"] = police_Seoul["주소"].apply(lambda x : str(x).split()[1])
del police_Seoul["주소"]
police_Seoul.info()
police_Seoul

#관서명 컬럼을 연결컬럼으로 crime_Seoul 데이터와 police_Seoul 데이터를 병합하기
#data_result 데이터에 저장하기

data_result = pd.merge(crime_Seoul,police_Seoul,on="관서명")
data_result.head()

#구별 범죄의 합계를 출력하기
data_result.groupby("구별").sum()

crime_sum = pd.pivot_table(data_result,index="구별",aggfunc=np.sum)
crime_sum
#crime_sum 데이터에 강간, 강도,살인,절도,폭력 검거율 컬럼 추가하기.
#검거율 = 검거/발생 * 100.
crime_sum["강간검거율"] = crime_sum["강간 검거"]/crime_sum["강간 발생"] * 100
crime_sum["강도검거율"] = crime_sum["강도 검거"]/crime_sum["강도 발생"] * 100
crime_sum["살인검거율"] = crime_sum["살인 검거"]/crime_sum["살인 발생"] * 100
crime_sum["절도검거율"] = crime_sum["절도 검거"]/crime_sum["절도 발생"] * 100
crime_sum["폭력검거율"] = crime_sum["폭력 검거"]/crime_sum["폭력 발생"] * 100
crime_sum.info()
crime_sum

#검거율 데이터 중 100보다 큰 값은 100으로 변경하기
crime_sum.loc[crime_sum["강도검거율"]>100,"강도검거율"]=100
crime_sum.loc[crime_sum["강도검거율"]>100,"강도검거율"]

col_list=['강간','강도','살인','절도','폭력']
for col in col_list:
    crime_sum.loc[crime_sum[col+"검거율"]>100,col+"검거율"]=100
    
crime_sum.info()
#검거 컬럼 제거하기
for column in col_list:
    del crime_sum[col+"검거"]
crime_sum.info()
## 강간 발생 컬럼을 강간, 강도 발생 컬럼을 강도,살인 발생 컬럼을 살인 ....
## 컬럼명 변경하기
crime_sum.rename(columns={"강간 발생":'강간',"강도 발생":'강도',"살인 발생":'살인',"절도 발생":'절도',"폭력 발생":'폭력'},inplace=True)
crime_sum.info()
crime_sum

#구별 절도검거율을 수평막대그래프로 출력하기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family ='malgun Gothic')
plt.figure()
crime_sum['절도검거율'].sort_values().plot(kind='barh', grid=True, figsize=(10,10))
plt.title("서울시 구별 절도 검거율")
plt.show()

#구별 검거율과, CCTV 개수를 산점도와 회귀선으로 출력하기.
# 오차가 큰 10개 구 이름을 그래프로 출력하기
CCTV_Seoul
crime_sum = crime_sum.reset_index()
crime_sum
#구별컬럼으로 CCTV_Seoul,crime_sum 변경하기
data_result = pd.merge(CCTV_Seoul,crime_sum, on="구별")
data_result.info()
data_result.head()

#절도검거율과 CCTV 회귀선과 산점도 출력하기
fp1 = np.polyfit(data_result['소계'], data_result['절도검거율'],1)
f1 = np.poly1d(fp1)
fx = np.linspace(500,4000,100)
data_result['오차'] = np.abs(data_result['절도검거율'] - f1(data_result['소계']))
df_sort = data_result.sort_values(by='오차', ascending=False)
df_sort.info()
plt.figure(figsize=(14,10))
plt.scatter(df_sort['소계'],df_sort["절도검거율"],c=df_sort['오차'], s=50)
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
for n in range(10):
    plt.text(df_sort['소계'][n]*1.001,df_sort['절도검거율'][n]*0.999,
             df_sort['구별'][n], fontsize=15)
plt.xlabel('CCTV 개수')
plt.ylabel('절도범죄 검거율')
plt.title("CCTV와 절도 검거율 분석")
plt.colorbar()
plt.grid()
plt.show()




















