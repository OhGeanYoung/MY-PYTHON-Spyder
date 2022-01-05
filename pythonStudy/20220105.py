# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 09:15:40 2022

@author: Zion_1956
20220105.py
"""
#titanic 데이터 전처리
import seaborn as sns

df = sns.load_dataset("titanic")
df.info()
#누락데이터(결촉값)의 컬럼별 객수 조회하기
df.head().isnull().sum(axis=0) #isnull()결과값이 True인 데이터의 합계
df.isnull().sum(axis=0) #isnull()결과값이 True인 데이터의 합계

#누락데이터(결촉값)이 아닌 컬럼별 개수 조회하기
df.notnull.sum(axis=0) #isnull()결과값이 True인 데이터의 합계

#dropna 함수 : 결촉값을 제거.
#결촉값이 500개 이상인 컬럼 제거하기
#thresh=500 : 결촉값의 개수 이상.
df_thresh = df.dropna(axis=1,thresh=500)
df_thresh.info()

#결촉값을 가진 행 제거
df_age = df_thresh.dropna(subset=["age"],how='any',axis=0)
df_age.info()

###결측값을 치환하기
###평균값으로 치환. 0(초기값)치환, 최빈값 치환, 앞의값 치환 ....
df.info()
# age컬럼의 값이 결측값인 경우 나이 데이터의 평균으로 변경하기
# age컬럼의 평균 조회하기
age_mean = df['age'].mean()
age_mean
df.mean()["age"]
df["age"].fillna(age_mean,inplace=True)
df.info()

#embark_town 컬럼의 결측값을 빈도수가 가장 많은 데이터로 치환하기
df["embark_town"].value_counts(dropna=False)
#idxmax() : 가장 큰 값의 인덱스
most_Freq = df["embark_town"].value_counts(dropna=False).idmax()
most_Freq
# embark_town 컬럼이 결측값을 가진 행을 조회하기
df["embark_town"][df["embark_town"].isnull()]
df.iloc[[61,829]]
df["embark_town"].fillna(most_Freq,inplace=True)
df.info()
df.info[[61,829]]

# embark 컬럼을 앞의 값으로 치환하기
df["embarked"][df["embarked"].isnull()]
df["embarked"][58:62] #61 : C
df["embarked"][825:830] #829: Q
df["embarked"].fillna(method="ffill", inplace=True)
df["embarked"][58:62] #61 : C
df["embarked"][825:830] #829: Q

#중복데이터 처리.
import pandas as pd
df = pd.DataFrame({'c1':['a','a','b','a','b'],
                   'c2':[1,1,1,2,2],
                   'c3':[1,1,2,2,2]})
df
#duplicated() 함수 : 중복데이터 찾기
df_dup = df.duplicated()
df_dup

#특정 컬럼기준 중복 검색
col_dup = df['c1'].duplicated()
df['c1']
col_dup

#drop_duplicates 함수 : 중복행 제거하기
df2 = df.drop_duplicates()
df2()
#특정 컬럼 (c2,c3) 에서 중복 행 제거하기
df[['c2','c3']]
df3 = df.drop_duplicates(subset=['c1','c3'])
df3

#새로운 컬럼 생성하기
df = pd.read_csv('data/auto-mpg.csv',header=None)
df.head()
df.columns=['mpg','cylinders','displacement','horsepower',\
            'weight','acceleration','model year','origin','name']
df.info()
#mpg : mile per gallon 연비
#kpl : kilometer per liter 연비 변환.
# kpl = mpg * 0.425
df['kpl']=df['mpg'] * 0.425
df.info()
df['kpl'].head()
df['mpg'].head()

#round(1) : kpl 컬럼의 값을 소숫점1자리로 변경하기. 반올림하기
df['kpl'] = df['kpl'].round(1)
df['kpl'].head()
df.info()
df['horsepower'].head()
#horsepower 컬럼의 값을 조회하기
df['horsepower'].unique()
df[['mpg','horsepower']].describe()

#오류 데이터 (?) 처리.
#replace 함수 : 결측값으로 변경.
import numpy as np
df['horsepower'].replace('?',np.nan,inplace=True)
df['horsepower'][df['horsepower'].isnull()]
# 결측값 행을 제거하기

df.info()
df.dropna(subset=['horsepower'],axis=0,inplace=True)
df.info()
df['horsepower'].unique()
#horsepower 컬럼의 자료형을 실수형 변환하기
df['horsepower'] = df['horsepower'].astype("float")
df['horsepower'].unique()
df.info()

# 정수형 컬럼을 문자열 형으로 변환하기
df['origin'].unique()
df['origin'].replace({1:'USA',2:'EU',3:'JAPAN'},inplace=True)
df.info()
# 범주형 : 값의 범위를 가지고 있는 자료형(category형)
df['origin'] = df['origin'].astype('category')
df.info()
# 범주형 데이터를 문자열 형으로 변경하기
df['origin'] = df['origin'].astype('str')
df.info()

# 정수형 category 형
age = pd.Series([26,42,27,25,20,20,21,22,23,25])
stu_class = pd.Categorical([1,1,2,2,2,3,3,4,4,4])
gender=pd.Categorical(['F','M','M','M','M','F','F','F','M','M'])
c_df = pd.DataFrame({'age':age,'class':stu_class,'gender':gender})
print(c_df.info())
print(c_df.describe())

c_df['class'] = c_df['class'].astype('int')
print(c_df.info())
print(c_df.describe())

#날짜 데이터 처리
dates = pd.date_range('20220101',periods=6)
dates
df = pd.DataFrame({'count1':[10,2,3,1,5,1],
                   'count2':[1,2,3,6,7,8]},index=dates)
df
#freq='Ms' : 월의 시작일 기준
dates = pd.date_range('20220101',periods=6,freq='MS')
dates
#freq='M' : 월의 종료일 기준
dates = pd.date_range('20220101',periods=6,freq='M')
dates
#freq='3M' : 분기의 종료일 기준
dates = pd.date_range('20220101',periods=6,freq='3M')
dates

## 주식 데이터 분석하기
df=pd.read_csv("data/stock-data.csv")
df.info()
#문자열 형태의 Data 컬럼을 Timestamp로 컬럼인 새컬럼 생성하기
df['new_Date'] = pd.to_datetime(df['Date'])
df.info()
df.head()
#new_Date 컬럼에서, 년,월,일 정보를 각각의 컬럼으로 생성하기
df['Year'] = df['new_Date'].dt.year
df['Month'] = df['new_Date'].dt.month
df['Day'] = df['new_Date'].dt.day
df.head()
df.info()
#월별 평균값을 조회하기
df.groupby("Month").mean()[["Close","Start","Volume"]]

#new_Date 컬럼을 문자열형으로 변경하여 연월일 컬럼으로 저장하기
df["연월일"] = df['new_Date'].astype("str")
df.info()
df.head()

#Date 컬럼을 이용하여 년,월,일 문자열형 컬럼을 생성하기
df["연월일"].str.split("-")
dates
df['년'] = dates.str.get(0)
df['월'] = dates.str.get(1)
df['일'] = dates.str.get(2)
df.info()
df.head()

#필터:조건에 맞는 데이터를 조회하기
#타이타닉 승객중 10대(10~19세)인 승객만 조회하기 df_teenage 데이터에 저장하기
import seaborn as sns
titanic = sns.load_dataset("titanic")
#df_teenage = titanic.loc[(titanic.age >= 10) & (titanic.age < 20)]
mask1 = (titanic.age >=10 ) & (titanic.age < 20)
mask1
df_teenage = titanic.loc[mask1] #mask1 값이 True 레코드만 조회하기
df_teenage.info()
df_teenage.age.unique()

#타이타닉 승객중 10세(age)미만 여성(sex) 승객만 조회하기. df_female_under10 데이터에저장
titanic.sex.unique()
df_female_under10 = titanic.loc[(titanic.age < 10 ) & (titanic.sex == 'female')]
df_female_under10.info()
df_female_under10.age.unique()
df_female_under10.age.unique()

#sibsp : 동행자인원수
titanic.sibsp.unique()
#alone : 동행자존재여부
titanic.alone.unique()
titanic.info()

#동행자의 수가 3,4,5인 승객들의 sibsp,alone 컬럼 조회하기. df_notalone 데이터에 저장
df_notalone = titanic.loc[(titanic.sibsp == 3) | (titanic.sibsp == 4) | (titanic.sibsp == 5),['sibsp','alone']]
df_notalone.info()

df_notalone = titanic.loc[titanic.sibsp.isin([3,4,5]),['sibsp','alone']]
df_notalone.info()

titanic["class"].unique()
#class 컬럼 중 First,Second인 행만 조회하기 df_class12 데이터에 저장
df_class12 = titanic[titanic["class"].isin(["First","Second"])]
df_class12["class"].value_counts()

#df : titanic데이터에서 나이(age),성별(sex),클래스(class),요금(fare),
#   생존여부(survived) 컬럼만 가진 데이터 프레임 객체
df = titanic[['age','sex','class','fare','survived']]
df.info()
#df 레코드 건수 : 승객인원수
len(df)
#class 컬럼으로 분할하기
grouped = df.groupby("class")
grouped
for key,group in grouped :
    print("=== key :",key,end=",")
    print("=== cnt :",len(group),type(group))

for a,b in grouped :
    print("=== key :",key,end=",")
    print("=== cnt :",len(a),type(b))
    
df["class"].value_counts()

#group별 평균
grouped.mean()
df.groupby("class").mean()

#3등석 정보만 조회하기 : get_group
group3 = grouped.get_group("Third")
group3.info()

#class,sex 컬럼으로 분활하기
grouped2 = df.groupby(["class","sex"])
for key,group in grouped2 :
    print("=== key :",key, end=",")
    print("=== cnt :",len(group))
    
#3등성, 여성 정보만 조회하기 : get_group(그룹화된값)
group3f = grouped2.get_group(('First','female'))
group3f.info()

#grouped2 표준편차 구하기
grouped2.std()
grouped.std()

#class로 그룹화한 데이터의 fare 의 표준편차 출력하기
grouped.fare.std()

#클래스별 나이가 가장 많은 나이와, 가장 적은 나이를 출력하기
grouped.age.max()
grouped.age.min()

df.groupby("class").age.max()
df.groupby("class").age.min()
df.groupby("class")['age'].max()
df.groupby("class")['age'].min()

df["class"].unique()
#df.class.unique #class 가 예약어임
df.age.unique()

#agg() 함수 : 여러개의 함수를 여러개의 컬럼에 적용할 수 있는 함수
#                   사용자 정의함수 적용
def max_min(x) :
    return x.max() - x.min()

#grouped 데이터에 max_min함수 적용
agg_maxmin = grouped.agg(max_min)
agg_maxmin

#class로 그룹화하여 최대값 조회
grouped.max()
grouped.agg(max)

#class로 그룹화하여 최대,최소값 조회
grouped.agg(['max','min'])

#각 컬럼마다 다른 함수 적용할 수 있음.
# 요금 : 평균값, 최대값
# 나이 : 평균값
grouped.add({'fare':['mean','max'],'age':'mean'})

#filter(조건)함수 : 그룹화 데이터의 조건 설정
#grouped 데이터의 개수가 200개 이상인 그룹만 조회하기
grouped.count()
#filter1 : 1,3등석 레코드만 저장
filter1 = grouped.filter(lambda x : len(x) >= 200)
filter1.info()

#age컬럼의 평군이 30보다 작은 그룹만을 filter2에 저장하기
grouped.mean()
filter2 = grouped.filter(lambda x : x.age.mean() < 30)
filter2.info()

#stockprice.xlsx,stockvaluation.xlsx
# 파일을 읽어 각각 df1,df2에 저장하기
df1 = pd.read_excel("data/stockprice.xlsx")
df2 = pd.read_excel("data/stockvaluation.xlsx")
df1
df2
#concat() : df1,df2를 열기준으로 연결하기 : 물리적인 연결.
result1 = pd.concat([df1,df2],axis=1)
result1
df1.info()
df2.info()
result1.info()

#merge() : id 컬럼을 기준으로 같은 id값을 가진 레코드 병합(sql:join)
result2 = pd.merge(df1,df2)
result2
result2.info()
'''
on="id" : 연결컬럼. 데이터프레임의 id컬럼의 값이 같은 경우 병합
how="outer" : 두데이터프레임의 값이 같지 않은 경우도 조회. 값이 같지 않은 경우 상대 컬럼의 값은 NaN임
                sql 의 full outer join과 같은 의미임.
'''
result3 = pd.merge(df1,df2,on="id",how="outer")
result3






