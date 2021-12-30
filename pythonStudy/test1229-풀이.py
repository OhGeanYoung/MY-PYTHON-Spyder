# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 15:50:18 2021

@author: KITCOOP
test1229-풀이.py
"""
#1. 년도별 서울의 전입과 전출 정보를 막대그래프로 작성하기
# 20211229-1.png 파일 참조
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import  rc
rc('font', family="Malgun Gothic") #현재 폰트 변경 설정.
df = pd.read_excel('시도별 전출입 인구수.xlsx', header=0)
df.info()
#결측값을 앞쪽 데이터로 채우기
df = df.fillna(method='ffill') 
#전출데이터  
mask = ((df['전출지별'] == '서울특별시') & (df['전입지별'] == '전국')) 
df_seoulout = df[mask]
print(df_seoulout)
#전출지별 컬럼 제거.
df_seoulout = df_seoulout.drop(['전출지별'], axis=1)
print(df_seoulout)
#인덱스값으로 전입지별 컬럼을 설정
df_seoulout.set_index('전입지별', inplace=True)
print(df_seoulout)
#전국 인덱스명을 전출건수 인덱스명 변경
df_seoulout.rename({'전국':'전출건수'}, axis=0, inplace=True)
print(df_seoulout)
mask = ((df['전입지별'] == '서울특별시') & (df['전출지별'] == '전국'))
df_seoulin = df[mask]
print(df_seoulin)
#전입지별 컬럼 제거 
df_seoulin = df_seoulin.drop(['전입지별'], axis=1)
#전출지별 컬럼을 인덱스로 변경
df_seoulin.set_index('전출지별', inplace=True)
#전국 인덱스명을 전입건수 인덱스명으로 변경 
df_seoulin.rename({'전국':'전입건수'}, axis=0, inplace=True)
print(df_seoulin)
#pd.concat : df_seoulout,df_seoulin 두개의 데이터프레임을 연결
df_seoul = pd.concat([df_seoulout,df_seoulin])
print(df_seoul)
df_seoul = df_seoul.T #전치 행렬.
print(df_seoul)
df_seoul.plot(kind='bar', figsize=(20, 10), width=0.7,
          color=['orange', 'green'])
plt.title('서울 전입 전출 건수', size=30)
plt.ylabel('이동 인구 수', size=20)
plt.xlabel('기간', size=20)
plt.ylim(1000000, 3500000)
plt.legend(loc='best', fontsize=15)
plt.show()
plt.savefig("20211229-1.png",dpi=400,bbox_inches="tight")


#2. 년도별 서울의 전입과 전출 정보이용하여 순수증감인원수를 
#  선그래프로 작성하기
# 20211229-2.png 파일 참조
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import  rc
rc('font', family="Malgun Gothic") #현재 폰트 변경 설정.
df = pd.read_excel('시도별 전출입 인구수.xlsx', header=0)
df = df.fillna(method='ffill')    
mask = ((df['전출지별'] == '서울특별시') & (df['전입지별'] == '전국')) 
df_seoulout = df[mask]
df_seoulout = df_seoulout.drop(['전출지별'], axis=1)
df_seoulout.set_index('전입지별', inplace=True)
df_seoulout.rename({'전국':'전출건수'}, axis=0, inplace=True)
print(df_seoulout)
mask = ((df['전입지별'] == '서울특별시') & (df['전출지별'] == '전국'))
df_seoulin = df[mask]
df_seoulin = df_seoulin.drop(['전입지별'], axis=1)
df_seoulin.set_index('전출지별', inplace=True)
df_seoulin.rename({'전국':'전입건수'}, axis=0, inplace=True)
print(df_seoulin)
df_seoul = pd.concat([df_seoulout,df_seoulin])
print(df_seoul)
df_seoul = df_seoul.T
print(df_seoul)
df_seoul["증감수"] = df_seoul["전입건수"] - df_seoul["전출건수"]
print(df_seoul)
plt.rcParams['axes.unicode_minus']=False #음수 출력 설정 
plt.style.use('ggplot') 
df_seoul["증감수"].plot()
plt.title('서울 순수 증감수', size=20)
plt.ylabel('이동 인구 수', size=20)
plt.xlabel('기간', size=20)
plt.legend(loc='best', fontsize=15)
plt.show()
plt.savefig("20211229-2.png",dpi=400,bbox_inches="tight")

#3. 남한의 전력량을(수력,화력,원자력)을 연합막대그래프로 작성하고,
#   전력증감율을 선그래프로 작성하기
# 20211229-3.png 파일 참조

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')   # 스타일 서식 지정
plt.rcParams['axes.unicode_minus']=False   # 마이너스 부호 출력 설정
df = pd.read_excel('남북한발전전력량.xlsx')
df = df.loc[0:4] #남한지역의 발전전력량만 조회.
print(df.head())
df.drop('전력량 (억㎾h)', axis='columns', inplace=True)
print(df.head())
df.set_index('발전 전력별', inplace=True)
print(df.head())
df = df.T
print(df.head())
df = df.rename(columns={'합계':'총발전량'})
print(df.head())
df['총발전량 - 1년'] = df['총발전량'].shift(1)
print(df.head())
df['증감율']=((df['총발전량'] / df['총발전량 - 1년']) - 1) * 100      
print(df)
ax1 = df[['수력','화력','원자력']].plot(kind='bar', \
             figsize=(20, 10),  width=0.7, stacked=False)  
ax2 = ax1.twinx() #ax1 영역을 복사해서 ax2에 할당. 
ax2.plot(df.index, df.증감율, ls='--', marker='o', markersize=10, 
        color='green', label='전년대비 증감율(%)')  
ax1.set_ylim(0, 5500)
ax2.set_ylim(-50, 50)
ax1.set_xlabel('연도', size=20)
ax1.set_ylabel('발전량(억 KWh)')
ax2.set_ylabel('전년 대비 증감율(%)')
plt.title('남한 전력 발전량 (1990 ~ 2016)', size=30)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()
plt.savefig("20211229-3.png",dpi=400,bbox_inches="tight")

