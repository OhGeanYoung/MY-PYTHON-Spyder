# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 08:33:16 2021

@author: KITCOOP
20211230.py
"""
import seaborn as sns
import matplotlib.pyplot as plt
titanic = sns.load_dataset("titanic")

# box 그래프 작성하기
# 박스그래프 : 값의 범주 표시.
# 바이올린 그래프 : 값의 범주와 분포도를 표시 
fig = plt.figure(figsize=(15, 10))   
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
# 생존여부별 나이값에 대한 분포를 박스 그래프로 구현 
sns.boxplot(x='alive', y='age', data=titanic, ax=ax1) 
# 생존여부별 나이값에 대한 분포를 박스 그래프로 구현 
# hue='sex' : 성별로 구분하여 그래프 작성
sns.boxplot(x='alive', y='age', hue='sex', data=titanic, ax=ax2) 
sns.violinplot(x='alive', y='age', data=titanic, ax=ax3) 
sns.violinplot(x='alive', y='age', hue='sex', data=titanic,ax=ax4) 
ax2.legend(loc="upper center")
ax4.legend(loc="upper center")
plt.show()

# 조인트 그래프 - 산점도(기본값), x,y축데이터의 히스토그램
j1 = sns.jointplot(x='fare', y='age', data=titanic) 
j2 = sns.jointplot(x='fare', y='age', kind='reg',data=titanic) 
j3 = sns.jointplot(x='fare', y='age', kind='hex',data=titanic) 
j4 = sns.jointplot(x='fare', y='age', kind='kde',data=titanic) 
j1.fig.suptitle('titanic fare - scatter', size=15)
j2.fig.suptitle('titanic fare - reg', size=15)
j3.fig.suptitle('titanic fare - hex', size=15)
j4.fig.suptitle('titanic fare - kde', size=15)
plt.show()

# FacetGrid : 조건에 따라 그리드 나누기.
# who컬럼 : man,woman, child
#survived : 0(구조안됨),1(구조됨)
titanic.who.value_counts()
titanic.survived.value_counts()
g = sns.FacetGrid(data=titanic, col='who', row='survived') 
g = g.map(plt.hist, 'age') #age컬럼의 히스토그램을 표시.
plt.show()


### pairplot : 각각의 컬럼별 데이터 분포 그리기
### 각 변수들의 산점도 출력,
### 대각선위치의 그래프는 히스토그램으로 표시.
### pairplot 
# titanic 데이터셋 중에서 분석 데이터 선택하기
titanic_pair = titanic[['age','pclass', 'fare']]
print(titanic_pair)
g = sns.pairplot(titanic_pair)


##### 지도 그리기 
##### folium 모듈 사용하기 
#####  pip install folium
import folium
# location=[37.55,126.98] : 지도의 중심 위도,경도
# zoom_start=10 : 지도 확대 값
seoul_map = folium.Map(location=[37.55,126.98],zoom_start=13)
seoul_map.save("seoul.html")

seoul_map2 = folium.Map(location=[37.55,126.98],zoom_start=12,
                        tiles='stamenwatercolor')
seoul_map2.save("seoul2.html")
'''
tiles : 지도 표시되는 형식 설정.
     openstreetmap : 기본값
     cartodbdark_matter
     cartodbpositron
     cartodbpositrononlylabels
     stamentonerbackground
     stamentonerlabels
     stamenterrain, Stamen Terrain
     stamenwatercolor
     stamentoner, Stamen Toner
'''
#파일 등록된 위도 경도를 읽어서 지도에 표시하기.
import pandas as pd
import folium
#index_col=0 : 첫번째 컬럼의 값으로 인덱스로 설정하기 
df = pd.read_excel('서울지역 대학교 위치.xlsx', index_col=0)
df.head()
#지도 생성하기
#folium.Marker : 지도에 마커를 위한 객체
#popup=name : 마커를 클릭하는 경우 표시되는 내용 
#tooltip=name : 마커에 마우스 커서가 들어온 경우 표시되는 내용 
seoul_map = folium.Map(location=[37.55,126.98],zoom_start=12) 
for name, lat, lng in zip(df.index, df.위도, df.경도):
   folium.Marker([lat, lng], popup=name,tooltip=name).add_to(seoul_map)
seoul_map.save('seoul_colleges.html')  

#zip 함수
lista = ['a','b','c']
list1 = [1,2,3]
listh=['가','나','다']
for d in zip(lista,list1,listh) :
    print(d)

for a,n,h in zip(lista,list1,listh) :
    print(a,n,h)

# 원형 마커 출력하기
seoul_map = folium.Map(location=[37.55,126.98],zoom_start=12) 
for name, lat, lng in zip(df.index, df.위도, df.경도):
    folium.CircleMarker([lat, lng],  #위도 경도 : 마커의 위치 
                    radius=10,       #원의 반지름
                    color='brown',   #원 둘레 색상
                    fill=True,       # 원 내부를 채움
                    fill_color='coral', #원 내부 색상
                    fill_opacity=0.7, #투명도. 1:불투명, 0:투명
                    popup=name        #클리시 나타나는 내용
    ).add_to(seoul_map)
seoul_map.save('seoul_colleges2.html')       

### 아이콘 마커 출력
seoul_map = folium.Map(location=[37.55,126.98],zoom_start=12)
for name, lat, lng in zip(df.index, df.위도, df.경도):
#icon = [home,flag,bookmark,star,...]    
   folium.Marker(location = [lat, lng],
           popup=name,
           icon=folium.Icon(color='red',icon='home')
          ).add_to(seoul_map)
seoul_map.save("seoul_colleges3.html")  
 
#Library.csv 파일을 읽어서 도서관 정보를 지도에 표시하기
#1.Library.csv 파일을 읽어서 libray 변수에 저장하기
library = pd.read_csv('Library.csv')
library.head()
library["시설명"].head()

#2.libray 를 이용하여 지도에 도서관 표시하기
seoul_map = folium.Map(location=[37.55,126.98],zoom_start=12)
for name, lat, lng in zip(library.시설명,library.위도, library.경도):
   folium.Marker(location = [lat, lng],
           popup=name,
           tooltip=name,
          ).add_to(seoul_map)
seoul_map.save("seoul_library.html")  


library["시설구분"].unique()
#시설 구분별로 색상 설정하기
seoul_map = folium.Map(location=[37.55,126.98],zoom_start=12)
for name, lat, lng,kbn in zip(library.시설명,library.위도, \
                          library.경도,library.시설구분):
    if kbn == '구립도서관' or kbn=='국립도서관' :
        color = 'green'
    elif kbn=='사립도서관' :        
        color = 'red'
    else :
        color = 'blue'
    folium.Marker(location = [lat, lng],
           popup=name,
           tooltip=kbn,
           icon=folium.Icon(color=color,icon="bookmark")
          ).add_to(seoul_map)
seoul_map.save("seoul_library2.html")  

# MarkerCluster 기능 : 지도 확대 정도에 따른 마커 표시 방법을 달리 설정
from folium.plugins import MarkerCluster
from folium import Marker
seoul_map = folium.Map(location=[37.55,126.98],zoom_start=12)
mc = MarkerCluster()
'''
library.iterrows() : 인덱스와 레코드 한개씩 목록
   _ : 인덱스값. 사용안함. 변수명을_로 설정함. 
       i,a 등등의 변수명으로 사용 가능.
  row : 레코드 값     
'''
for _, row in library.iterrows():
    mc.add_child(    #MarkerCluster 객체에 마커 등록.
        Marker(location = [row['위도'], row['경도']],
               popup=row['시설구분'],
               tooltip=row['시설명']
              )
    )
seoul_map.add_child(mc) #MarkerCluster 를 지도에 추가
seoul_map.save("seoul_library3.html")

#### 경기도 인구 데이터와 위치 정보를 가지고 지도 표시
import pandas as pd
import folium
import json  #json 형태의 파일 처리를 위한 모듈

#1. 경기도 인구 데이터 읽어 df 데이터 저장하기. 단 구분컬럼은 인덱스로 저장하기
file_path = './경기도인구데이터.xlsx'
df=pd.read_excel(file_path,index_col='구분')  
df.head()
df.info()
df.columns #정수값으로 컬럼명이 설정됨.
#columns 데이터의 자료형을 문자열형 목록으로 변경하기
df.columns = df.columns.map(str)
df.columns #문자열형으로 컬럼명이 설정됨.

geo_path = './경기도행정구역경계.json' #텍스트파일 
#geo_data : 경기도의 시군의 경계부분을 좌표로 가지고 있는 데이터.
# json.load : json 파일을 읽기 
try:
    geo_data = json.load(open(geo_path, encoding='utf-8'))
except:
    geo_data = json.load(open(geo_path, encoding='utf-8-sig'))
print(type(geo_data)) #딕셔너리 위도 경도값저장
# 지도로 표시하기 : 인구데이터에 따라 지도에 색상으로 표시
g_map = folium.Map(location=[37.5502,126.982],zoom_start=9)
year = '2007'  
'''
geo_data : 위도 경도 값.
data = df[year] : 지도에 표시할 데이터. 데이값에 따라서 색상 결정  
fill_color='YlOrRd' : 색상표. 파렛트. 데이터 맞는 색상값
  'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 
  'PuRd', 'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd'

fill_opacity=0.7   : 투명도.
line_opacity=0.3   : 경계면 투명도
threshold_scale   : 데이터와 색상 표시할때 범위 지정.
key_on='feature.properties.name' : 데이터와 지도표현을 위한 연결 컬럼 설정  
'''
folium.Choropleth(geo_data=geo_data, 
     data = df[year],  
     columns = [df.index, df[year]],  #지역명, 데이터 
     fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.3,
     threshold_scale=[10000, 100000, 300000, 500000, 700000],               
     key_on='feature.properties.name',
   ).add_to(g_map)
g_map.save('./gyonggi_population_' + year + '.html')

# us-states.json, US_Unemployment_Oct2012.csv
# 사용하여 지도로 표시하기
import folium
import pandas as pd
state_geo = "us-states.json"
state_unemployment = "US_Unemployment_Oct2012.csv"
state_data = pd.read_csv(state_unemployment)
m = folium.Map(location=[48, -102], 
               zoom_start=3, tiles="Stamen Toner")
folium.Choropleth(
    state_geo,  #위도 경도를 위한 지도파일(json형태 )의 이름 
    data=state_data,    #데이터 
    columns=["State", "Unemployment"],  #컬럼명 설정 
    key_on="feature.id",  #데이터와 지도표시를 연결 
    fill_color="YlGn",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Unemployment Rate (%)",
).add_to(m)
m.save('usa1.html')

##########################################
# numpy : 행렬, 통계관련 기본 함수, 다차원배열 객체 제공

import numpy as np
#배열 생성
a = np.arange(15).reshape(3,5) #1차원 배열을 3행5열 2차원 배열로 재편성 생성
a
b= np.arange(15) #0~ 14까지의 숫자를 1차원 배열 리턴
b
b= b.reshape(3,5) # b배열을 3행5열로 재편성 
b
type(b)  #배열 
type(np.arange(15)) #배열 

#배열의 형태
a.shape  # (3,5)             #2차원배열
np.arange(15).shape # (15,)  #1차원배열
#배열의 차수 조회 
a.ndim                       #2차원
np.arange(15).ndim           #1차원 

#배열의 요소의 자료형.
a.dtype        #요소의 자료형 
a.dtype.name   #요소의 자료형 문자열로 리턴 
np.arange(15).dtype

#배열 요소의 크기
a.itemsize   #int32 자료형의 바이트 크기 

#배열 요소의 갯수 
a.size
#실제 저장된 영역값 
a.data


# 리스트로 부터 배열 생성하기
b = np.array([6,7,8])
b
type(b)
b.ndim
b = np.array(6,7,8) #오류 발생

b = np.array((6,7,8)) #튜플로 부터 배열 생성
b

#2차원배열 생성
c=np.array([[1.5,2,3],[4,5,6]])
c
c.shape  #배열 형태
c.ndim   #배열의 차원 
c.dtype  #배열 요소의 자료형
c.dtype.name #배열 요소의 자료형이름
c.itemsize #배열요소의 바이트 크기

#0으로 초기화된 3행 4열 배열 d 를 생성하기
d = np.array([(0,0,0,0),(0,0,0,0),(0,0,0,0)])
d.shape

e = np.zeros((3,4)) # 3행4열 배열을 생성. 0으로 채움
e.shape

#100의 0으로 이루어진 배열 생성하기
f = np.zeros(100)
f.shape

#0 ~ 9999 까지의 값을 가진 배열을 100행 100열의 2차원배열  g로 생성하기
g = np.arange(10000).reshape(100,100)
g.shape

#100개의 0으로 채워진 배열을 10행 10열의 배열 h로 생성하기
h=np.zeros((10,10))
h.shape
h

#100개의 1로 채워진 배열을 10행 10열의 배열 i로 생성하기
i=np.ones((10,10))
i.shape
i

#0 부터 2까지의 숫자를 9개로 균등 분할하여 배열 생성하기
j = np.linspace(0,2,9)
j

# 원주율
np.pi
#0부터 원주율 까지 균등하게 10개의 배열로 생성하기
k = np.linspace(0,np.pi,10)
k

# np.sin : 삼각함수.
#          라디안단위로 값을 구함.
#          np.sin(90)  (X)
#          np.sin(np.pi/2)  (X)
m = np.sin(k)
m
np.sin(np.pi/2) #sin(90도)
np.sin(np.pi/4) #sin(45도)
# 데이터 연산
a = np.array([20,30,40,50])
b = np.arange(4)
c=a-b #배열간 연산. 
c
c = b**2  #배열 상수간 연산 
c
c = a < 35
c
a+b

#2차원 배열의 연산 
a= np.array([[1,1],[0,1]])
b= np.array([[2,0],[3,4]])

a*b  #[1,1] * [2,0] = [2,0]
     #[0,1] * [3,4] = [0,4]

a+b  #[1,1] + [2,0] = [3,1]
     #[0,1] + [3,4] = [3,5]

#행렬의 곱
#       a      b
a@b  #[1,1]  [2,0] = [1*2+1*3][1*0+1*4] = [5,4]
     #[0,1]  [3,4] = [0*2+1*3][0*0+1*4] = [3,4]

a.dot(b) # == a@b 같은 연산

a*2

#난수값을 이용하여 배열 생성하기
rg = np.random.default_rng(1)
rg

a=rg.random((2,3))  #실수형 요소
a

b=np.ones((2,3),dtype=int)  #정수형 요소로 생성
b
b.dtype

a += b # 실수형 = 실수형 + 정수형
a

b += a # 정수형 = 정수형 + 실수형

c = b + a
c

a
b
# a 배열의 전체 요소의 합
a.sum()
b.sum()
# a 배열의  요소 중 최소값
a.min()
# a 배열의  요소 중 최대값
a.max()
# a 배열의  요소 의 평균
a.mean()
# a 배열의  요소 의 중간값
a.median() #오류
# a 배열의  요소 의 표준편차
a.std()
a
#a 행렬의 행 중 최소값
a.min(axis=1)
#a 행렬의 열 중 최소값
a.min(axis=0)
#a 행렬의 행 중 최대값
a.max(axis=1)
#a 행렬의 열 중 최대값
a.max(axis=0)
#a 행렬의 행의 누적값
a.cumsum(axis=1)
a.sum(axis=1)
#a 행렬의 열의 누적값
a.cumsum(axis=0)
a.sum(axis=0)

'''
 중간값 : 
  요소의 갯수가 홀수 : 정렬 후 가장 가운데 값
  요소의 갯수가 짝수 : 정렬 후 가장 가운데 2개의 값의 평균
'''
