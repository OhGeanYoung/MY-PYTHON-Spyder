# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 09:19:33 2021

@author: KITCOOP
20211229.py
"""
#막대 그래프 출력하기
import matplotlib.pyplot as plt
subject=["Oracle","Python","SKlearn","Tensorflow"] #x축의 값
score=[65,90,85,95]                                #y축의 값
plt.style.use("ggplot")
fig=plt.figure()                                   # 그래프 작성 창
ax1=fig.add_subplot(1,1,1)                         # 1행1열 1번째
#bar(x축,y축,...) : 막대그래프 작성 함수
ax1.bar(range(len(subject)),score,align="center",color="darkblue")
plt.xticks(range(len(subject)),subject,rotation=0,fontsize="small")
plt.xlabel("Subject")
plt.ylabel("Score")
plt.title("Class Score")
#plt 에서 작성된 그래프 를 score.png 파일로 저장하기
plt.savefig("score.png",dpi=400,bbox_inches="tight")

#남북한발전전력량.xlsx 데이터를 이용하여 연합막대 그래프 그리기
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#남북한발전전력량.xlsx 파일을 pandas로 읽기
df = pd.read_excel("남북한발전전력량.xlsx")
df.info()
#북한지역의 발전 전력량만 조회 df 데이터에 저장.
df = df.loc[5:]
df
# 전력량 (억㎾h) 컬럼 삭제
#df.drop("전력량 (억㎾h)",axis=1,inplace=True)
#df.drop("전력량 (억㎾h)",axis="columns",inplace=True)
del df["전력량 (억㎾h)"]
df
#발전 전력별
df.set_index("발전 전력별",inplace=True)
df
#전치행렬로 변경
df = df.T
df
#합계 컬럼을 "총발전량" 컬럼명으로 변경하기
df = df.rename(columns={"합계":"총발전량"})
df
# 총발전량 - 1년 컬럼 추가하기 
#df["총발전량"].shift(1) : 총발전량 컬럼의 앞 레코드의 데이터
df["총발전량 - 1년"] = df["총발전량"].shift(1)
df

#증감율 컬럼 추가 : ((총발전량/총발전량 - 1년 ) -1 ) * 100
'''
   (현재-전년도)/전년도 * 100
   (현재/전년도 - 1 ) * 100
'''
df['증감율'] = ((df['총발전량'] / df['총발전량 - 1년']) - 1) * 100
df
from matplotlib import  rc
rc('font', family="Malgun Gothic")       #한글폰트 설정
plt.rcParams['axes.unicode_minus']=False #음수표시.
#stacked=True : 수력,화력 데이터를 하나의 막대그래프로 표시
ax1 = df[['수력','화력']].plot(kind='bar', figsize=(20, 10), width=0.7, stacked=False)  
ax2 = ax1.twinx()  #ax1 그래프영역을 복사 ax2 그래프영역으로 사용. 
'''
df.index : x축의값. 년도
df.증감율 : y축의값. 선그래프의 값
ls='--'/'-' : 선의 종류. 점선/실선
'''
ax2.plot(df.index, df.증감율, ls='-', marker='o', markersize=20, 
        color='green', label='전년대비 증감율(%)')  
ax1.set_ylim(0, 500)
ax2.set_ylim(-50, 50)
ax1.set_xlabel('연도', size=20)
ax1.set_ylabel('발전량(억 KWh)')
ax2.set_ylabel('전년 대비 증감율(%)')
plt.title('북한 전력 발전량 (1990 ~ 2016)', size=30)
ax1.legend(loc='upper left')   #범례 위치 지정 
ax2.legend(loc='upper right')
plt.savefig("북한전력량.png", dpi=400, bbox_inches="tight")
plt.show()

#### 히스토그램 그리기 : 값의 구간별 건수/밀도 출력. 값의 분포를 알수 있다
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot') #classic

df = sns.load_dataset("mpg") #자동차 연비 데이터 읽기
df.info()
# df["mpg"] 데이터의 값의 분포 그래프로 작성 
# bins=20 : 값의 구간 20개 설정 
df["mpg"].plot(kind="hist",bins=20,color='coral',figsize=(10,5))
plt.title('Histogram')
plt.xlabel('mpg')
plt.show()

'''
 DataFrame.plot() 함수를 이용하여 그래프 작성
    기본 : 선그래프
    kind="bar" : 막대그래프
    kind="barh": 수평막대그래프
    kind="hist": 히스토그램
    kind="scatter": 산점도
    kind="box": 박스그래프
'''
# matplot 함수를 이용하여 히스토그램 작성
# bins : 값의 구간 지정. 기본:10
plt.hist(df["mpg"],bins=20)

# df 데이터의 x:weight, y:mpg 데이터를 이용하여 산점도 그리기.
#dataframe.plot() 함수
df.plot(kind='scatter', x='weight', y='mpg',  c='coral', s=20,figsize=(10, 5))
# matplot 모듈을 이용 
plt.figure(figsize=(10, 5))
plt.scatter(df["weight"], df["mpg"],c='coral', s=20 )

# bubble 그래프 : 산점도.
# 점의 크기의 데이터의 값으로 결정하기.
# 3개의 변수 지정됨 : x축(weight),y축(mpg),점의 크기(cylinders)
#cylinders 값을 최대값의 비율로 계산하여 데이터 생성하기
df["cylinders"].value_counts()
cylinders_size = df.cylinders / df.cylinders.max() * 300
cylinders_size.value_counts()
#alpha=0.7 : 투명도 
df.plot(kind='scatter', x='weight', y='mpg',  c='coral',\
        s=cylinders_size,figsize=(10, 5),alpha=0.7)
plt.title("Scatter plot:mpg-weight-cylinders")    
plt.show()

#색상으로 데이터값을 설정하기
# marker='+' : 산점도에 표시되는 모양을 설정. 
# cmap='viridis' : 맵플롯에서 색상모임. 값의 따른 색상의 목록
#                  'viridis','plasma', 'inferno', 'magma', 'cividis'  
# c=cylinders_size : 색상.

df.plot(kind='scatter', x='weight', y='mpg', marker='+',\
        figsize=(10, 5), cmap='plasma', c=df["cylinders"], \
        s=50, alpha=0.7)
plt.title('Scatter Plot: mpg-weight-cylinders')
plt.savefig("scatter_transparent.png",transparent=True) #투명그림으로 저장
plt.show()

#파이그래프
#origin 컬럼의 비율을 파이그래프로 출력하기
#origin 컬럼 값의 내용을 출력하기
df_origin = df["origin"].value_counts()
df_origin
type(df_origin)
'''
autopct="%.1f%%" : 비율표시
   %.1f : 소숫점이하 1한자리 표시
   %%   : % 문자의미
startangle=10 : 파이조각의 시작 각도    
'''
df_origin.plot(kind="pie",figsize=(7,5),autopct="%.1f%%",startangle=10,
               colors=['chocolate','bisque','cadetblue'])
plt.title("Model Origin",size=20)
#plt.axis("equal") 
#plt.axis("auto") 
plt.axis("square") 
plt.legend(labels=df_origin.index,loc="upper right")
plt.show()

#박스 그래프 : 
from matplotlib import  rc
rc('font', family="Malgun Gothic") 

fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
'''
   origin 컬럼으로 그룹화 하여 연비 데이터를 그래프 출력 
   
   vert=False : 수평박스그래프로 출력
'''
ax1.boxplot(x=[df[df['origin']=='usa']['mpg'],
               df[df['origin']=='japan']['mpg'],
               df[df['origin']=='europe']['mpg']], 
         labels=['USA', 'JAPAN','EU'])
ax2.boxplot(x=[df[df['origin']=='usa']['mpg'],
               df[df['origin']=='japan']['mpg'],
               df[df['origin']=='europe']['mpg']], 
         labels=['USA', 'JAPAN','EU'],vert=False)
ax1.set_title('제조국가별 연비 분포(수직 박스 플롯)')
ax2.set_title('제조국가별 연비 분포(수평 박스 플롯)')
plt.show()

df[df['origin']=='japan']['mpg']

# origin 컬럼을 그룹화 하기
df_origin = df.groupby("origin").sum()
df_origin
df_origin = df.groupby("origin").count()
df_origin
df.origin.value_counts()

# seaborn 모듈을 이용한 시각화. matplot 모듈의 확장판.
#    시각화도구 + 데이터셋
#titanic 데이터 로드하기
import seaborn as sns
titanic = sns.load_dataset("titanic")

titanic.info()
sns.set_style('darkgrid')  #style 설정

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 2, 1) 
ax2 = fig.add_subplot(1, 2, 2)
# regplot : 선형회귀그래프 : 산점도 + 선형회귀선 표시
# 회귀선 : 모든 점에서 가장 가까운 점들을 선으로 표시.     
# x='age' : x축의 값 
# y='fare' : y축의 값 
# fit_reg=False : 회귀선 표시 안함
sns.regplot(x='age',  y='fare', data=titanic, ax=ax1)  
sns.regplot(x='age', y='fare',  data=titanic, ax=ax2, fit_reg=False)  
plt.show()

### 히스토그램 작성하기
fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
#distplot() : kde(밀도),hist(빈도수)
#kdeplot() : kde(밀도)
#histplot() :hist(빈도수)
#sns.distplot(titanic['fare'],ax=ax1) # kde(밀도),hist(빈도수) 출력
#sns.distplot(titanic['fare'],hist=False, ax=ax1) # kde(밀도))출력
sns.distplot(titanic['fare'],kde=False, ax=ax1) # hist(빈도수) 출력
sns.kdeplot(x='fare',data=titanic, ax=ax2) 
sns.histplot(x='fare',data=titanic,ax=ax3)  
ax1.set_title('titanic fare - distplot')
ax2.set_title('titanic fare - kdeplot')
ax3.set_title('titanic fare - histplot')
plt.show()

#matplot 이용
titanic['fare'].plot(kind='hist', bins=50, color='coral', figsize=(10, 10))
plt.title('Histogram')
plt.xlabel('fare')
plt.show()

### 히트맵 : 범주형 데이터의 수치를 색상과 값으로 표시
# pivot_table : 2개의 범주형데이터를 행,열로 재구분 
# aggfunc='size' : 데이터의 갯수 
table = titanic.pivot_table(index=['sex'], columns=['class'], aggfunc='size')
table
# 성별 인원수 조회하기
titanic["sex"].value_counts()
titanic.groupby("sex").count()["survived"]
'''
table : 표시할 데이터
annot=True : 데이터값 표시
fmt='d' : 정수형 으로 데이터 출력 
cmap='YlGnBu' : 컬러맵. 색상표.
cbar=False : 컬러바 표시안함 
'''
sns.heatmap(table,annot=True,fmt='d',
            cmap='YlGnBu',linewidth=.5,cbar=True)
plt.show()    
    
### 막대 그래프
fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
titanic.groupby("sex").survived.value_counts()

# 막대그래프 :
# data=titanic : 데이터프레임 객체.     
sns.barplot(x='sex', y='survived', data=titanic, ax=ax1) 
# hue='class' : titanic['class'] 별로 분리 
sns.barplot(x='sex', y='survived', hue='class', data=titanic, ax=ax2)
#dodge=False : 누적 표시 
sns.barplot(x='sex', y='survived', hue='class', dodge=False, data=titanic, ax=ax3)   
ax1.set_title('titanic survived - sex')
ax2.set_title('titanic survived - sex/class')
ax3.set_title('titanic survived - sex/class(stacked)')
plt.show()    
    
### countplot : 막대그래프로 건수 출력
fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
print(titanic["who"].head())
titanic["who"].unique()
titanic["class"].value_counts()
#palette='Set1' : 설정된 색상표. 
sns.countplot(x='class',palette='Set1', data=titanic, ax=ax1) 
#hue='who' : 그룹화 
sns.countplot(x='class', hue='who', palette='Set2', data=titanic, ax=ax2) 
# class별, who별로 count값 출력하기
table = titanic.pivot_table\
    (index=['class'], columns=['who'], aggfunc='size')
table
#dodge=False : 누적데이터로 출력
sns.countplot(x='class', hue='who', palette='Set3',dodge=False, data=titanic, ax=ax3)       
ax1.set_title('titanic class')
ax2.set_title('titanic class - who')
ax3.set_title('titanic class - who(stacked)')
plt.show()


# 산점도 
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
# 이산형 변수의 분포 - 데이터 분산 미고려
sns.stripplot(x="class", y="age",data=titanic, ax=ax1)       
# 이산형 변수의 분포 - 데이터 분산 고려
sns.swarmplot(x="class", y="age",data=titanic, ax=ax2)     
ax1.set_title('Strip Plot')
ax2.set_title('Swarm Plot')
plt.show()

# 산점도 : 성별을 색상으로 표시
sns.set_style('whitegrid')
fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
# 이산형 변수의 분포 - 데이터 분산 미고려
sns.stripplot(x="class", y="age",data=titanic,hue="sex", ax=ax1)       
# 이산형 변수의 분포 - 데이터 분산 고려
sns.swarmplot(x="class", y="age",data=titanic,hue="sex", ax=ax2)     
ax1.set_title('Strip Plot')
ax2.set_title('Swarm Plot')
plt.show()
    