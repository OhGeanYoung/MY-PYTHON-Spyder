# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 09:24:52 2021

@author: KITCOOP
20211228.py
"""
#판다스를 이용하여 excel 파일 읽기

import pandas as pd
infile = "sales_2015.xlsx"
#read_excel(파일명,sheet이름,인덱스컬럼)
# sheet_name=None : 모든 sheet 읽기
# index_col=None : 인덱스 컬럼 없음
df = pd.read_excel(infile,sheet_name=None,index_col=None)
print(df)
print(type(df)) #딕셔너리
#name : sheet 이름
#data : DataFrame객체
for name,data in df.items() :
    print("sheet_name:",name)
    print(type(data))
    #Sale Amount 컬럼의 값이 500보다 큰 레코드 조회 
    #data["Sale Amount"].replace("$","") : dataframe의 Sale Amount 컬럼의 값을
    # $를빈문자열로 변경.
    # replace(",","") : ,를 빈문자열로 변경 
    # astype(float) : dataframe의 Sale Amount 컬럼의 값을 실수형으로 변환.
    #                 현변환 함수.
#    print(data[data["Sale Amount"].replace("$","").\
#         replace(",","").astype(float) > 500.0])
    print(data[data["Sale Amount"] > 500.0])
#data[조건값] : 조건값이 True인 레코드만 조회
data[data["Sale Amount"].replace("$","").replace(",","").astype(float) > 500.0]
        
#하나의 sheet만 읽기
df = pd.read_excel(infile,"january_2015",index_col=None)
print(df)
print(type(df)) #dataFrame객체
#Sale Amount 컬럼의 값이 500보다 큰 레코드만 pd_sale_2015.xlsx파일의 jan_2015_500 sheet로 저장하기
#df 데이터 중 Sale Amount 컬럼의 값이 500보다 큰 레코드만 df500에 저장
df500 =  df[df["Sale Amount"] > 500]
df500
df
#df500 데이터를 pd_sale_2015.xlsx파일의 jan_2015_500 sheet로 저장하기
#ExcelWriter(파일명) : 저장할 excel 파일데이터 
outexcel = pd.ExcelWriter("pd_sale_2015.xlsx")
df500.to_excel(outexcel,sheet_name="jan_2015_500",index=False) #sheet 설정
outexcel.save()  #파일 생성
# dict_data 데이터를 이용하여 데이터프레임객체 df 생성하기
# 단  index 이름은 r0,r1,r2로 설정
import pandas as pd
dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], \
             'c3':[10,11,12], 'c4':[13,14,15]}
#df = pd.DataFrame(dict_data, index=['r0', 'r1', 'r2'])
df=pd.DataFrame(dict_data)
df.index=['r0','r1','r2'] #행의 갯수와 인덱스의 갯수가 같아야 함
print(df)    
#reindex() : 인덱스를 5개로 설정하기. 행이 추가됨. 
ndf = df.reindex(['r0','r1','r2','r3','r4'])
print(ndf) #NaN : 결측값 (값이 없음)
# df 데이터를 행을 5개로 생성하여 ndf2 데이터 저장하기.
# 결측값은 0으로 채우기 
ndf2 = df.reindex(['r0','r1','r2','r3','r4'],fill_value=0)
print(ndf2)
#정렬하기
#sort_index : 인덱스이름으로 정렬
#sort_values : 기준 컬럼을 설정하여 컬럼값으로 정렬
# 인덱스 기준 내림차순으로 정렬하기
print(df.sort_index(ascending=False))
# c1 컬럼 기준 내림차순으로 정렬하기
print(df.sort_values(by='c1',ascending=False))
####
import pandas as pd
exam_data = {'이름' : [ '서준', '우현', '인아'],
             '수학' : [ 90, 80, 70],
             '영어' : [ 98, 89, 95],
             '음악' : [ 85, 95, 100],
             '체육' : [ 100, 90, 90]}
# dataframe 객체 df로 생성하기
df = pd.DataFrame(exam_data)
print(df)
# 이름 컬럼을 인덱스로 변경하기
df.set_index("이름",inplace=True)
print(df)

# 이름의 역순으로 출력하기
#df 객체 자체가 변경안됨. inplace=True면 자체객체가 변경됨
df.sort_index(ascending=False) 
df

# 수학점수 순으로 출력하기
df.sort_values(by="수학",ascending=True) 
# 수학점수 내림차순으로 출력하기
df.sort_values(by="수학",ascending=False)

#titanic 데이터 셋으로 연습하기
import pandas as pd
'''
ImportError: DLL load failed while importing _arpack: 지정된 프로시저를 찾을 수 없습니다.
1. anaconda Prompt cmd 창 로드
2. pip uninstall scipy
3. pip install scipy
4. pip uninstall seaborn
5. pip install seaborn
'''

import seaborn as sns
print(sns.get_dataset_names()) #seaborn모듈에 저장된 데이터셋 목록
titanic = sns.load_dataset("titanic")# titanic 데이터 로드
print(titanic.head())
print(titanic.info())
'''
survived	생존여부
pclass	좌석등급 (숫자)
sex	성별
age	나이
sibsp	형제자매 + 배우자 인원수
parch: 	부모 + 자식 인원수
fare: 	요금
embarked	탑승 항구
class	좌석등급 (영문)
who	성별
adult_male 성인남자여부 
deck	선실 고유 번호 가장 앞자리 알파벳
embark_town	탑승 항구 (영문)
alive	생존여부 (영문)
alone	혼자인지 여부
'''
print(titanic["adult_male"].head())
print(type(titanic))
# 컬럼명만 출력하기
print(titanic.columns)
#titanic의 age,fare 컬럼만을 df 데이터셋에 저장하기
df = titanic[["age","fare"]]
df = titanic.loc[:,["age","fare"]]
print(df.info())
'''
   빅데이터 특징
   3V
   1. Volume : 데이터의 양이 대용량.
   2. Velocity : 데이터의 처리 속도가 빨라야 한다. 
   3. Variety : 데이터의 다양성.
        정형데이터 : dbms
        반정형데이터 : JSON,XML,
        비정형 : 이미지
'''
#df 데이터의 레코드 건수 조회
df.count() # 결측데이터는 제외.
#df 데이터의 age컬럼의 레코드 건수 조회
df.age.count()
df["age"].count()
df.count()["age"]
df.mean()#df 데이터 평균 조회
df.median()#df 데이터 중간값 조회
df.age.value_counts()#나이별로 인원수 조회
df.age.max()#최고령자 나이 조회
#최고령자 정보조회
df[df["age"]==df.age.max()]
titanic[df["age"]==df.age.max()]

# seaborn 데이터에서 mpg 데이터 로드하여 mpg에 저장하기
mpg = sns.load_dataset("mpg")  #자동차 연비 관련 데이터 
mpg.info()
mpg.head()
#mpg 행열의 갯수
mpg.shape  #398행 9열 데이터.
mpg.dtypes #각 컬럼의 자료형
# mpg 데이터의 mpg 컬럼의 자료형 출력하기
mpg.mpg.dtypes
#mpg 데이터의 기술통계 정보 조회하기
mpg.describe() #숫자값 조회
mpg.describe(include="all") #모든 데이터  조회

mpg.info()
# origin컬럼의 값에 따른 건수를 조회하기
mpg.origin.value_counts()
# origin컬럼의 값조회하기
list(mpg.origin.unique())
# mpg 데이터 중 mpg,weight 컬럼의 최대값 출력하기
mpg[["mpg","weight"]].max()
# mpg 데이터 중 mpg,weight 컬럼의 기술통계 출력하기
mpg[["mpg","weight"]].describe()

#상관계수 : 컬럼(변수)간의 관계를 수치로 표시함.
mpg.corr()
mpg[["mpg","weight"]].corr()

# 남북한발전전력량.xlsx 파일을 읽어 df에 저장하기
df = pd.read_excel("남북한발전전력량.xlsx")
df.head()
df
#0행 5행, 2열이후의 정보만 ndf 데이터 저장하기
ndf = df.iloc[[0,5],2:]
ndf
ndf.columns
#선그래프로 출력하기 : 컬럼별로 그래프의 선이 출력됨. 
ndf.plot()
# 전치 행렬 : 행과 열을 바꾸기
ndf2 = ndf.T
ndf2
#ndf2의 컬럼명을 "South","North"로 변경하기
ndf2.columns=["South","North"]
ndf2
ndf2.plot() #선그래프
ndf2.plot(kind="bar") #막대그래프
ndf2.plot(kind="hist") #히스토그램그래프. 데이터의 범위별 빈도수를 그래프 
#산점도 그래프. 값의 범위.값의 분포 
mpg.plot(x="mpg", y="weight",kind="scatter")

#mpg 데이터 중 "mpg"컬럼의 히스토그램 출력하기
mpg.mpg.plot(kind="hist")

#matplotlib : 시각화를 위한 기본적인 모듈 
import matplotlib.pyplot as plt
plt.plot(ndf2.South)


### 시도별 전출입 인구수 분석하기 
df = pd.read_excel("시도별 전출입 인구수.xlsx")
df.info()
df.head()

# 결측값 처리 : 앞데이터로 채움 
# fillna : 결측값을 다른값으로 치환
df = df.fillna(method="ffill") #앞데이터로 채움
df.head()

#전출지가 서울에서 다른 지역으로 이동한 데이터만 추출하기
#전출지가 서울이고, 전입지가 서울이 아닌데이터만 추출하기.
mask = (df["전출지별"] == '서울특별시') & (df["전입지별"] != '서울특별시')
print(mask)
print(mask.value_counts())

df_seoul = df[mask]
df_seoul

#컬럼명이 전입지별=>전입지 설정하기
df_seoul.rename(columns={'전입지별':'전입지'},inplace=True)
df_seoul

#df_seoul의 전출지별 컬럼을 제거하기 
df_seoul = df_seoul.drop('전출지별',axis=1)
df_seoul

#df_seoul 데이터에서 전입지 컬럼을 인덱스로 설정하기
df_seoul.set_index("전입지",inplace=True)
df_seoul

# 경기도로 이동 데이터만 sr1 값에 저장하기
sr1 = df_seoul.loc["경기도"]
sr1
sr1.index
#sr1 데이터 시각화 하기
import matplotlib.pyplot as plt
# 기본설정된 폰트가 한글 출력 안됨. 한글이 가능한 폰트를 설정해야 함 
plt.rc("font", family="Malgun Gothic") #맑은 고딕 폰트로 변경
#plt.plot(sr1.index,sr1.values) #x축,y축 데이터
plt.plot(sr1) #x축:index값 ,y축 데이터: values 값
plt.title("서울=>경기 인구 이동")
plt.xlabel("년도")
plt.ylabel("이동인구수")
plt.xticks(rotation='vertical') #x축 값을 수직방향으로 설정 
#스타일 정보 
print(plt.style.available)
#그래프의 스타일 지정하기
plt.style.use('ggplot') #ggplot 스타일로 설정
plt.figure(figsize=(14,5)) #그래프 작성 창을 가로:14,세로:5크기로 설정 
plt.xticks(rotation='vertical',size=10) #x축의 값을 세로로. 글자크기:10
plt.plot(sr1,marker='o',markersize=10) #marker :선그래프에 표시할 점. 점크기 설정
plt.title("서울=>경기 인구 이동")
plt.xlabel("년도")
plt.ylabel("이동인구수")
plt.legend(labels=['서울->경기'],loc='best') #범례. 범례위치는 적당한곳

### 그래프에 설명(주석) 추가하기
plt.rc('font', family="Malgun Gothic") #한글폰트
plt.style.use('ggplot')                #스타일 설정
plt.figure(figsize=(14,5))             #그래프 창 크기.
plt.xticks(rotation='vertical',size=10) #x축 데이터를 수직방향. 글자 크기 설정
plt.plot(sr1,marker='o',markersize=10)  #선그래프의 각 값에 o 표시.
plt.title("서울=>경기 인구 이동")
plt.xlabel("년도")
plt.ylabel("이동인구수")
plt.legend(labels=['서울->경기'],loc='best')
plt.ylim(50000, 800000)   #y축의 범위 설정
#annotate() : 그래프에 주석 표시 
#화살표 표시.
plt.annotate('',   
             xy=(20, 620000),    #화살표의 머리부분. (끝점) 
             xytext=(2, 290000), #화살표의 꼬리부분. (시작점) 
             xycoords='data',    #좌표 체계. 
             arrowprops=dict(arrowstyle='->', color='skyblue', lw=5) ) #화살표 설정
plt.annotate('', 
             xy=(47, 450000),   #화살표의 머리부분. (끝점) 
             xytext=(30, 580000), #화살표의 꼬리부분. (시작점) 
             xycoords='data',  
             arrowprops=dict(arrowstyle='->', color='olive', lw=5))
#내용출력
plt.annotate('인구이동 증가(1970-1995)', 
             xy=(10, 450000), 
             rotation=22, #출력방향 22도정도로 회전
             va='baseline', #텍스트 상하 정렬
             ha='center',   #텍스트 좌우 정렬
             fontsize=15 )
plt.annotate('인구이동 감소(1995-2017)', xy=(40, 560000),  rotation=-11,               
             va='baseline',  ha='center',   fontsize=15 )

#한번에 2개의 그래프 작성하기
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,1,1) #2,1,1 :2행1열 중 첫번째 위치 
ax2 = fig.add_subplot(2,1,2) #2,1,2 :2행1열 중 두번째 위치
ax1.plot(sr1,'o',markersize=10) #sr1:데이터 'o' : o표현. 
#marker='o' : 선그래프위에 마컷표시
# markerfacecolor='green' : 마커 색상
# color="olive" : 선의 색상
# label='서울->경기' : legend에 출력될 데이터 
ax2.plot(sr1,marker='o',markersize=10,\
         markerfacecolor='green',color='olive',\
             linewidth=2,label='서울->경기')
ax2.legend(loc='best')
ax1.set_ylim(50000,800000)  #y축의 범위 지정 
ax2.set_ylim(50000,800000)
ax1.set_xticklabels(sr1.index,rotation=75) #75도정도 회전
ax2.set_xticklabels(sr1.index,rotation=75)
    