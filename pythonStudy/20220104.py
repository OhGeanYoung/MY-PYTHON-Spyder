# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 08:34:24 2022

@author: KITCOOP
20220104.py
"""
import numpy as np
#numpy의 버전
np.__version__
import pandas as pd
pd.__version__

# 10부터 49까지의 숫자를 배열 a 생성하기
a = np.arange(10,50)
a
# 10개의 요소를 가지고 값이 1로채워진 배열 b 생성하기
b=np.ones(10)
b
# 100개의 요소를 가지고 값이 0로채워진 배열 c 생성하기
c=np.zeros(100)
c
#c배열의 요소의 크기 출력하기
c.size
#c배열의 요소의 자료형 출력하기
c.dtype
#c배열의 요소의 자료형 바이트크기 출력하기
c.itemsize

#c배열의 5번째 값을 1로 변경하기
c[4]=1
c

#0부터 8까지 값을 가진 3행 3열 배열 d 생성하기
d=np.arange(0,9).reshape(3,3)
d
d.ndim  #배열의 차수
d.shape

#0부터 11까지 값을 가진 3행 4열 배열 e 생성하기
e=np.arange(0,12).reshape(3,4)
e
e.shape
e[1,1] #1행1열의 값 
e[:2,:2] #0행부터 1행,0열 1열까지 까지의 값
e[::2,::2] #0행부터 2씩,0열 2씩 값
e[::3,::3] #0행부터 3씩,0열 3씩 값

# 1값으로 채워지 10행 10열 배열 생성하기
f = np.ones((10,10))
f
#f배열을 가장자리는 1로 내부는 0으로 채워진 배열 수정하기 
f[1:-1,1:-1] = 0
f
f[1:9,1:9] = 0
f

# 0값으로 채워지 8행 8열 배열 생성하기
f = np.zeros((8,8))
f
#pad 함수 :행과 열에  값을 추가.
#  pad_width=1 : 가장자리 1줄을 
#  constant_values=1 값으로 추가하기
f = np.pad(f,pad_width=1,constant_values=1)
f

#np.fromfunction : 함수를 이용하여 요소의 값설정하기
def f(x,y) : # x:행인덱스, y:열인덱스
    return 10*x+y
g=np.fromfunction(f,(5,4),dtype=int)
g
#g 배열의 0행 출력하기
g[0]
#g 배열의 2행 출력하기
g[2]
#g 배열의 0열 출력하기
g[:,0]
#g 배열의 0열에서1열까지 출력하기
g[:,0:2]


#g.flat : 반복문에서 배열의 요소들만 리턴
for e in g.flat :
    print(e)

#난수를 이용한 배열 생성 : 0~9사이의 정수값을 가진 난수로
# 3행 4열의 배열을 생성하기
# np.floor : 작은 근사정수 
# np.ceil  : 큰 근사정수 
h = np.floor(10*np.random.random((3,4)))
h
#h배열을 1차원배열 h1으로 변경하기
h1=h.ravel() #h배열이 변경되지 않음. 결과만 리턴 
h1
h
h.shape
h1.shape
#h배열을 6행2열로 h2 변경하기
h2=h.reshape(6,2) #h배열이 변경되지 않음. 결과만 리턴 
h2
h2.shape
h.shape

h.resize(6,2)  #h배열이 변경됨. 
h.shape
# 요소의 갯수가 변경되는 배열의 갯수가 맞아야함
h.reshape(3,5) #오류 발생
#행을 지정하고 열의 값에 -1 지정하면, 열을 알아서 처리함.
h.reshape(3,-1)
h.reshape(4,-1)
#행을 -1 지정하고 열의 값을 설정하면, 행을 알아서 처리함.
h.reshape(-1,4)
h.shape

#배열 합하기
#i 배열 : 0~9 정수값을 난수요소를 가진 2행2열배열 생성하기
#randint : 정수형 난수리턴.10:0~9,size=(행,열):
i = np.random.randint(10,size=(2,2))
i
#j 배열 : 0~9 정수값을 난수요소를 가진 2행2열배열 생성하기
j = np.random.randint(10,size=(2,2))
j

np.vstack((i,j)) #행을 합하기. 열의 갯수가 맞아야 함 
np.hstack((i,j)) #열을 합하기. 행의 갯수가 맞아야 함


# 배열 분리하기 
k = np.random.randint(10,size=(2,12))
k
np.hsplit(k,3)
np.vsplit(k,2)
k

#k 배열의 모든 요소값을 100으로 변경하기
# k=100 #k변수에 100값을 저장 
k[:]=100
k

#0부터 19까지의 수의 sin 값을 가진  5행4열의 배열 l 생성하기
l = np.sin(np.arange(0,20)).reshape((5,4))
l
#각 행 중 최대값을 가진 요소들 출력하기
l.max(axis=1)
#각 열 중 최대값을 가진 요소들 출력하기
l.max(axis=0)

#각 행 중 최대값을 가진 요소의 인덱스들 출력하기
l.argmax(axis=1)
#각 열 중 최대값을 가진 요소의 인덱스들 출력하기
l.argmax(axis=0)

#단위 행렬 생성
m=np.eye(10,10)
m
#요소의 값이 0이 아닌 배열의 인덱스 출력하기
np.nonzero(m)
n=[1,2,0,4,0]
np.nonzero(n)

#정규 분포값을 가진 난수로 배열 o 생성하기
o = np.random.normal(0,1,10000) #평균:0,표준편차:1 난수 10000개 
o
import matplotlib.pyplot as plt
plt.hist(o,bins=50) #히스토그램. 
len(o)
o.mean()
o.std()

p = np.random.normal(2,1,10000) #평균:2,표준편차:1 난수 10000개 
p
plt.hist(p,bins=50) #히스토그램. 
p.mean()
p.std()
#0~9사이의 임의 정수 5개를 중복되지 않도록 선택.
#choice(값의범위,갯수,재선택여부)
# replace=False : 중복불가
q = np.random.choice(10,5,replace=False)
q
#1~45까지 수를 중복되지 않도록 6개의 숫자를 선택.
r=np.random.choice(45,6,replace=False)+1
r
r.sort()
r

#확률을 적용하여 난수 추출
#0~5까지의 숫자 100개를 추출. 중복가능 
# p=[0.1,0.2,0.3,0.2,0.1,0.1] : p속성의 값의 합은 1이어야함.
s=np.random.choice(6,100,p=[0.1,0.2,0.3,0.2,0.1,0.1])
s
lists = list(s)
lists.count(0)
lists.count(1)
lists.count(2)
lists.count(3)
lists.count(4)
lists.count(5)

fruits = ['apple', 'banana', 'cherries', 'durian', 'grapes']
t=np.random.choice(fruits,100,p=[0.1,0.2,0.3,0.2,0.2])
t
listt = list(t)
for d in fruits :
   print(d,"=",listt.count(d))

'''
 -- 행정안전부 홈페이지 : www.mois.go.kr
   -> 정책자료 -> 주민등록인구통계 -> 연령별인구현황
   -> 계:선택, 남여구분:선택안함
   -> 연령 구분 단위 : 1세
   -> 만 연령구분 : 0세 ~ 100세이상
   -> 전체읍면동현황 선택
   -> csv파일로 다운받기 (age.csv 파일로 이름 변경)
 -- age.csv 파일 data폴더에 저장하기 (data 폴더 생성)  
'''
import numpy as np
import csv
import matplotlib.pyplot as plt

#age.csv파일을 읽어, 인구구조알고 싶은 동을 찾아서
# 인구 구조 그래프를 작성하기
f=open("data/age.csv")
data=csv.reader(f) #csv 모듈을 이용하여 csv 파일 읽기. 
data
type(data)
next(data) # 1줄을 읽기. header 정보 읽기. 
name='역삼'
for row in data :
    #row : data에서 한줄 저장
    if row[0].find(name) >= 0 : #신림동 정보?
        row = list(map((lambda x : x.replace(",","")),row)) #숫자의 ,부분 제거. 각열을 리스트로 변경
        print(row)
        home = np.array(row[3:],dtype=int) #row의 0세데이터 이후 부터 데이터를 정수형 배열 생성 
        print(home)
        break

plt.style.use('ggplot')
plt.figure(figsize=(10,5),dpi=100) 
plt.rc('font',family='Malgun Gothic') #한글 폰트 설정. 
plt.title(name+' 지역의 인구 구조')   
plt.plot(home)  #home 데이터를 선그래프 출력
plt.show()

# 같은 이름을 가진 동이 있는 경우 모든 동을 그래프로 작성하기
f=open("data/age.csv")
data=csv.reader(f) #csv 모듈을 이용하여 csv 파일 읽기. 
next(data)
data=list(data) #파일스트림의 데이터를 리스트 객체로 변경. 여러번 읽기가능함
name='신사'
homelist=[] #데이터 저장
namelist=[] #이름저장
import re
for row in data :
     if row[0].find(name) >= 0 : #name에 해당되는 레코드 
        row = list(map((lambda x:x.replace(",","")),row)) #숫자 ,를 제거
        homelist.append(np.array(row[3:], dtype = int)) #0세이후 데이터를 정수 배열. 리스트저장
#        namelist.append(row[0][:row[0].find('(')]) #이름 정보 저장 ( 이전데이터까지만 추가 
        namelist.append(re.sub('\(\d*\)', '', row[0])) #이름 정보 저장 (숫자들)인데이터 제거한 후 추가 
plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=100)
plt.rc('font', family ='Malgun Gothic')
plt.title(name +' 지역의 인구 구조')
for h,n in zip(homelist,namelist) :
    plt.plot(h,label=n) #선그래프 
plt.legend()    #범례 표시 
plt.show()

# age.csv 파일을 이용해서 선택한 지역의 인구구조와, 
# 가장비슷한 인구구조를 가진 지역의 그래프와 지역을 출력하기
# 선택한지역은 한개만 가능하도록 한다. 
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
f=open("data/age.csv")
data=csv.reader(f)
next(data) 
data=list(data) 
name='역삼2동'  #선택한지역 
mn=1
result_name = ''
result=0
np.array([2,4,6,8,10]) / 20  #배열 / 정수 연산
for row in data :
    if name in row[0] : #row[0] 데이터에 name값이 존재?
       row = list(map((lambda x:x.replace(",","")),row)) #, 값을 제거. 
       home = np.array(row[3:], dtype =int) /int(row[2]) #인구비율 = 연령별인구수/총인구수.
       home_name = re.sub('\(\d*\)', '', row[0]) #역삼2동의 첫번째 값. 
#home : 선택한지역의 인구 비율 데이터 저장하는 배열 
#home_name : 선택한 지역의 전체 이름

for row in data :
    row = list(map((lambda x:x.replace(",","")),row))
    away = np.array(row[3:], dtype =int) /int(row[2]) #인구비율계산
    #s : (선택한지역의인구비율 - 지역데이터의인구비율) ** 2 의 합
    s = np.sum((home - away) **2) 
    if s < mn and name not in row[0] :
        mn = s
        result_name = re.sub('\(\d*\)', '', row[0])
        result = away
#result : 선택한지역과 차이가 가장적은 지역의 인구 비율 저장         
#result_name : 선택한지역과 차이가 가장적은 지역의 이름 저장 
plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=100)            
plt.rc('font', family ='Malgun Gothic')
plt.title(name +' 지역과 가장 비슷한 인구 구조를 가진 지역')
plt.plot(home, label = home_name)
plt.plot(result, label = result_name)
plt.legend()
plt.show()

# 설정된 지역과 가장 비슷한 인구 비율가진 지역을 그래프로 표시하기
# 판다스로 변경하기
import pandas as pd
#판다스의 csv파일의 기본인코딩방식:utf-8. utf-8 방식이 아닌 파일은 encoding 방식 설정 필요
# 행정구역 컬럼을 인덱스로 저장 
# thousands="," : 숫자사이의 ,를 무시하고 숫자만 저장
df = pd.read_csv("data/age.csv",encoding="cp949",index_col=0,thousands=",") 
df.head()
df.info()
#컬럼이름 변경하기
col_name=['총인구수','연령구간인구수']
for i in range(0,101) :
    col_name.append(str(i)+'세')
df.columns=col_name
df.columns
#총인구수로 나눠 비율 값으로 변경
df = df.div(df["총인구수"],axis=0)
df.head()
#'총인구수','연령구간인구수' 컬럼 제거하기
#df.drop(['총인구수','연령구간인구수'],axis=1)
del df['총인구수'],df['연령구간인구수']
df.info()
# 결측값을 0으로 변경하기
df.fillna(0,inplace=True)
name="역삼2동" #선택데이터 
'''
df.index : 행정구역이름들 
df.index.str : 문자열 형태로 변환 
df.index.str.contains(name) : 선택된 이름을 포함하는 데이터만 결과 True
'''
a=df.index.str.contains(name)
a
df2 = df[a]  #df2 : 선택한 데이터 
df2
#b : 선택한 데이터를 제외한 데이터만 True 값
b = list(map((lambda x : not x),a)) #True값과 False을 변경 
df3 = df[b] # 선택한 데이터 외의 모든 데이터 
df3.head()
df3.T.head()
df3.T #전치 행렬
mn = 1
for label, content in df3.T.items():
    s = sum((content - df2.iloc[0])**2)
    if s < mn :
        mn = s
        result = content  #인구비율
        result_name = label #동의이름
df2.T.plot() #선택된 동의 그래프 
result.plot(legend=result_name) #비숫한 동의 그래프 
plt.legend()
plt.show()
        
###########################
### 데이터 전처리 : 원본데이터를 원하는 형태로 변경.

#titanic 데이터 전처리 
import seaborn as sns

df = sns.load_dataset("titanic")
df.info()

#deck 컬럼의 값을 종류 조회
df.deck
df.deck.unique()
#deck 컬럼의 값의 갯수 
df.deck.value_counts()
#결측값을 포함하여 deck 컬럼의 값의 갯수 
df.deck.value_counts(dropna=False)

#isnull() : 결측값인 경우 True 반환
df.deck.head()
df.deck.head().isnull()
#notnull() : 결측값인 경우 False 반환
df.deck.head().notnull()
