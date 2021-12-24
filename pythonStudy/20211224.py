# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 09:15:15 2021

@author: Zion_1956
"""

# sql에 파라미터 값 사용하기
import sqlite3
conn = sqlite3.connect("iddb")
cur = conn.cursor()
while True:
    param=[]
    d1 = input("사용자ID: ")
    if d1 =='':
        break
    d2 = input("사용자이름: ")
    d3 = input("이메일: ")
    d4 = input("출생년도: ")
    sql = "insert into usertable (id,username,email,birthyear) \
        values (?,?,?,?)"
    param.append(d1)
    param.append(d2)
    param.append(d3)
    param.append(d4)
    cur.execute(sql,param)
    conn.commit()
conn.close()

conn = sqlite3.connect("iddb")
cur = conn.cursor()
cur.execute("select * from usertable")
while True :
    row = cur.fetchone() #레코드 한건씩 조회
    if row == None:
        break
    print(row)
    
# 여러건은 한꺼번에 추가하기
data=[('test4','테스트4','test4@aaa.bbb',1991),
      ('test5','테스트5','test5@aaa.bbb',1993),
      ('test6','테스트6','test6@aaa.bbb',1994),
      ('test7','테스트7','test7@aaa.bbb',1995)]
con = sqlite3.connect("iddb")
cur = con.cursor()
# cur.executemany : 데이터를 여러개 저장 
cur.executemany("""insert into usertable
    (id,username,email,birthyear) values (?,?,?,?)""",data)
con.commit();
con.close()

conn = sqlite3.connect("iddb")
cur = conn.cursor()
cur.execute("select * from usertable")
while True:
    row = cur.fetchone() #레코드 한건씩 조회
    if row == None:
        break
    print(row)
    
#db 내용 수정하기
con = sqlite3.connect("iddb")
cur = conn.cursor()
cur.execute("update usertable set id='0003' where username='김삿갓'")
con.commit()
con.close()

conn = sqlite3.connect("iddb")
cur = conn.cursor()
cur.execute("select * from usertable")
while True:
    row = cur.fetchone() #레코드 한건씩 조회
    if row == None:
        break
    print(row)
        
#db내용 삭제하기
con = sqlite3.connect("iddb")
cur = conn.cursor()
cur.execute("delete from usertable where username='김삿갓'")
con.commit()
con.close()
con = sqlite3.connect("iddb")
cur = conn.cursor()
cur.execute("select * from usertable")
while True:
    row = cur.fetchone() #레코드 한건씩 조회
    if row == None:
        break
    print(row)

#오라클 연결하기
#1. anaconda prompt : pip install cx_Oracle 실행
#2. 오라클 클라이언트 파일 설정
# https://www.oracle.com/database/technologies/instant-client/downloads.html
# 압축푼 폴더를 환경변수의 path에 추가한다.
import cx_Oracle 
con = cx_Oracle.connect('scott','tiger','localhost/orcl')
cur = con.cursor()
cur.execute("select * from item")
for st in cur.fetchall() :
    print(st)
con.close()

'''
mariadb 설정
1. docker로 실행
'''
import pandas as pd
import pymysql
conn = pymysql.connect(host='localhost', port=3333, user='root', password='1234', db='fundb', charset='utf8mb4') # 접속정보
df_customer = pd.read_sql(sql='select * from customer', con=conn)
print(df_customer)

con = pymysql.connect(host="localhost", port=3333, user="kic", passwd="1234", db="kicdb", charset="UTF8")
try :
    cur = con.cursor()
    cur.execute("select * from item")
    for row in cur.fetchall():
        print(type(row),row)
finally:
    cur.close()
    con.close()        

#pandas
import pandas as pd  #pandas 모듈을 pd 이름으로 사용함
dict_data = {'a':1,'b':2,'c':3} # 딕셔너리 데이터
sr = pd.Series(dict_data)
print(sr)
print(type(sr))
'''
index   value
a       1
b       2
c       3
'''
print(sr.index) #index만 출력

print(sr.values) #values만 출력

#리스트를 시리즈 객체로
list_data = ['2021-12-24',3.14,'ABC',100,True]
sr = pd.Series(list_data)
print(sr)
'''
index
0      2021-12-24
1            3.14
2             ABC
3             100
4            TRUE
'''
print(sr.index) #index만 출력

print(sr.values) #values만 출력

#튜플 시리즈 객체로
tup_data = ("홍길동",'1990-01-01','남',True)
sr = pd.Series(tup_data,index=['이름','생년월일','성별','학생여부'])
print(sr)

#인덱스 출력하기
print(sr.index)

#한개의 값 조회
print(sr[0])      # 순서로 값을 조회
print(sr['이름']) # 인덱스를 이용하여 값을 조회
#생년월일 조회
print(sr[1])           # 순서로 값을 조회
print(sr['생년월일'])  # 인덱스를 이용하여 값을 조회

#여러개의 값 조회
print(sr[[1,2]])   # 순서로 값을 조회
print(sr[['생년월일','성별']])  # 순서로 값을 조회

print(sr[1,2])               #오류발생
print(sr['생년월일','성별'])  #오류발생

#여러개의 값 조회 : 범위 지정
print(sr[1:2])               #1번 순위 인덱스부터 2번 순위 인덱스 앞까지.
print(sr['생년월일':'성별'])  #생년 월일 부터 성별까지
print(sr['생년월일':'학생여부'])  #오류발생

# 딕셔너리를 이용하여 데이터프레임 객체 생성하기
dict_data = {'c0':[1,2,3],'c1':[4,5,6],'c2':[7,8,9],'c3':[10,11,12],'c4':[13,14,15]}
df = pd.DataFrame(dict_data)
print(df)

#데이터 프레임 : 행, 열.
# index : 행의이름
# columns : 열의 이름

df = pd.DataFrame([[15,"남","서울중"],[17,"여","서울여고"],[17,"남","서울고"]],
                  index=["홍길동","성춘향","이몽룡"],
                  columns=['나이','성별','학교'])
print(df)

#인덱스 이름을 변경하기
df.index=['학생1','학생2','학생3']
print(df)
#컬럼이름을 조회하기
print(df.columns)
#컬럼이름을 변경하기 : age,gender,school
df.columns=['age','gender','school']
print(df)

#rename 함수 : 일부분만 변경 가능
df.rename(columns={"age":"나이"},inplace=True)
print(df)

df.rename(columns={"gender":"남녀"})
print(df)
#rename 함수를 이용하여 index만 변경
df.rename(index={"학생1":"홍길동"},inplace=True)
print(df)

#exam_data 의 인덱스가 홍길동,이몽룡,김삿갓으로 나오도록 dataFrame 객체 생성하기
exam_data ={'수학':[90,80,70],'영어':[98,89,95],'음악':[85,95,100],'체육':[100,90,90]}

df = pd.DataFrame(exam_data, index=['홍길동','이몽룡','김삿갓'])
print(df)

# mean() : 평균
df.mean()
df["수학"].mean() #df 데이터와 수학 컬럼의 평균
type(df["수학"])
df.mean()["수학"] #df 데이터의 평균값들 중 인덱스가 수학인 데이터만 출력해
#홍길동 평균. loc 값 행을 선택. 인덱스 접근 
df.loc["홍길동"].mean()
#합계 : sum()
df.sum()
#최대값 : max()
df.max()
#수학점수중 가장 큰 값 출력
df.max()["수학"]
df["수학"].max()
#최소값 : min()
df.min()
#중간값 : median()
df.median()
df
#std() 표준편차
df.std()
df
#var() 분산
df.var()

# 기술 통계 정보
df.describe()

# 데이터 프레임 데이터의 간략 정보
df.info()

# 복사하기
df2 = df #얕은 복사, df2객체와 df 객체는 동일한 객체임.
print(df2)
df.rename(index={"홍길동":"학생1"},inplace=True)
df.rename(index={"학생1":"홍길동1"},inplace=True)
print(df)
print(df2)
df3 = df[:] # 깊은 복사 . df3과 df 객체는 다른 객체임.
print(df3)
df.rename(index={"홍길동1":"학생1"},inplace=True)
print(df)
print(df3)

#행 제거하기
#axis=0 : 행을 의미
#axis=1 : 열을 의미
df3.drop(["홍길동1","김삿갓"],axis=0, inplace=True)
print(df3)
#df3의 체육 컬럼 제거
df3.drop(["체육"],axis=1,inplace=True)
print(df3)

del df3["음악"]
print(df3)

df4 = df.copy()
print(df4)
df4.rename(index={"학생1":"홍길동1"},inplace=True)
print(df4)

#df4 데이터프레임 객체에서 음악,체육 컬럼을 제거하기
del df4["음악"]
del df4["체육"]
print(df4)

df4.drop(["음악","체육"],axis=1,inplace=True)
df4

#df 수학 컬럼 조회하기
df["수학"]
df.수학

#df 수학,영어 컬럼 조회하기
df[["수학","영어"]]
df["수학":"영어"]
df
#이몽룡 의 점수를 조회하자
#loc : index 이름으로 조회
#iloc : index 순번으로 조회
df.loc["이몽룡"]
df.iloc[1]
# 이몽룡,김삿갓 학생의 점수를 조회하기
df.loc[["이몽룡","김삿갓"]]
df.iloc[[1,2]]

#범위를 이용하여 조회하기
df.loc["이몽룡":"김삿갓"]
df.iloc[1,3]

print(df.loc["이몽룡"])   # 시리즈 객체
print(df.loc[["이몽룡"]]) # 데이터프레임 객체
print(type(df.loc["이몽룡"])) #시리즈 객체
print(type(df.loc[["이몽룡"]])) #데이터프레임 객체

#jeju1.csv 파일을 판다스 데이터 읽기
df = pd.read_csv("jeju1.csv")
df
# df 데이터의 간략 정보 조회하기
df.info()
df.head() # 처음 5개 레코드만 조회
df.tail() # 마지막 5개 레코드만 조회
print(type(df))

#경도(LON)만 출력하기.
print(df["LON"])           #Series 객체
print(df[["LON"]])         #DataFrame 객체
print(df.LON)              #Series 객체
print(type(df["LON"]))     #Series 객체
print(type(df[["LON"]]))   #DataFrame 객체
print(type(df.LON))        #Series 객체

#경도(LON),위도(LAT) 출력하기.
print(df[["LON","LAT"]])   #DataFrame 객체
#장소 컬럼을 인덱스로 변경하기
df.set_index("장소",inplace=True)
df
df.loc["제주국제공항"]
#인덱스 값을 "여행지"컬럼으로 생성하기
df["여행지"]=df.index
df
#인덱스를 컬럼으로 변경하기
df.reset_index(inplace=True)
df
#장소 컬럼 제거하기
df.drop("장소",axis=1,inplace=True)
df
# df 데이터를 csv 파일로 생성하기
# index=False : 인덱스는 파일로 저장 안함

df.to_csv("df__jeju.csv",index=False)
