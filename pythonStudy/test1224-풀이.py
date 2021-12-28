# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:56:12 2021

@author: KITCOOP
test1224.py
"""
'''
 1. 다음 number 데이터를 이용하여 큰글씨를 출력하는 프로그램 작성하기 
   [결과]
   숫자를 입력하세요 =>12345
     *  ***  ***  * *  ***  
     *    *    *  * *  *    
     *  ***  ***  ***  ***  
     *  *      *    *    *  
     *  ***  ***    *  ***  
'''
number= [["*** ","* * ","* * ","* * ","*** "],
	     ["  * ","  * ","  * ","  * ","  * "],
		 ["*** ","  * ","*** ","*   ","*** "],
		 ["*** ","  * ","*** ","  * ","*** "],
		 ["* * ","* * ","*** ","  * ","  * "],
		 ["*** ","*   ","*** ","  * ","*** "],
		 ["*** ","*   ","*** ","* * ","*** "],
		 ["*** ","  * ","  * ","  * ","  * "],
		 ["*** ","* * ","*** ","* * ","*** "],
		 ["*** ","* * ","*** ","  * ","  * "]
		]

num = input("숫자를 입력하세요 =>")
for i in range(5) :
    for n in num :
        print(number[int(n)][i],end=" ")
    print()    

'''
 2. 오라클 데이터 베이스의 테이블을 조회하기
select 구문을 입력하고 다음과 같은 결과가 출력되도록 프로그램을 작성하시오
[결과]
sql 입력하세요=========
select id,name,price from item

조회 레코드수: 7 ,조회 컬럼수: 3

(1, '레몬', 50)
(2, '오렌지', 100)
(3, '키위', 200)
(4, '포도', 300)
(5, '딸기', 800)
(6, '귤', 1000)
(7, '사과', 10000)

sql 입력하세요=========

'''
import cx_Oracle
conn = cx_Oracle.connect('kms','1234','localhost/xe')
cur = conn.cursor()
while True :
    sql = input("sql 입력하세요=========\n")
    if sql=="" :
        break
    cur.execute(sql)
    rows = cur.fetchall()
    print()
    print("조회 레코드수:",len(rows),",조회 컬럼수:",len(rows[0]))    
    print()
    for row in rows :
        print(row)
cur.close()
conn.close();

'''
3. 주어진 자연수 N에 대해 N이 짝수이면 N!을,  홀수이면 ΣN을 구하는 함수
   calc 함수를 작성하기
'''
def calc(n):
    if n % 2 == 0:
        fac = lambda x :  x if x==1 else x*fac(x-1)
        return fac(n)
    else:
        return sum([x for x in range(n+1)])

num = int(input("숫자를 입력하세요")) 
print(calc(num))

"""
4. 화면에서 숫자를 입력받아 야구 게임하기
   1. 시스템이 중복되지 않은 숫자 4개를 저장
   2. 화면에서 숫자를 입력받으면, 
      strike, ball을 출력
   3. 4 strike인 경우 정답.   
[결과]
서로다른 4자리 숫자를 입력하세요: 1234
Strike: 0 Ball: 3

서로다른 4자리 숫자를 입력하세요: 5678
Strike: 0 Ball: 1

서로다른 4자리 숫자를 입력하세요: 2348
Strike: 3 Ball: 0

서로다른 4자리 숫자를 입력하세요: 2346
4 번만에 맞췄습니다.

"""
import random 
 
list1=[]
set1 = set(list1) #중복되지 않도록 데이터를 저장
while len(set1) < 4 :
    rnum = random.randrange(0,10) #0~9사이의 값
    set1.add(rnum)
list1 = list(set1) #컴퓨터가 저장하고 있는 숫자
print(list1) 
cnt = 0
while True :
    number = input("서로다른 4자리 숫자를 입력하세요: ")
    cnt += 1
    strike = 0
    ball = 0
    for n in number:
        num = int(n) #입력받은 수 한자리.
        if list1.count(num)  == 1:
            if number.find(n) == list1.index(num):
                strike += 1
            else:
                ball += 1
                
    if(strike == 4) : #정답
        break                
    else :
        print("Strike:", strike, "Ball:", ball)

print(cnt,"번만에 맞췄습니다.")       

'''
5. supplier_data.csv 파일을 
  pandas를 이용하여 읽고 Invoice Number,Cost,Purchase Date
  컬럼만 df_data.csv 파일에 저장하기
'''
import pandas as pd

infile = "supplier_data.csv"
df = pd.read_csv(infile)

df_inset= df[["Invoice Number","Cost","Purchase Date"]]
print(df_inset)
df_inset.to_csv("df_data.csv",index=False)
