# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 17:14:38 2021

@author: Zion_1956
"""

#주민번호감추기
data= ''' 
    park 800905-1234567
    kim 700905-2234567
    choi 850101-a123456
'''
print(data)
result=[]
for line in data.split("\n"):
    word_result = []
    for word in line.split(" "):
        if len(word) == 14 and word[:6].isdigit() and word[:7].isdigit():
            word = word[:6]+"-"+"*******"
        word_result.append(word)
    result.append("".join(word_result))
print("\n".join(result))
#정규화모듈을 이용하여 주민번호 감추기
#패턴생성 후 감추기
#정규식을 이용한 데이터찾기
str1 = "The quick brown fox jumps over the laze dog Te Thhhhe te thhhe"
#대소문자 구문 x
str1 = "The quick brown fox jumps over the laze dog Te Thhhhe te thhhe"
#결과2 맞는 문자열 출력하기
#str2 문자열에서 온도의 평균을 출력하기
# 파일 읽기
#파일 쓰기 콘솔에서 내용을 입력받아 파일로 생성하기
#data.txt 파일을 읽어서 콘솔에 출력하기
#apple.gif 파일을 읽어서, apple2.gif파일로 저장하기
'''
    score.txt 파일의 내용
    홍길동,100
    김삿갓,50
    이몽룡,90
    임꺽정,80
    김길동,88
    
    문제 : score.txt 파일을 읽어서 점수의 총점과 평균 구하기
'''
#파일 정보 조회하기.
#존재여부
#작업폴더의 하위 파일 목록 출력하기
#문제: 하위 파일 목록의 파일/폴더 정보를 출력하기
#폴더 생성/제거
### 데이터 베이스에서 데이터 가져오기
#sqlite : 파이썬 내부에 존재하는 dbms
'''
 drop table if exists items;
 create table items(item_id integer primary key,
                    name text unique, price integer);
 insert into items (name,price) values ('Apple',800);
 insert into items (name,price) values ('Orange',500);
 insert into items (name,price) values ('Banana',300);
'''
'''
    'iddb' sqlite db를 생성하기
    'iddb' db에 이름이 usertable인 테이블 생성하기
     id char(4) primary key, username char(15),email char(15),
     birthyear int) 컬럼을 가진 usertable 생성하기
'''
# 화면에서 id,이름,이메일,출생년도를 입력받아 db에 등록하기
#등록된 내용 조회하기