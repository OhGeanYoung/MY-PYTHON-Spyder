# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 09:28:24 2021

@author: Zion_1956
"""

'''
    예외처리 : try,except,else,finally,raise 예약어.
    클래스 : 상속,self,추상함수
        특수함수 : 생성자,__repr__ ....
    모듈 : import
       if __name__ == '__main__' : 
'''
#주민번호 감추기
#주석이 아니라 문자열 연결
data= ''' 
    park 800905-1234567
    kim 700905-2234567
    choi 850101-a123456
'''
print(data)
result = []
for line in data.split("\n"):
    #line : park 800905-1234567
    word_result = [] # [park, 800905-******]
    for word in line.split(" "):
        #word : 800905-1234567
        if len(word) == 14 and word[:6].isdigit() and word[7:].isdigit() :
            word = word[:6]+"-"+"*******"
        word_result.append(word)
    result.append(" ".join(word_result)) #join => 두개의 데이터를 연결 park + 800905-******
#"문자열".join(리스트) :
# 리스트의 모든 요소를 문자열로 연결하여 하나의 문자열로 생성
print("\n".join(result))
'''
park 800905-******
kim 700905-******
choi 850101-a123456
'''

#정규화 모듈 이용하여 주민번호 감추기
import re
data= ''' 
    park 800905-1234567
    kim 700905-2234567
    choi 850101-a123456
'''
#패턴 생성
"(\d{6,7})[-]\d{7}"

'''
() : 그룹
\d{6,7} : \d 숫자 {자리숫}. 숫자 6자리에서 7자리
[-] : -문자
\d{7} : 숫자 7자리.
'''
pat = re.compile("(\d{6,7})[-]\d{7}")
#pat.sub : data 문자열에서 패턴 찾아서 변경해
#\g<1> : 첫번째() 그룹
print(pat.sub("\g<1>-*******",data))

#정규식을 이용하여 데이터 찾기
str1 = "The quick brown fox jumps over the laze dog Te Thhhhe te thhhe"
str_list = str1.split() # 공백으로 구분
print(str_list)
# * : h 문자가 0개 이상인 단어.
pattern = re.compile("Th*e") # 패턴 설정
count = 0
for word in str_list :
    if pattern.search(word) : # The 패턴에 맞는지 검증/ 두번째 the는 해당 x
        count += 1
print("결과1 : {0:s}:{1:d}".format("개수", count))
#대소문자 구분 없이 검색
pattern = re.compile("Th*e",re.I) # 패턴 설정
count = 0
for word in str_list :
    if pattern.search(word) : # The 패턴에 맞는지 검증/ 두번째 the는 해당 x
        count += 1
print("결과1 : {0:s}:{1:d}".format("개수", count))

#결과2 맞는 문자열 출력하기
print("결과 3 :",end="")
for word in str_list :
    if pattern.search(word) : # The 패턴에 맞는지 검증/ 두번째 the는 해당 x
       print(word, end=" ")
print()

print("결과 4 :",re.findall(pattern, str1))
print("결과 5 : 개수 :",len(re.findall(pattern, str1)))
#결과 2에 패턴에 맞는 데이터를 aaa로 변경하기
print("결과 6 :",pattern.sub("aaa", str1))

#문제
#str2 문자열에서 온도의 평균을 출력하기
import re

str2 = "서울:25도,부산:23도,대구:27도,광주:26도,대전:25도,세종:270도"
pattern = re.compile("\\d{2,3}") # 잘라내서 출력
list1 = re.findall(pattern, str2) # 패턴과 일치하는 것을 list로 반환 
print(list1)
list1 = list(map(int,list1)) #map은 리스트의 요소를 지정된 함수로 처리해주는 함수
print(sum(list1)/len(list1))

'''
정규식
정규식에서 사용되는 기호
1. () : 그룹. 설정
2. \g : n번째 그룹 조회.
3. [] : 문자
    [a] : a문자
    [a-z] : 소문자
    [A-Za-z] : 영문자
    [0-9A-Za-z] : 영문자 + 숫자
4. {n} : n개.갯수
      ca{2}t : a문자가 2개
      ct : false
      cat : false
      caat : true
      caaat: false
    {n,m} : n이상 m개 이하 갯수
      ca{2,5}t : a 문자가 2개이상 5개이하
        ct : false
        cat : false
        caat : true
        caaat: true
5. \d : 숫자.
6. ? : 0 또는 1
    ca?t : a문자는 없거나, 1개만 가능
     "ct" : true
     "cat" : true
     "caat" : false
7. * : 0개이상
    ca*t : a문자는 0개이상
    "ct" : true
    "cat" : true
    "caat" : true
 8. + : 1개이상
    ca+t : a문자는 1개이상
    "ct" : false
    "cat" : true
    "caat" : true
9. \s : 공백¶
    \s+ : 공백문자 1개이상.
'''

# 파일 읽기
'''
    open("파일명","파일모드",[encoding='인코딩방식'])
    파일모드
        r: 읽기
        w: 쓰기
        a: 쓰기. append 모드. 기존 파일의 내용에 추가하는 방식
        t: text. 기본값
        b: 이진모드. binary 모드. 이미지,동영상
'''
infp = open("test1222.py","rt",encoding="UTF-8")
while True :
    instr = infp.readline() # 한줄읽기
    if instr == None or instr == '': #파일의 끝 
        break
    print(instr,end="")
infp.close()

#파일 쓰기
#콘솔에서 내용을 입력받아 파일로 생성하기
outfp = open("data.txt","w",encoding="UTF-8")
while True :
    outstr = input("내용 입력=>")
    if outstr == '':
        break
    outfp.writelines(outstr+"\n")
outfp.close()

#data.txt 파일을 읽어서 콘솔에 출력하기
'''
    readline() : 한줄씩 읽기
    read() : 한번에 읽기
    readlines() : 한줄씩 리스트형태로 반환
'''
infp = open("data.txt","rt",encoding="UTF-8")
while True :
    instr = infp.readline()
    if instr == None or instr == "" :
        break
    print(instr,end="")
infp.close()

infp = open("data.txt","rt",encoding="UTF-8")
print(infp.read())
infp.close() # 대용량에서는 비추천

infp = open("data.txt","rt",encoding="UTF-8")
print(infp.readlines())
infp.close()

#이미지파일 읽고 쓰기.
#apple.gif 파일을 읽어서, apple2.gif파일로 저장하기
infp = open("apple.gif","rb")
outfp = open("apple2.gif","wb")
while True :
    indata = infp.read()
    if not indata :
        break
    outfp.write(indata)
infp.close()
outfp.close()

'''
    score.txt 파일의 내용
    홍길동,100
    김삿갓,50
    이몽룡,90
    임꺽정,80
    김길동,88
    
    문제 : score.txt 파일을 읽어서 점수의 총점과 평균 구하기
'''
import re
infp = open("score.txt","r",encoding="UTF-8")
instr = infp.read()
print(instr)
pattern = re.compile("[0-9]") 
list1 = re.findall(pattern, instr)
print(list1)
list1 = list(map(int,list1)) 
print(sum(list1),sum(list1)/len(list1))

#파일 정보 조회하기.
import os.path
file = "data.txt" #상대경로
file="c:\\"       #절대경로
file="module"     #상대경로
if os.path.isfile(file):
    print(file,"은 파일입니다.")
elif os.path.isdir(file):
    print(file,"은 폴더입니다.")
    
#존재여부
file = "data1.txt" #상대경로
if os.path.exists(file) :
    print(file,"은 존재합니다.")
else :
    print(file,"은 없습니다.")
    
#현재 작업 폴더 위치
import os
print(os.getcwd())
cwd = os.getcwd()
#작업폴더의 하위 파일 목록 출력하기
print(os.listdir())
#문제: 하위 파일 목록의 파일/폴더 정보를 출력하기
for f in os.listdir():
    if os.path.isfile(f):
        print(f,":파일, 크기:", os.path.getsize(f))
    elif os.path.isdir(f):
        os.chdir(f) #작업 폴더의 위치를 수정
        print(f,":폴더, 하위파일개수:",len(os.listdir()))
        os.chdir(cwd)

#현재 작업 폴더 변경
os.chdir("C:\webtest\7.python\workspace\MY-PYTHON-Spyder")
os.chdir(cwd)

#폴더 생성
#temp 폴더 생성
os.mkdir("temp")
#temp 폴더 제거
os.rmdir("temp")

#엑셀 파일 읽기 : 외부 모듈이 필요함
# xlsx : openpyxl 모듈
# xls  : xlrd 모듈
import xlrd

import openpyxl
filename="sales_2015.xlsx"
book = openpyxl.load_workbook(filename)
sheet = book.worksheets[0] #첫번째 sheet 데이터
data =[]
#row : 행 한줄
for row in sheet.rows:
    line = []# 한행의 셀의 값들을 리스트로 저장
#enumerate(row) : 리스트에서 데이터와 인덱스 값을 리턴
    for l,d in enumerate(row):
        print(1,",",d.value)
        line.append(d.value)
    print(line)
    data.append(line)
print(data)

#xls 파일 읽기
from xlrd import open_workbook
infile = "ssec1804.xls"
workbook = open_workbook(infile) #엑셀파일 전체
print("sheet 의 개수",workbook.nsheets)
# sheets() : 엑셀의 sheet 데이터들
for worksheet in workbook.sheets() :
    # worksheet : sheet 데이터
    print("worksheet 이름:",worksheet.name)
    print("행의 수:",worksheet.nrows)
    print("컬럼의 수 :",worksheet.ncols)
    for row_index in range(worksheet.ncols):
        for column_index in range(worksheet.ncols) :
            print(worksheet.cell_value(row_index,column_index),",",end="")
        print()
        

'''
    csv    => 표형태의 데이터. 행,열로 이루어진 데이터
    excel  => pandas를 이용하여 모듈과 상관 없이 처리 가능.
                데이터 프레임 형식.
'''
### 데이터 베이스에서 데이터 가져오기
#sqlite : 파이썬 내부에 존재하는 dbms

import sqlite3
dbpath = "test.sqlite"
conn = sqlite3.connect(dbpath) #test.sqlite 데이터베이스 파일 생성.
cur = conn.cursor()            # sql구문 실행 객체
#executescript : sql구문 여러개 실행
cur.executescript("""
                  drop table if exists items;
                  create table items(item_id integer primary key,
                                     name text unique, price integer);
                  insert into items (name,price) values ('Apple',800);
                  insert into items (name,price) values ('Orange',500);
                  insert into items (name,price) values ('Banana',300);
""")
conn.commit()
cur = conn.cursor()
cur.execute("select * from items")
item_list = cur.fetchall() #cursor를 통해 실행한 모든 결과값 조회
print(item_list)
#item_list 데이터를 한줄씩 출력하기
for id,name,price in item_list :
    print(id,name,price)
    
'''
    'iddb' sqlite db를 생성하기
    'iddb' db에 이름이 usertable인 테이블 생성하기
     id char(4) primary key, username char(15),email char(15),
     birthyear int) 컬럼을 가진 usertable 생성하기
'''

import sqlite3
#dbpath = "iddb.sqlite"
conn = sqlite3.connect('iddb')
cur = conn.cursor()
cur.executescript("""
   drop table if exists usertable;
   create table usertable (id char(4) primary key,
                     username char(15), email char(15),
                    birthyear int);
""")
conn.commit()

# 화면에서 id,이름,이메일,출생년도를 입력받아 db에 등록하기
while True :
    d1 = input("사용자ID : ")
    if d1 == '':
        break
    d2 = input("사용자이름: ")
    d3 = input("이메일: ")
    d4 = input("출생년도: ")
    sql = "insert into usertable (id,username,email,birthyear) values ('"\
        +d1+"','"+d2+"','"+d3+"',"+d4+")"
    print(sql)
    cur.execute(sql)
    conn.commit()
    
conn.close()

#등록된 내용 조회하기
conn = sqlite3.connect("iddb")
cur = conn.cursor()
cur.execute("select * from usertable")
userlist = cur.fetchall()
for u in userlist :
    print(u)
