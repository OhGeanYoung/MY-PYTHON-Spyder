# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

20211220.py
"""

'''
 여러 줄 주석입니다.
'''

"""
 여러 줄 주석 입니다.
    '',""동일합니다
"""

# 한 줄 주석입니다.
# 한개의 내용을 콘솔에 출력하기
print("# 하나만 출력합니다")
'''
 스파이더에서 실행시 커서가 위치하는 라인을 실행하거나,
 선택된 부분을 실행할때 F9 키를 이용함
'''
print(10,20,30,40,50) #값을 여러개 출력할때는 , 를 이용함
print("안녕","나는","홍길동이야.","나이는",20,"살이다.")
print("홍길동","홍길동","홍길동")
print("홍길동" * 3)

#인용부호 사용하기 ",'
print("'안녕' 이라고 말했습니다.")
print('"안녕" 이라고 말했습니다.')
#특수문자 : \
print('\'안녕\' 이라고 말했습니다.')
print('파이썬에서 문자열은 "또는 \'를 사용할 수 있다.')
print("파이썬에서 문자열은 \"또는 '를 사용할 수 있다.")
# \n: 한줄
print("동해물과 백두산이 마르고 닳도록\n하느님이 보우하사 우리나라만세\n")
##문자열 : """ ... """ 여러줄 출력하기. 보여지는 대로 출력함
print("""동해물과 백두산이 마르고 닳도록
하느님이 보우하사 우리나라만세
      """)
print('''동해물과 백두산이 마르고 닳도록
      하느님이 보우하사 우리나라만세''')

#문자열 연결 : +
print("동해물과 백두산이 마르고 닳도록" + "하느님이 보우하사 우리나라만세")
print("동해물과 백두산이 마르고 닳도록" , "하느님이 보우하사 우리나라만세")

print("나이:"+20) #오류. 문자열 + 정수 +연산 불가
print("나이:",20) #, 는 가능하다.

#문자열 : 문자들의 모임. 문자 여러개. 문자의 배열이다.
print("안녕하세요"[0])
print("안녕하세요"[1])
print("안녕하세요"[2])
print("안녕하세요"[3])
print("안녕하세요"[4])
#안녕하세요 문자열에서 "안녕"문자 출력하기
print("안녕하세요"[0]+"안녕하세요"[1])
print("안녕하세요"[0:2]) #0번 인덱스 부터 (2-1)번 인덱스까지의 부분 문자열 출력
print("안녕하세요"[:2]) #0번 인덱스 부터 (2-1)번 인덱스까지의 부분 문자열 출력
#안녕하세요 문자열에서 "하세요" 문자 출력하기
print("안녕하세요"[2:5])
print("안녕하세요"[2:])
print("안녕하세요"[4]) #안녕하세요 문자열의 4번인덱스의 문자를 출력 
print("안녕하세요"[-1]) #안녕하세요 문자열의 마지막인덱스의 문자를 출력 . 뒤에서 첫번째
print("안녕하세요"[-2]) #안녕하세요 문자열의 마지막인덱스의 문자를 출력 . 뒤에서 두번째
#문자열[시작인덱스:종료인덱스+1:증가값]
print("안녕하세요"[::2]) #안하요
'''
     0 1 2 3 4 [0:5:2] = [::2]
    안녕하세요
    안하
'''


#len() : 문자열의 길이
print(len("안녕"))

###################자료형 : 선언이라는 개념이 없음. 변수 사용시 선언하지 않아도 됨.
# 파이썬에서 변수의 자료형은 값을 저장할때 자료형이 결정됨.
n = 10
print(n)
#type(n)변수의 자료형 출력
print(type(n))
print(type(10.5))

n=10.5
print(type(n))

n="안녕하세요"
print(type(n))

#연산자
#산술연산자 : +,*,-,/,%
print(5+7)
print(5-7)
print(5*7)
print(7/5) # 1.4 실수형 결과 
print(7//5) # 1 정수형 결과
print(7%5) #2
print(5**3) #125 = 5*5*5

#문제 : 3741초가 몇시간 몇분 몇초인지 출력하기
print(3741//3600,"시간",(3741%3600)//60,"분",(3741%3600)%60,"초")

#키보드에서 초를 입력받아 시분초 출력하기
# input 함수 : 키보드에서 데이터를 입력. 문자열 형태로 입력받는다.
# int 함수 : 형변환 함수 : 정수형 <- 문자열 변경
second=int(input("초를 입력하세요:"))
print(second//3600,"시간",(second%3600)//60,"분",(second%3600)%60,"초")

# 대입연산자 : =,+=,-=,*= ....
a=10
a=a+10
print(a)
a += 10
print(a)
a -= 10
print(a)
a *= 10
print(a)

#문자열에 사용가능한 대입연산자 : +=,*=
s="abcde"
s += 'b' # "abcde" + b

s *= 3
print(s) #abcdebabcdebabcdeb

# 문제 : 화면에서 자연수를 입력받아 입력받은 값에 +100 값을 화면에 출력하기
num = int(input("자연수를 입력하세요"))
print(num+100) #num 변수의 값을 변경 안됨
num += 100
print("num+100",num) #num  변수의 값이 변경됨.

#형변환 함수
#int() : 정수형 변환
# float(): 실수형 변환
# str() : 문자열 형 변환

print("num+100=" , str(num) + str(100))
print("num+100=" , float(num) + 100)

#2,8,16 진수 인식하여 => 10진수로 출력하기
print(int("11",2)) #3 int("11",2) : 11문자열을 2진수로 인식하여 정수로 변환해
print(int("11",8)) #3 int("11",2) : 11문자열을 8진수로 인식하여 정수로 변환해
print(int("11",16)) #3 int("11",2) : 11문자열을 16진수로 인식하여 정수로 변환해

# 10진수 데이터를 2,8,16진수로 출력하기
print(10,"을 2진수로 표현:",bin(10))
print(10,"을 8진수로 표현:",oct(10))
print(10,"을 16진수로 표현:",hex(10))

#format문자를 이용하여 출력하기
# %d : 10진 정수
# %f : 10진 실수
# %s : 문자열 출력하기
print("%d * %d= %d" % (2,3,6))
print("%f * %f= %f" % (2,3,6))
print("%.2f * %.2f= %.2f" % (2,3,6))

print("%s %s" % ("홍길동","안녕"))
print("%X" % (255)) #대문자로 A~F 표시
print("%x" % (255)) #소문자로 a~f 표시

# format 함수 이용하기
'''
{0:d}  format함수에 첫번째 숫자를 10진수로 출력 0번을 10진수로 출력해
{1:5d} format 함수에 두번째 숫자를 5자리를 확보하여 10진수로 출력
{2:5d} format 함수에 번째 숫자를 5자리를 확보하여 10진수로 출력
'''

print("{0:d}{1:5d}{2:5d}".format(100,200,300)) #100 200 300 최소5자리를 출력하라
'''
{2:d} : format 함수에 세번째 숫자를 10진수로 출력
{1:5d} : format 함수에 두번째 숫자를 5자리를 확보하여 10진수로 출력
{0:5d} : format 함수에 첫번째 숫자를 5자리를 확보하여 10진수로 출력
'''
print("{2:d}{1:5d}{0:5d}".format(100,200,300)) #300 200 100

#한줄로 출력하기
# end=" " : 기본값은 \n임.
print("홍길동", end=" ") #홍길동을 출력 후 공백을 출력해.
print("김삿갓")

print("안녕", end="! ") #홍길동을 출력 후 공백을 출력해.
print("김삿갓")

print("홍길동", end="") #홍길동을 출력 후 공백을 출력해.
print("김삿갓")

# 조건문 
'''
 자바
     if(3>7) {
             ......
             }
 파이썬 : 들여쓰기 필수
     if 3>5 :
         ........
         ........
         
         
    ....... => if 조건문 밖 문장
'''
score = 55
if score >= 90:
    print("A학점")
else :
    if score >= 80:
        print("B학점")
    else :
        if score >= 70:
            print("C학점")
        else :
            if score >= 60:
                print("D학점")
            else :
                print("F학점")
                
#if elif 
if score >= 90:
    print("A학점")
elif score >= 80:
    print("B학점")
elif score >= 70:
    print("C학점")
elif score >= 60:
    print("D학점")
else :
    print("F학점")
    
#문제 : 점수를 입력받아서 60점 이상이면 PASS 60미만인 경우 FAIL 출력하기
score=int(input("점수를 입력하세요"))
if score >=60:
    print("PASS")
else :
    print("FAIL")

#간단한 조건식으로 표현하기
# TRUE if 조건식 else FALSE
print(score,'점수는','PASS' if score >= 60 else 'FAIL',"입니다.")

#반복문
#숫자를 입력받아서 입력받은 숫자까지의 합을 출력하기
# range(시작값,종료값) : 시작값 부터 종료값 앞까지의 증감한 숫자들
num = int(input("숫자를 입력하세요 : ")) #10
sum = 0
for i in range(1,num+1) :
    sum += i
print("1부터 %d까지의 합: %d" % (num,sum))# 반복문 바깥 문장

#숫자를 입력받아서 입력받은 숫자까지의 합을 출력하기
sum = 0
for i in range(2,num+1,2) :
    if (i%2) == 0:
        sum += i
print("1부터 %d까지의 짝수의 합: %d" % (num,sum))# 반복문 바깥 문장

sum = 0
for i in range(2,num+1,2) :
    sum += i
print("1부터 %d까지의 짝수의 합 : %d" % (num,sum)) #반복문 바깥 문장

# while 구문으로 1~5까지의 숫자 출력하기
num = 1
while num <= 5 :
    print(num, end="")
    num += 1
    
# break     :반복문에서 빠짐
# continue  :반복문의 처음으로 제어이동
sum=0
for i in range(1,11):
    if i == 5:
        break
    sum += i
    
print("sum=",sum)

sum=0
for i in range(1,11):
    if i == 5:
        continue
    sum += i # 1+2+3+4+6...
    
print("sum=",sum)   #50

# 난수 생성하기. 모듈이용
import random
rnum = random.randrange(1,100) #1부터 99까지의 임의의 수를 리턴
print(rnum)

#1부터 10까지의 임의의 수 10개를 출력하기
for i in range(1,11) :
 rnum = random.randrange(1,10)
 print(rnum,end=" ")
print()

'''
    컴퓨터가 1부터 99사이의 임의의 수를 저장하고, 숫자를 입력받아서 컴퓨터가 저장한 수를 맞추기,
    컴퓨터는 입력한 숫자가 정답과 비교하여 큰수, 작은수 인지 출력
    정답 입력시 입력한 횟수를 출력하기.
    1. 난수 생성
    2. 정답을 맞추는 동안 계속 입력 받기 => while TRue : 
'''

import random
rnum = random.randrange(1,100)
cnt = 0
while True :
    ans = int(input("숫자를 입력하세요"))
    cnt += 1
    if ans > rnum:
        print("작은수 입니다.")
    elif ans < rnum:
        print("큰 수 입니다.")
    else :
        print("정답입니다.")
        break # 반복문 종료
print("%d번 만에 정답을 맞췄습니다." % cnt)

#중첩 반복문 : 반복문 내부에 반복문이 존재
#구구단 출력하기
i,j = 0,0 #i=0 j=0
for i in range(2,10): #i=2
    print("%5d단" % i)
    for j in range(1,10): #j=2
        print("%2d X %2d = %3d" % (i,j,(i*j)))
    print()
    
# 삼각형의 높이를 입력받아 삼각형을 *로 출력하기
# 삼각형의 높이 : 3
# *
# **
# ***


h = int(input("높이를 입력하세요"))
for i in range(1,h+1) :
    print("*" * i)
'''
    ***
    **
    *
'''
h = int(input("높이를 입력하세요"))
for i in range(h,0,-1) :
    print("*" * i)
'''
*** : 공백 : 0(3-3), *:3
 **   공백 : 1(3-2), *:2
  *   공백 : 2(3-1, *:1
'''
h = int(input("높이를 입력하세요"))
for i in range(h,0,-1) :
    print(" " * (h-i), end="")
    print("*" * i)

'''
  * : 공백 : 2(3-1), *:1
 **   공백 : 1(3-2), *:2
***   공백 : 0(3-3), *:3
'''
h = int(input("높이를 입력하세요"))
for i in range(1,h+1) :
    print(" " *(h-i), end="")
    print("*" * i)

#구구단 가로로 출력하기

i,j = 0,0 #i=0 j=0
for i in range(2,10): #i=2
    print("%5d단%3s" % (i," "),end="") #%4s : 공백문자열 4자리 출력
print()
for j in range(2,10): #j=2
    for i in range(2,10) :
        print("%2dX%2d=%3d" % (i,j,(i*j)),end="")
    print()
    
a="hello"
#a 문자열의 l자의 개수를 출력하기
cnt=0
for i in range(len(a)):
    if a[i] == 'l' :
        cnt += 1
print(a,"에서 l문자의 개수:",cnt)

#문자열 함수
#len(문자열) : 문자열의 길이 리턴
#count(문자) : 문자열에서 문자의 개수 리턴
#find(문자) : 문자열에서 문자의 위치 리턴. 없는 문자인 경우 -1 리턴
#index(문자) : 문자열에서 문자의 위치 리턴. 없는 문자인 경우 오류 발생
print(a,"에서 l문자의 개수:",a.count('l')) #l문자의 개수 : 2
print(a,"에서 l문자의 개수:",a.count('a')) #a문자의 개수 : 0

print(a,"에서 l문자의 위치:",a.find('l')) #l문자의 위치 : 2
print(a,"에서 l문자의 위치:",a.find('l',3)) #3번 인덱스부터 l문자의 위치 : 3
print(a,"에서 l문자의 위치:",a.find('a')) #a문자의 위치 : -1 => 없는 문자

print(a,"에서 l문자의 위치:",a.index('l')) #l문자의 위치 : 2
print(a,"에서 l문자의 위치:",a.index('l',3)) #3번 인덱스부터 l문자의 위치 : 3
print(a,"에서 l문자의 위치:",a.index('a')) #a문자의 위치 : 오류 발생

#문자열의 종류
str="123"
str="Aa123"
str="aa"
str=" "
if str.isdigit():
    print(str,':숫자')
if str.isalpha():
    print(str,':문자')
if str.isalnum():
    print(str,':문자 + 숫자') # 또는 의 의미이다.
if str.isupper():
    print(str,':대문자')
if str.islower():
    print(str,':소문자')
if str.isspace():
    print(str,':공백')