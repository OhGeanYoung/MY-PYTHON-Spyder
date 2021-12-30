# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 08:59:08 2021

@author: KITCOOP
test1222-풀이.py
"""
'''
1. 화면에서 주민등록번호를 000000-0000000 형태로 입력받는다.
   주민등록번호 뒷자리의  첫 번째 숫자는 성별을 나타낸다. 
   주민등록번호에서 성별을 나타내는 숫자를 조회하여
   성별을 나타내는 숫자가 1,3 이면 남자로 2,4면 여자로 출력한다. 
   그외는 내국인아님으로 출력한다.
   -이 없는 경우는 '주민번호 입력오류' 출력하기
   
[결과]
000000-0000000 형태로 주민번호를 입력하세요123456-1234567
남자 

000000-0000000 형태로 주민번호를 입력하세요1234567891423
주민번호 입력 오류

000000-0000000 형태로 주민번호를 입력하세요123456-7891234
내국인 아님  
'''

jumin = input("000000-0000000 형태로 주민번호를 입력하세요 : ")
try :
    index = jumin.index("-") #- 문자가 없는 경우 예외 발생
    if(index!=6) :
        raise ValueError #예외 강제 발생
    gender = jumin[index+1:index+2]
    if(gender== '1' or gender == '3') :
        print("남자")
    elif (gender== '2' or gender == '4') :
        print("여자")
    else :
        print("내국인 아님")
except :
    print("주민번호 입력 오류")    
    
    
'''
2. 소문자와 숫자로 이루어진 문자를 암호화 하고 복호화 하는 프로그램 작성하기
  원래 문자 : a b c d e f g h i j k l m n o p q r s t u v w x y z 
  암호 문자 : ` ~ ! @ # $ % ^ & * ( ) - _ + = | [ ] { } ; : , . /

  원래 숫자 : 0 1 2 3 4 5 6 7 8 9 
  암호 숫자 : q w e r t y u i o p

[결과]
문자를 입력하세요 
원문 : abc123
암호화
`~!wer
복호화
abc123
'''    
den = "abcdefghijklmnopqrstuvwxyz0123456789"
cen = "`~!@#$%^&*()-_+=|[]{};:,./qwertyuiop"
src = input("문자를 입력하세요 : ") #abc123
result = ""
for i in range(len(src)): 
    result += cen[den.find(src[i])]
print("암호화")            
print(src,"=",result) 
      
src = result
result = ""
for i in range(0, len(src)):
    result += den[cen.find(src[i])]
print("복호화")            
print(src,"=",result)       


'''
3. 16진수를 입력하면 16진수 인지 아닌지 판단하여
   16진수가 맞으면 10진수로 변경하기.
   16진수가 아닌 경우 16진수 아님을 출력하기
[결과]
16진수 입력 : 12
ff 의 10진수: 255

16진수 입력 : 12
12 의 10진수: 18
   
16진수 입력 : ga
ga 는 16진수가 아닙니다.

'''
   
num16=input("16진수 입력 : ")
try :
   num10= int(num16,16)
except ValueError :
   print(num16,"는 16진수가 아닙니다.")
else :
    print(num16,"의 10진수:",num10)
    

'''
4. main이 실행 되도록  Rect 클래스 구현하기
    가로,세로를 멤버변수로.
    넓이(area),둘레(length)를 구하는 멤버 함수를 가진다
    클래스의 객체를 print 시 :  (가로,세로),넓이:xxx,둘레:xxx가 출력
[결과]
(10,20), 넓이:200,둘레:60
(10,10), 넓이:100,둘레:40
200 면적이 더 큰 사각형 입니다.
'''    
class Rect :
    w=0
    h=0
    def __init__(self,w,h) :  #생성자
        self.w = w
        self.h = h
    def __repr__(self) :  #객체 print시 호출되는 함수
        return "(%d,%d), 넓이:%d,둘레:%d" % \
       (self.w,self.h,self.area(),self.length())
    def __gt__(self,other) :
        return self.area() > other.area()
    def __lt__(self,other) :
        return self.area() < other.area()
    def __eq__(self,other) :
        return self.area() == other.area()
    def area(self) :    #넓이
        return self.w * self.h
    def length(self) :  #둘레
        return (self.w + self.h) * 2
    

if __name__ == "__main__" :
   rect1 = Rect(10,20)  #생성자
   rect2 = Rect(10,10)
   print(rect1)   #__repr__
   print(rect2)
   if rect1 > rect2 : #__gt__
      print(rect1.area(),"면적이 더 큰 사각형 입니다.")
   elif  rect1 < rect2 :  #__lt__
      print(rect2.area(),"더 큰 사각형 입니다.")
   elif rect1 == rect2 :  #__eq__
      print(rect1.area(),"=",rect2.area(),"같은 크기의 사각형 입니다.")


'''
5. 입력된 자연수가 홀수인지 짝수인지 판별해 주는 함수를 람다식을 이용하여 작성하기.
[결과]
자연수를 입력하세요 : 20
20 숫자는 짝수 입니다.

자연수를 입력하세요 : 25
25 숫자는 홀수 입니다.
'''
num = int(input("자연수를 입력하세요"))
odd = (lambda x: True if x % 2 == 1 else False)
print(num,"숫자는 ","홀수" if odd(num) else "짝수","입니다.")

num = int(input("자연수를 입력하세요"))
print(num,"숫자는 ","홀수" if (lambda x: True if x % 2 == 1 else False)(num) else "짝수","입니다.")

def odd(x) :
    if x % 2 == 1 :
        return True
    else :
        return False
    
num = int(input("자연수를 입력하세요"))
if (odd(num)) :
    print(num,"숫자는 홀수 입니다.")
else :
    print(num,"숫자는 짝수 입니다.")


num = int(input("자연수를 입력하세요"))
if (num%2==1) :
    print(num,"숫자는 홀수 입니다.")
else :
    print(num,"숫자는 짝수 입니다.")










