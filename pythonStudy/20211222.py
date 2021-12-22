# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 09:31:51 2021

@author: KITCOOP
20211222.py
"""
'''
  컬렉션 객체
   리스트 : 배열. 인덱스로 객체를 접근. [  ] 표현
   튜플   : 상수화된 리스트.  ( ) 표현
   딕셔너리 : (키,값) 쌍인 객체들의 모임. { } 표현
   셋 : 중복불가. 순서를 모름. 인덱스 사용안됨. 집합관련객체 { } 표현
  
  컴프리헨션 : 컬렉션객체 생성시 사용되는 간편한 방식

  함수 : def 함수 정의
         return 리턴값
         매개변수 : 가변매개변수 : * p : 매개변수 0개이상 
                   기본값 설정 : n1=0,n2=1   : 0,1,2 개의 매개변수 
'''

# 람다식을 이용한 함수
def hap1(num1,num2) :
    return num1+num2

print(hap1(10))  #오류
print(hap1(10,20))
print(hap1(10.5,20.1))

hap2 = lambda num1,num2:num1+num2
print(hap2(10))  #오류
print(hap2(10,20))
print(hap2(10.5,20.1))

#기본값을 가진 매개변수로 변경하기 
hap3 = lambda num1=0,num2=1:num1+num2
print(hap3(10)) 
print(hap3(10,20))
print(hap3(10.5,20.1))

mylist=[1,2,3,4,5]
# mylist 객체의 요소들에 10을 더한 값을 가진 mylist2 객체 생성하기
mylist2 = []
for n in mylist :
    mylist2.append(n+10)
print(mylist2)    

def add10(n) :
    return n+10

mylist2 = []
for n in mylist :
    mylist2.append(add10(n))
print(mylist2)    

#map : list객체의 각각의 요소들에 적용되는 함수를 설정
#map(add10,mylist) : mylist 리스트의 요소들에 각각 add10 함수를 적용
mylist2 = list(map(add10,mylist))
print(mylist2)    

mylist2 = list(map(lambda n:n+10,mylist))
print(mylist2)    

#예외 처리 : 
#  try except 예약어 사용.

idx = "파이썬".index("일")
print(idx)

try :
    idx = "파이썬".index("일")
    print(idx)
except :
    print("파이썬 문자열에는 일자는 존재하지 않습니다.")    

#mystr문자열에 파이썬 문자의 위치를 strpos 리스트에 저장하기
mystr = "파이썬 공부 중입니다. 파이썬을 열심히 공부합시다"
strpos = [] #파이썬의 위치값 저장.
idx = 0
while True :
    try :
        idx = mystr.index("파이썬",idx)
        strpos.append(idx) #0,13
        idx += 1
    except :
        break
print(strpos)

try :
    print(int("010-12345678")) #숫자만 입력하세요 메세지 출력
except :
    print("숫자만 입력하세요") 
    
#다중 예외처리 : 하나의 try구문에 여러개의 except 구문이 존재    
#               예외별로 다른 처리 가능 

num1 = input("숫자형 문자 1 입력 :") #10
num2 = input("숫자형 문자 2 입력 :") #0
try :
   a=[]
   print(a[0])  #IndexError 
   num1=int(num1) #ValueError 발생가능. 정수형문자가 아닌경우
   num2=int(num2)
   print(num1+num2)
   print(num1/num2) #ZeroDivisionError. 0으로 나눈경우 발생
except ValueError as e :  
    print("숫자로 변환 할 수 없습니다.")
    print(e)
except ZeroDivisionError as e :
    print("두번째 문자는 0 입력 불가합니다.")    
    print(e)
except : #그외 모든 오류 처리. 다중예외처리의 마지막에 구현
    print("프로그램 오류가 발생했습니다. 확인하세요")    
finally : #정상,예외발생 모두 실행 되는 구문 
    print("프로그램 종료")   

# 다중예외처리를 하나의 변수로 묶기
try :
    a=[1]
    print(a[0]) #IndexError
    print(int('10')) #ValueError
    4/0   #ZeroDivisionError
except (ValueError,IndexError) as e :
    print("프로그램 입력 오류")
except ZeroDivisionError as e :
    print("0으로 나누지 마세요")

#else : 예외 발생이 안된 정상 적인 경우 실행됨.
try :
    age = int(input("나이를 입력하세요:")) #ValueError
except :
    print("숫자로 입력해 주세요")    
else : #try 블럭에서 예외 발생이 안된 경우
   if age < 19 :
       print("미성년 입니다.")
   else :
       print("성인입니다.")

try :
   age = int(input("나이를 입력하세요:")) #ValueError
   if age < 19 :
       print("미성년 입니다.")
   else :
       print("성인입니다.")
except :
    print("숫자로 입력해 주세요")    

#pass 예약어 : 구문이 없는 경우
n=10
if n > 0 :
    pass  #실행할 문장이 없음.
else :
   print(0)   

try :
   age = int(input("나이를 입력하세요:")) #ValueError
   if age < 19 :
       print("미성년 입니다.")
   else :
       print("성인입니다.")
except :
    pass

# raise : 예외 강제 발생 
try :
    print(1)
    raise ValueError  #ValueError 강제 발생
    print(2)
except ValueError :
    print("ValueError 예외 발생")    

'''
   클래스 : 사용자 정의 자료형
           구조체 + 함수
           멤버변수 + 멤버함수
           
   객체지향언어 : 파이썬 불완전 객체지향. 
        상속 : 다중상속이 가능함.
   
    self : 자기참조변수. 인스턴스 함수의 매개변수로 설정해야함     
    
    기본생성자 : 
        def __init__(self) :
            pass
'''
class Car : #클래스 정의. 자료형 정의. 생성자 구현하지 않으면 기본생성자 제공 
    color=""
    speed=0 
    def upSpeed(self,value) :  #self : 자기참조변수.
        self.speed += value
    def downSpeed(self,value) :
        self.speed -= value

car1 = Car()  #객체화
car1.color="빨강"
car1.speed=10
car2 = Car()
car2.color="파랑"
car2.speed=0
car3 = Car()
car3.color="노랑"
car3.speed=0

car1.upSpeed(30)
print("자동차 1의 색상은 %s 이며, 현재 속도는 %dkm 입니다."  %(car1.color,car1.speed))
car2.upSpeed(60)
print("자동차 2의 색상은 %s 이며, 현재 속도는 %dkm 입니다."  %(car2.color,car2.speed))
car3.upSpeed(10)
print("자동차 3의 색상은 %s 이며, 현재 속도는 %dkm 입니다."  %(car3.color,car3.speed))

'''
  생성자 : 객체화시 관여하는 메서드 
           def __init__(self,...) 
'''

class Car :
    color=""
    speed=0 
    def __init__(self,v1,v2) : #생성자
        self.color = v1
        self.speed = v2
    def upSpeed(self,value) : 
        self.speed += value
    def downSpeed(self,value) :
        self.speed -= value

car1 = Car("빨강",10)  #객체화
car2 = Car("파랑",0)
car3 = Car("노랑",0)
car1.upSpeed(30)
print("자동차 1의 색상은 %s 이며, 현재 속도는 %dkm 입니다."  %(car1.color,car1.speed))
car2.upSpeed(60)
print("자동차 2의 색상은 %s 이며, 현재 속도는 %dkm 입니다."  %(car2.color,car2.speed))
car3.upSpeed(10)
print("자동차 3의 색상은 %s 이며, 현재 속도는 %dkm 입니다."  %(car3.color,car3.speed))

# 인스턴스 변수 : 생성자에서 self.멤버명 
# 클래스 변수 : 생성자에서 클래스명.멤버명. 모든 객체의 공통 변수 
class Car :
    color=""  #인스턴스변수
    speed=0   #인스턴스변수
    num=0     #인스턴스변수
    count=0   #클래스변수
    def __init__(self,v1='',v2=0) : #생성자
        self.color = v1   #인스턴스변수
        self.speed = v2   #인스턴스변수
        Car.count += 1    #클래스 변수
        self.num = Car.count  #인스턴스변수
    def printMessage(self) :
       print("색상: %s, 속도:%dkm, 번호:%d, 생산번호:%d" % \
             (self.color,self.speed,self.num,Car.count))

car1,car2 = None,None
car1=Car("빨강",10)
car1.printMessage()
car2=Car("파랑")
car1.printMessage()
car2.printMessage()

#문제
# Card 클래스 구현하기
#  멤버변수 : kind,number, no, count
#  멤버함수 : printMessage()
class Card :
    kind=""
    number=0
    no=0
    count=0;
    def __init__(self,v1="Spade",v2=1) :
        self.kind = v1
        self.number=v2
        Card.count += 1
        self.no = Card.count
    def printMessage(self) :
        print("kind:",self.kind,",number:",self.number,
              "no:",self.no,",count:",Card.count)
        
card1 = Card()
card1.printMessage() #kind:Spade,number:1, no:1, count:1
card2 = Card("Heart")
card3 = Card("Spade",10)
card1.printMessage() #kind:Spade,number:1, no:1, count:3
card2.printMessage() #kind:Heart,number:1, no:2, count:3
card3.printMessage() #kind:Spade,number:10, no:3, count:3

# 상속 : 기존의 클래스를 상속받아 새로운 클래스를 생성. 다중상속 가능
# class 자손클래스명 (부모클래스1,부모클래스2,...) :
class Car :
    speed=0
    door = 3
    def upSpeed(self,v) :
        self.speed = v
        print("현재 속도(부모클래스):%d" % self.speed)
class Sedan(Car) : # Sedan 클래스는 Car 클래스를 상속받음. 
    pass

class Truck(Car) : # Truck 클래스는 Car 클래스를 상속받음. 
    def upSpeed(self,v) :  #오버라이딩됨.
        self.speed = v
        if self.speed > 150 :
            self.speed = 150
        print("현재 속도(자손클래스):%d" % self.speed)
        
car1 = Car()
car1.upSpeed(200)
car2 = Sedan()
car3 = Truck()
print("승용차:",end="")
car2.upSpeed(200)
print("트럭:",end="")
car3.upSpeed(200)

#클래스에서 사용되는 특별한 메서드들
class Line :
    length=0
    def __init__(self,length) : #생성자
        self.length =length
    def __repr__(self) :        # 객체의 문자열화하여 출력값제공
        return "선길이:" + str(self.length)
    def __add__(self,other) :
        print("+ 연산자 호출 : ",end="")
        return self.length + other.length        
    def __lt__(self,other) :
        print("< 연산자 호출 : ",end="")
        return self.length < other.length        
    def __gt__(self,other) :
        print("> 연산자 호출 : ",end="")
        return self.length > other.length        
    def __eq__(self,other) :
        print("== 연산자 호출 : ",end="")
        return self.length == other.length        

line1 = Line(200)        
line2 = Line(100)
print("line1=",line1)  #__repr__
print("line2=",line2)  #__repr__
print("두선의 합:",line1+line2) #__add__
print("두선의 합:",line1.__add__(line2)) #__add__

if line1 < line2 :             #__lt__
    print("line2선이 더 깁니다.")
elif  line1 == line2 :         #__eq__
    print("두선의 길이는 같습니다.")
elif  line1 > line2 :          #__gt__
    print("line1선이 더 깁니다.")

'''
클래스에서 사용되는 연산자에 사용되는 특수 함수
+   __add__(self, other)
–	__sub__(self, other)
*	__mul__(self, other)
/	__truediv__(self, other)
//	__floordiv__(self, other)

%	__mod__(self, other)

**	__pow__(self, other)

&	__and__(self, other)
|	__or__(self, other)
^	__xor__(self, other)

<	__lt__(self, other)
>	__gt__(self, other)
<=	__le__(self, other)
>=	__ge__(self, other)
==	__eq__(self, other)
!=	__ne__(self, other)


생성자 : __init__(self,...) : 클래스 객체 생성시 요구되는 매개변수에 맞도록 매개변수 구현
출력   : __repr__(self) : 클래스의 객체를 출력할때 문자열로 리턴.
'''
    
#추상함수 : 부모클래스의 멤버 중 추상 함수가 있으면, 자손에서 반드시 오버라이딩 해야 함.
#         함수의 구현부에 raise notImplementedError 기술함

class Parent :
    def method(self) : #추상함수 
        raise NotImplementedError

class Child(Parent) :
#    pass        
    def method(self) :
        print("자손클래스에서 오버라이딩 함")

ch = Child()
ch.method()

### 모듈 
#모듈 위치 설정하기
import sys
sys.path.append("C:/20211108/python/workspace/module")

import mod1  #mod1.py 파일의 내용을 가져옴.
import mod2  #mod2.py 파일의 내용을 가져옴.

print("mod1 모듈 add():",mod1.add(4,3))
print("mod1 모듈 sub():",mod1.sub(4,3))
print("mod2 모듈 add():",mod2.add(4,3))
print("mod2 모듈 sub():",mod2.sub(4,3))

from mod1 import add,sub
print(add(3,4))
print(sub(3,4))

import mod1 
import mod2 

print("mod1:",dir(mod1))
print("mod2:",dir(mod2))

print("mod1.__name__:",mod1.__name__)
print("__name__:",__name__)





