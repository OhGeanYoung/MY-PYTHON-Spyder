# -*- coding: utf-8 -*-
'''
    print() : 화면 출력. 표준 출력.
        print('...',end="\n") => 한줄출력이 기본. end="" 옆쪽에 계속 출력 설정
        print(10,20,30) => ,로 여러개의 값 출력
        print("%d+%d" % (10,20)) => 형식화 문자 사용
    input('....') : 키보드에서 입력. 표준 입력. 문자열 값으로 리턴.
    int() : 정수형으로 변환. 형변환 함수
        int("숫자형값",진수) => 숫자형 값을 진수로 인식
        int("11",2) => 3
        int("10",2) => 2
    조건문 : if, if else, if elif else, 참인 값 if 조건 else 거짓의 값
    반복문 : for, while, break, continue
    문자열 : 인덱스로 접근 가능 0부터 시작하는 인덱스.
            문자열[시작인덱스:종료인덱스:증감값] => 시작인덱스부터 종료 인덱스-1까지 문자들
            함수들.
'''
'''
 collection : 여러개의 데이터를 저장 할 수 있는 객체
  list(리스트) : 배열의 형태. 인덱스 사용 가능. [ ]
  tuple(튜플) : 상수화된 리스트. (  )
  dictionary(딕셔너리) : (key,value) 쌍인 객체들 { }
  set(셋) : 중복 불가.           { } 한개 
'''

a=[0,0,0,0]
b=[]
print(a,len(a))
print(b,len(b))
#a 길이 만큼 숫자를 입력받아, 입력받은 수의 전체 합계 출력하기
#입력받은 숫자를 b리스트에 저장하고, 합계 출력하기

hapa = 0
for i in range(len(a)) :
    a[i] = int(input(str(i+1)+"번째 숫자 입력: "))
    # b[i] = a[i] 오류발생
    b.append(a[i]) #b리스트에 값을 추가.
    hapa += a[i]
print('a=',a,', 요소의 합:',hapa)
print('a=',a,', 요소의 합:',sum(a))
print('b=',b,', 요소의 합:',sum(b))

mylist = [30,20,10]
print(mylist)
# 형식문자를 이용하여 출력
print("mylist=%s" % (mylist))
# mylist 에 40 값 추가하기
mylist.append(40)
print(mylist)
# sort() : 정렬하기
mylist.sort()
print(mylist)
# 마지막 데이터 조회하기
print(mylist[len(mylist)-1])
print(mylist[-1])
#pop () : 마지막 데이터를 제거하면서, 조회하기
print("pop() 결과값=%s" % mylist.pop())
print(mylist)

#reverse() : 요소들을 역순으로 처리
mylist.reverse()
print(mylist)

# index(값) : 리스트에서 값의 위치값 리턴
# find() : 리스트에서는 사용 안됨
print("mylist 에서 20의 위치",mylist.index(20))
#print("mylist에서 20의 위치:", mylist.find(20))
# 요소가 없는 경우 오류 발생.
print("mylist 에서 40의 위치:",mylist.index(40))

# 중간에 요소 추가하기
mylist.insert(2, 222)
print(mylist)
# 마지막 요소 추가하기
mylist.append(333)
print(mylist)

# 요소 제거하기
mylist.remove(222)
print(mylist)

# extend() :  다른 리스트를 추가하기
mylist.extend([50,60,70])
print(mylist)

# count() : 요소의 개수 조회

print("30요소의 개수:",mylist.count(30))
print("300요소의 개수:",mylist.count(300))

# 문자열을 분리하여 리스트로 저장하기
date ='2021/12/21'
d = date.split("/")
print(d)

# date보다 10년 전의 일자를 출력하기
print(int(d[0])-10,end="/")
print(d[1],end="/")
print(d[2])

#d : 일,월, 년으로 출력하기
d.reverse()
print(d)
print(d[::-1]) # 출력만 역순으로
print(d)

#문제 : ss 문자열의 모든 숫자들의 합을 출력하기
ss="10,20,50,60,30,40,50,60,30"
slist = ss.split(",")
slist
print(slist)
# print(slist.sum()) slist가 문자열이기 때문에 합계 불가
hap = 0
for n in slist :
    hap += int(n)
print("합:",hap)

# map 함수 : JAVA에서와는 완전히 다른 의미 
# list 객체의 요소들의 적용되는 함수를 설정하는 함수
print("합:",sum(list(map(int,slist))))

mlist = list(map(int,slist))
print(mlist)
print(sum(mlist))

#dictionary : {(키,값)}
member_dic = {'lee':100,'hong':70,'kim':90}
print(member_dic)
#'hong' 값 출력하기
print(member_dic['hong'])
# 자료형 출력하기
print(type(member_dic))

#값 추가하기
member_dic['park'] = 80 #park 키값이 없는 경우 추가
print(member_dic)

#값 수정하기
member_dic['park'] = 85 #park 키값이 있는 경우 수정
print(member_dic)

#값 제거하기
del member_dic['park']
print(member_dic)

# 키들만 조회하기
print(member_dic.keys())
# 값들만 조회하기
print(member_dic.values())
# 키,값 쌍 조회하기
print(member_dic.items())

singer = {} #비어있는 dictionary 객체
singer['이름'] = '트와이스'
singer['인원수'] = 9
singer['소속사'] = 'JYP'
print(singer)
for i in singer.keys() :
    print("%s=>%s" % (i,singer[i]))

#키와값의 쌍인 객체 출력하기
print(singer.items())
for i in singer.items():
#    print(i)
     print("%s=>%s" % (i[0],i[1]))
     
'''
문제 : 궁합음식의 키를 입력받아 해당되는 음식을 출력하기
[예시]
음식 입력 : 라면
라면의 궁합음식은 김치입니다. 라면=>김치
음식 입력 : 불고기
등록된음식이 아닙니다.
음식 입력 :종료
프로그램을 종료합니다.
등록된음식: 
    떡볶이:오뎅
    짜장면:단무지
    라면:김치
    맥주:치킨
    
2. 등록된 음식이 아닌경우
    등록하시겠습니까(y/n)?
    y입력 : foods 객체에 추가 :
        궁합음식 입력받아서 추가함
    y가 아닌경우 :
        음식을 다시 입력하기
'''
foods ={"떡볶이":"오뎅","짜장면":"단무지","라면":"김치","맥주":"치킨"}
'''
if "떡볶이" in foods: # 떡볶이 foods 객체에 키로 존재?
    print("떡볶이는 어묵 입니다")
if "어묵" in foods:
    print("떡볶이는 어묵 입니다")
name=str(input("음식 입력"))
if name in foods :
    print("%3s는 %3s 입니다" % foods)
else :
    print("등록된 음식이 아닙니다.")
if name = "종료" :
    print("프로그램을 종료합니다.")
    print(foods.items())
    for i in foods.items():
        print(i)
'''
while True :
    myfood = input("음식 입력 :")
    if myfood == '종료' :
        break
    if myfood in foods :
        print("%s의 궁합음식은 %s 입니다." % (myfood,foods[myfood]))
    else :
        print("등록된 음식이 아닙니다.?")
        yn = input("등록하시겠습니까(y/n)?")
        if yn == 'y' :
            f = input(myfood + "의 궁합 음식 입력:")
            foods[myfood] = f

print("프로그램을 종료합니다.")
print("등록된 음식 : ")
for k in foods.keys() :
    print("%s:%s" % (k,foods[k]))
    
#튜플 : 상수화된 리스트 ()
tp1 = (10,20,30)
print(tp1)
tp1.append(40) #튜플에 요소 추가 안됨. 리스트로 변경 후 추가 가능

list1 = list(tp1)
list1.append(40)
print(list1)
tp1=tuple(list1)
print(tp1)

#요소값 변경
tp1[0]=50

#tp1의 첫번째 요소를 50으로 변경하기
list1 = list(tp1)
list1[0]=50
tp1=tuple(list1)
print(tp1)

#첫번째, 두번째 요소만 조회하기
print(tp1[:2])
#요소를 역순으로 조회하기
#tp1.reverse() 불가능
list1 = list(tp1)
list1.reverse()
print(list1)
print(tp1[::-1])

a,b,c,d = tp1 #tp1의 요소의 개수와 변수의 개수가 동일해야함
print(a,b,c,d)

# a,b,c = tp1 #tp1의 요소의 개수와 변수의 개수가 동일해야함

# foods 딕셔너리의 (키,값) 이 튜플객체임

for k,v in foods.items() :
    print(k,":",v)
    
# set : 중복 불가. 집합을 표현하는 객체

set1 = {30,10,20,10}
print(set1) #10 요소는 한개만 저장. 순서지정 안됨.
print(set1[0]) #인덱스 사용 안됨.

set1 = {1,2,3,4,5}
set2 = {1,2,3,4,5,1,2,3,4,5}
print(set1)
print(set2)
set3 = {5,6,7,8}
print("set1과 set2의 교집합 : " , set1 & set3)
print("set1과 set3의 교집합 : " , set1 & set2)
print("set1과 set3의 교집합 : " , set1.intersection(set3)) # 교집합 함수
print("set1과 set2의 합집합 : " , set1 | set3)
print("set1과 set3의 합집합 : " , set1 | set2)
print("set1과 set3의 합집합 : " , set1.union(set3)) # 교집합 함수

#comprehension(컴프리헨션) : 패턴이 있는 list,dictionary.set을 생성 방법
numbers=[]
#numbers 1 ~ 10 까지의 데이터 저장하기
for n in range(1,11) :
    numbers.append(n)
print(numbers) 

# comprehension 방식
numbers=[x for x in range(1,11)]
print(numbers)
# 2부터 20까지의 짝수들을 numbers 리스트에 저장하기
numbers=[]
for n in range(2,21,+2) :
    numbers.append(n)
#numbers=[x*2 for x in range(1,11)]
print(numbers)

# 2부터 10까지의 짝수들을 evens리스트에 저장하기
evens=[x for x in range(2,11,2)]
print(evens)

evens=[x for x in range(1,11) if x%2==0]
print(evens)

#1 ~ 10까지의 수중 2의 배수와 3의 배수만 가지고 있는 nums 리스트 생성
nums=[n for n in range(1,11) if (n%2==0) or (n%3==0)]
print(nums)

#두개의 리스트 데이터를 각각 한개씩
colorlist = ['black','white']
sizelist=['S','M','L']
dresslist=list((c,s) for c in colorlist for s in sizelist)
print(dresslist)

#set 객체 생성하기
# 1~10 사이의 짝수의 제곱으로 이루어진 set 객체 생성하기
set1 = {x**2 for x in range(1,11) if x%2==0}
print(set1)

#dictionary 데이터 생성하기
products = {"냉장고":220,"건조기":140,"TV":130,"세탁기":150,"오디오":50,"컴퓨터":250}
#200만원 미만의 제품만 product1 객체에 저장하기
products1 = {}
for k in products :
    if products[k] < 200:
        products[k] = products[k]

product1 = {}
for k,v in products.items() :
    if v < 200:
        product1[k] =v
print(product1)

#comprehension 방식
product2 = {p:v for p,v in products.items() if v < 200}
print(product2)

# 함수와 람다.
# 함수 정의 : def 예약어 사용.

def func1() :
    print("func() 함수 호출")
    return 10

a = func1()
print(a)

#전역변수 : 모든  함수에서 접근이 가능한 변수. 함수 외부에 선언됨.
#지역변수 : 변수가 사용된 함수에서만 접근이 가능한 변수. 함수 내부에서 선언됨.

def func2() :
    a = 20
    b = 30
    print("func2() 함수 : ",gval, a,b)
    
gval = 100 #전역변수
a = 10     #전역변수

func2()
print("main 함수 : ",gval,a)
print(b)

#전역변수 값을 변경하기
def func3() :
    global gval #gval 변수는 전역변수의 gval임
    gval=200
    a=20
    b=30
    print("func3()함수.",gval,a,b) #200 20 30
    
print("1.main 함수 : ",gval,a) #100 10
func3()                        #200 20 30
print("2.main 함수 : ",gval,a) #100 10


#매개변수
def add1(v1,v2) :
    return v1+v1
def sub1(v1,v2) :
    return v1-v2

hap = add1(10,20)
sub = sub1(10,20)
print(hap)
print(sub)
hap = add1(10.5,20.3)
print(hap)
hap = add1("test","python")
print(hap)

#리턴값이 두개인 경우 : 리스트값을 리턴
def multi(v1,v2) :
    list1=[]
    list1.append(v1+v2)
    list1.append(v1-v2)
    return list1
list1 = multi(200,100)
print(list1)

#가변매개변수 : 매개변수의 개수 정해지지 않은 경우
def multiparam(* p) : #매개변수의 개수가 0개 이상
    result = 0
    for i in p :
        result += i
    return result

print("multiparam()=",multiparam())
print("multiparam(10)=",multiparam(10))
print("multiparam(10,20)=",multiparam(10,20))
print("multiparam(10,20,30)=",multiparam(10,20,30))

#매개변수 기본값 설정하기
def hap1(num1=0,num2=1) :
    return num1+num2

print("hap1()=",hap1()) # num1=0,num2=1 => 1
print("hap1(10)=",hap1(10)) # num1=10,num2=1 => 11
print("hap1(10,20)=",hap1(10,20)) # num1=10,num2=20 => 30
print("hap1(10,20,30)=",hap1(10,20,30)) # 오류 발생. 매개변수는 2개까지만 가능



def getSum(numlist) :
    return sum(numlist)

def getMean(numlist) :
    return sum(numlist)/len(numlist) if len(numlist) > 0 else 0

list1=[2,3,3,4,4,5,5,6,6,8,8] 
print("list1 의 합:",getSum(list1))
print("list1 의 평균:",getMean(list1))

list2=[]
print("list2 의 합:",getSum(list2))
print("list2 의 평균:",getMean(list2)) #내용이 없는 경우 0















