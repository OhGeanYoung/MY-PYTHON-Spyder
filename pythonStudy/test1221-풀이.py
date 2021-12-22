# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:44:51 2021

@author: KITCOOP
test1221-풀이.py
"""
'''
1.  화면에서 한개의 문자를 입력받아
     대문자인 경우는 소문자로, 
     소문자인 경우는 대문자로 
     숫자인 경우는 20을 더한 값을   출력하기
[예시]
한개의 문자를 입력하세요 : A
A 문자의 소문자는  a

한개의 문자를 입력하세요 : a
a 문자의 대문자는  A

한개의 문자를 입력하세요 : 1
1 + 20 = 21
'''
ch = input("한개의 문자를 입력하세요 : ")
if ch.isdigit() : #숫자?
   print("%s + 20 = %d" % (ch,int(ch) + 20))
elif ch.isupper() : #대문자?
   print("%c 문자의 소문자는  %c" % (ch,ch.lower()))
elif ch.islower() : #소문자?
   print("%c 문자의 대문자는  %c" % (ch,ch.upper()))

'''
2 (1)+(1+2)+(1+2+3)+... (1+2+3+...10)=220 출력하기
'''
hap=0
for i in range(1,11) : #1~10
    for j in range (1,i+1) : #1부터 i값까지 
        hap += j
print(hap)        


hap=0
for i in range (1,11) :
    print("(",end="")
    for j in range (1,i+1) :
        print(j, end="")
        if(j!=i) :
            print("+",end="")
        hap += j
    print(")",end="")
    print("=" if i==10 else "+", end="") #조건연산자.
print(hap)    

'''
3. 화면에서 자연수를 입력받아 각각의 자리수의 합을 구하기.

 [결과]
 자연수를 입력하세요 : 12345
 12345 자리수의 합: 15
'''
#1
num = int(input("자연수를 입력하세요 : ")) #12345
hap = 0 #자리수 합
tmp = num 
while(tmp > 0) : 
    hap += tmp % 10  #5+4+3+2+1
    tmp //= 10
print(num,"자리수의 합:",hap)

#2
num = input("자연수를 입력하세요 : ") #문자열형태
hap = 0  
tmp = num 
for i in range(0,len(tmp)) :
    hap += int(tmp[i]) #1+2+3+4+5
print(num,"자리수의 합:",hap)

#3
num = input("자연수를 입력하세요 : ")
nums = [int(i) for i in num] #입력받은 자연수 문자열의 각각의 자리수를 정
print(num,"자리수의 합:",sum(nums))

'''
4. aa,bb 리스트를 생성하고, 
  aa 리스트는 0부터 짝수 100개를 저장하고
  bb 리스트는 aa 배열의 역순으로 값을 저장하기.
  aa[0] ~ aa[9], bb[99]~bb[90] 값을 출력하기
'''   
#1
aa = []
bb = []
value = 0
for i in range(0,100) : 
    aa.append(value)
    value += 2
for i in range(0,len(aa)) :
    bb.append(aa[len(aa)-1-i])

print(aa[:10])
print(bb[99:89:-1])

#2 컴프리헨션 방식    
aa = [a*2 for a in range(0,100)]
bb = aa[::-1]    
print(aa[:10])
print(bb[99:89:-1])

'''
5. 다음 모레시계모양을 출력하기
모래시계의 높이를 홀수로 입력하세요 : 7
*******
 *****
  ***
   *
  ***
 *****
*******
'''    
row = int(input("모래시계의 높이를 홀수로 입력하세요 : ")) #5
for i in range(0, row//2+1) :    #0 ~ 2
    for j in range(0, row - i ) :
         if j < i :
             print(" ",end="")
         else :   
             print("*",end="")
    print() 
for i in range(row//2+1,row) : #3~4
    for j in range(0,i+1) :
        if(j >= row-i-1 and j<= i) : #*출력위치
            print("*", end="")
        else :
            print(" ",end="")
    print()   

'''
6. 피보나치 수열 출력하기
   피보나치 수열은 0,1로 시작하고
   앞의 두수를 더하여 새로운 수를 만들어 주는 수열을 의미한다.
   피보나치 수열의 갯수를 입력받아 피보나치 수열을 갯수만큼 저장한
   리스트를 생성하는 함수 fibo 함수를 작성하기

[결과]
피보나치 수열의 요소 갯수를 입력하세요(3이상의 값) :10
fibo( 10 )=[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]   

'''
def fibo(n) :
    fibolist = [0,1] 
    num1 = 0
    num2 = 1
    num3 = num1+num2
    fibolist.append(num3) 
    for i in range(3,n) :
        num1 = num2
        num2 = num3
        num3 = num1 + num2
        fibolist.append(num3)
    return fibolist 

num = int(input("피보나치 수열의 요소 갯수를 입력하세요(3이상의 값) :"))
print("fibo(",num,")=",end="")
print(fibo(num))

