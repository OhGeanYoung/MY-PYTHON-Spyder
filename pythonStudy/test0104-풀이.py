# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:01:29 2022

@author: KITCOOP
test0104.py
"""
'''
1. 임의의 값으로 10*10 배열을 만들고, 전체 최소값과 최대값, 
 행별 최대값과 최소값, 열별 최대값과 최소값을 출력하기
  임의의 값이므로 결과에 표시된 숫자는 다름
[결과]
a 최대값 : 0.9957371929996585
a 최소값 : 0.013959842183549176
a 행별 최대값 : [0.85702828 0.99573719 ... 0.99251539 0.93112787]
a 행별 최소값 : [0.04483455 0.08025441 ... 0.27046588 0.05850934]
a 열별 최대값 : [0.93112787 0.91484578 ... 0.62135503 0.95952193]
a 열별 최소값 : [0.08247677 0.04483455 ... 0.05383681 0.05850934 0.09736856] 
'''
import numpy as np
rg=np.random.default_rng(2) #난수생성기 seed값. 데이터 복원시 필요 
a= rg.random((10,10)) #10행 10열 배열
a
print("a 최대값 :",a.max())
print("a 최소값 :",a.min())
print("a 행별 최대값 :",a.max(axis=1))
print("a 행별 최소값 :",a.min(axis=1))
print("a 열별 최대값 :",a.max(axis=0))
print("a 열별 최소값 :",a.min(axis=0))
'''
2. 임의의 값을 30개 저장하고 있는 배열을 만들고 평균값을 출력하기
  임의의 값이므로 결과에 표시된 숫자는 다름
[결과]
0.44769045640141436
'''
import numpy as np
a= np.random.random((30)) #seed값 설정 없음
a
a= np.random.random(30)
a
a.mean()

'''
3.  결과와 같은 값을 저장하고 있는 8*8 행렬을 생성하기
[결과]
[[0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]
 [0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]
 [0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]
 [0. 1. 0. 1. 0. 1. 0. 1.]
 [1. 0. 1. 0. 1. 0. 1. 0.]]

'''
Z = np.zeros((8,8))
Z
Z[1::2,::2] = 1
Z
Z[::2,1::2] = 1
print(Z)

'''
4. 0부터 10까지의 요소를 가진 배열을 생성하고
그중 3에서 8사이의 모든 요소를 음수인 값을 갖는
배열을 생성
[결과]
[ 0,  1,  2,  3, -4, -5, -6, -7,  8,  9, 10]
'''
a=np.arange(11) #0부터 10까지의 요소를 가진 배열
#a[4:8] = -a[4:8]
a[(3<a) & (a<8)] *= -1
a


'''
5. age.csv 파일에서 해당 지역의 인구비율과 전체지역의 인구 비율을 함께  
   그래프로 작성하기
'''
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
f =open('data/age.csv')
data = csv.reader(f)
next(data) 
data=list(data) 
name='역삼' 
homelist=[]
namelist=[] 
for row in data :
     if row[0].find(name) >= 0 :
        print(name,"===",row[0],":",row[0].find(name))
        row = list(map((lambda x:x.replace(",","")),row))
        homelist.append \
            (np.array(row[3:], dtype =int) / int(row[2]) *100) 
        import re
        namelist.append(re.sub('\(\d*\)', '', row[0]))
        alldata = np.array(row[3:], dtype =int) / int(row[2])*100

#전체 지역의 데이터 alldata에 저장
for row in data :
    row = list(map((lambda x:x.replace(",","")),row))
    away = np.array(row[3:], dtype = int) / int(row[2]) *100
    if np.isnan(away).any() :  #away 데이터셋에 한개라도NA값이 존재하면 
        continue    #반복문의 처음으로 
    alldata = np.vstack((alldata,away))  #행을 기준으로 연결

#연령별 인구수/전체인구수 평균값
alldata = alldata.mean(axis=0)  #열별 평균
plt.style.use('ggplot')
plt.figure(figsize = (10,5), dpi=100)
plt.rc('font', family ='Malgun Gothic')
for h,n in zip(homelist,namelist) :
    plt.plot(h,label=n) 
plt.plot(alldata, label="전체")
plt.xlabel("나이")    
plt.ylabel("비율(%)")    
plt.legend()
plt.show()

