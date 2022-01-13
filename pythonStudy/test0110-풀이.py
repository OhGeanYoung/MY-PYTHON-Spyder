# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:57:09 2022

@author: KITCOOP
test0110-풀이.py
"""
'''
1.
  경찰서별 전체 범죄 검거율 컬럼생성하고, 검거율별로 수평막대그래프를 출력하기
  검거 =   (살인,강도,강간,절도,폭력)검거합
  발생 =   (살인,강도,강간,절도,폭력)발생합
  검거율 = 검거 / 발생 * 100
'''
import numpy as np
import pandas as pd
crime_Seoul = pd.read_csv('data/02. crime_in_Seoul.csv',
                          thousands=',', encoding='euc-kr')
crime_Seoul.head()
crime_Seoul["발생"]=crime_Seoul["살인 발생"] + \
                    crime_Seoul["강도 발생"] + \
                    crime_Seoul["강간 발생"] + \
                    crime_Seoul["절도 발생"] + \
                    crime_Seoul["폭력 발생"]
crime_Seoul["검거"]=crime_Seoul["살인 검거"] + \
                    crime_Seoul["강도 검거"] + \
                    crime_Seoul["강간 검거"] + \
                    crime_Seoul["절도 검거"] + \
                    crime_Seoul["폭력 검거"]
crime_Seoul["검거율"] = crime_Seoul["검거"]/crime_Seoul["발생"] * 100
crime_Seoul.info()
crime_Seoul.set_index("관서명",inplace=True)
import matplotlib.pyplot as plt
plt.rc('font', family ='Malgun Gothic')
plt.figure()
crime_Seoul['검거율'].sort_values().\
    plot(kind='barh', grid=True, figsize=(10,10))
plt.title("서울 경찰서별 전체 범죄 검거율")
plt.xlabel('검거율(%)')
plt.ylabel('경찰서명')
plt.show()

'''
2. 경찰서별 범죄발생건수, CCTV 갯수를 산점도와 회귀선으로 출력하기.
   단 CCTV의 갯수는 구별로 지정한다.
'''
import numpy as np
import pandas as pd
crime_Seoul = pd.read_csv('data/02. crime_in_Seoul.csv',
                          thousands=',', encoding='euc-kr')
crime_Seoul["발생"]=crime_Seoul["살인 발생"] + \
                    crime_Seoul["강도 발생"] + \
                    crime_Seoul["강간 발생"] + \
                    crime_Seoul["절도 발생"] + \
                    crime_Seoul["폭력 발생"]
crime_Seoul["검거"]=crime_Seoul["살인 검거"] + \
                    crime_Seoul["강도 검거"] + \
                    crime_Seoul["강간 검거"] + \
                    crime_Seoul["절도 검거"] + \
                    crime_Seoul["폭력 검거"]
crime_Seoul["검거율"] = crime_Seoul["검거"]/crime_Seoul["발생"] * 100

police_state = pd.read_csv('data/경찰관서 위치.csv', encoding='euc-kr')
police_Seoul = police_state[police_state["지방청"]=='서울청']

police_Seoul["구별"] = police_Seoul["주소"].apply(lambda x : str(x).split()[1])
police_Seoul["구별"]
police_Seoul["관서명"] = police_Seoul["경찰서"].apply((lambda x : str(x[2:]+'서' )))
police_Seoul = police_Seoul.drop_duplicates(subset=['관서명'])
del police_Seoul["지방청"],police_Seoul["경찰서"],police_Seoul["구분"],police_Seoul["주소"]
police_Seoul.info()
police_Seoul.head()
crime_Seoul = pd.merge(crime_Seoul,police_Seoul,on="관서명")
crime_Seoul.info()
CCTV_Seoul = pd.read_csv('data/01. CCTV_in_Seoul.csv',  encoding='utf-8')
CCTV_Seoul.head()
CCTV_Seoul.rename(columns={"기관명" : '구별'}, inplace=True)
del CCTV_Seoul["2013년도 이전"]
del CCTV_Seoul["2014년"]
del CCTV_Seoul["2015년"]
del CCTV_Seoul["2016년"]
CCTV_Seoul.info()
data_result = pd.merge(CCTV_Seoul,crime_Seoul,on="구별")
data_result
#data_result의 index를  관서명으로 변경하기
data_result.set_index("관서명",inplace=True)
#data_result["소계"] : cctv 갯수
#data_result["발생"] : 전체 범죄 발생 건수 
# ax+b : a,b값 리턴
fp1 = np.polyfit(data_result["소계"],data_result["발생"],1)
fx = np.linspace(500, 3000, 100) #500 ~ 3000까지의 데이터를 100개로 균등분할
f1 = np.poly1d(fp1) #함수 : ax+b

data_result["오차"]=np.abs(data_result["발생"]-f1(data_result["소계"]))
#df_sort : data_result 데이터를 오차값의 내림차순으로 정렬
df_sort = data_result.sort_values(by="오차", ascending=False)
plt.figure(figsize=(14,10))
plt.scatter(df_sort['소계'],df_sort["발생"],c=df_sort["오차"],s=50)
plt.plot(fx,f1(fx),ls="dashed",lw=3,color='g') #회귀선그래프
#오차가 큰 10개 관서이름을 그래프에 출력   
for n in range(10):
    plt.text(df_sort['소계'][n]*1.02, df_sort['발생'][n]*0.997, 
             df_sort.index[n], fontsize=15)
plt.xlabel('CCTV 갯수')
plt.ylabel('범죄발생건수')
plt.title('범죄발생과 CCTV 분석')
plt.colorbar()
plt.grid()
plt.show()

#상관계수
data_result[["소계","발생"]].corr()
