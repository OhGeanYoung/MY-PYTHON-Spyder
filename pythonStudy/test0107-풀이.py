# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:46:19 2022

@author: KITCOOP
test0107-풀이.py
"""
'''
0.total_servings 전체 술소비량을 막대그래프로 작성하고,
 대한민국의 위치를 빨강색으로 표시하기
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family ='Malgun Gothic')

drinks = pd.read_csv("data/drinks.csv")

drinks["total_servings"] = drinks["beer_servings"] +\
     drinks["spirit_servings"] +drinks["wine_servings"]

country_serving_rank = drinks[['country', 'total_servings']]
country_serving_rank = country_serving_rank.sort_values\
                   (by=['total_servings'], ascending=0)
country_serving_rank.head()                   
country_list = country_serving_rank.country.tolist()
country_list
x_pos = np.arange(len(country_list))
rank = country_serving_rank.total_servings.tolist()
bar_list = plt.bar(x_pos, rank)
#한국에 해당되는 막대그래프의 색을 빨강색으로 변경
bar_list[country_list.index("South Korea")].set_color('r')
plt.ylabel('total servings')
plt.title('drink servings rank by contry')
plt.axis([0, 200, 0, 700])
korea_rank = country_list.index("South Korea")
print(korea_rank)
country_serving_rank[country_serving_rank['country']=='South Korea']\
    ['total_servings']
korea_serving_rate = country_serving_rank\
[country_serving_rank['country']=='South Korea']\
    ['total_servings'].values[0]
korea_serving_rate    
plt.annotate('South Korea : ' + str(korea_rank + 1)+"번째", 
          xy=(korea_rank, korea_serving_rate), 
          xytext=(korea_rank + 10, korea_serving_rate + 0.05),
          arrowprops=dict(facecolor='red', shrink=0.05))
plt.show()


'''
1. drinks.csv 데이터를 읽고,대륙별 술 소비량 대비 알콜 비율 컬럼
    (alcohol_rate_continent) 구하기
    alcohol_rate_continent =
 대륙별total_litres_of_pure_alcohol/대륙별전체술소비량  
   으로 계산한다.

[결과]
  country continent alcohol_rate_continent
0 Afghanistan AS    0.020293
1 Bahrain       AS    0.020293
2 Bangladesh  AS    0.020293
3 Bhutan        AS    0.020293
4 Brunei         AS    0.020293
'''
import pandas as pd
drinks = pd.read_csv("data/drinks.csv")
# 국가별 전체 술 소비량.
drinks["total_servings"] = drinks["beer_servings"] +\
    drinks["spirit_servings"] + drinks["wine_servings"]
continent_sum = drinks.groupby('continent').sum()
continent_sum
# alcohol_rate_continent 컬럼 추가
continent_sum['alcohol_rate_continent'] = \
   continent_sum['total_litres_of_pure_alcohol']  \
                 / continent_sum['total_servings']
continent_sum
continent_sum.info()
# 현재의 index를 컬럼(피처)으로 변경
continent_sum = continent_sum.reset_index()
continent_sum
continent_sum.info()
# 대륙컬럼, 대륙별 알콜비율
continent_sum = \
    continent_sum[['continent', 'alcohol_rate_continent']]
continent_sum
drinks = pd.merge\
    (drinks, continent_sum, on='continent', how='outer')
drinks.info()    

drinks[['country', 'continent', "total_servings",'alcohol_rate_continent']].head()
drinks[['country', 'continent', "total_servings", 'alcohol_rate_continent']].tail()
#무작위로 5개 추출
drinks[['country', 'continent', 'alcohol_rate_continent']].sample(5)

'''
 2. 전체 평균보다 적은 알코올을 섭취하는 대륙 중에서, 
    spirit을 가장 많이 마시는 국가 구하기
[결과]
Russian Federation
'''
import pandas as pd
drinks = pd.read_csv("data/drinks.csv")

#spirit 소비량순으로 정렬
drinks.sort_values(by=['spirit_servings'],ascending=False)\
    [["country","continent","spirit_servings"]]
    
    
#전체 알콜 섭취량의 평균
total_mean = drinks.total_litres_of_pure_alcohol.mean()
total_mean
#대륙별 알콜 섭취량의 평균
continent_mean = drinks.groupby('continent').total_litres_of_pure_alcohol.mean()
continent_mean
#전체 알콜 섭취량의 평균 보다 대륙별 알콜 섭취량의 평균이 작은 대륙정보 조회
continent_under_mean = continent_mean[continent_mean < total_mean]
continent_under_mean
# 대륙 정보만 리스트로 저장.
continent_under_mean = continent_under_mean.index.tolist()
continent_under_mean

# AF,AS,OC 대륙에 속한 국가들 중 spirit_servings의 값이 가장 큰 국가 조회하기
# df_continent_under_mean : 레코드들 중 대륙정보가 
#                            continent_under_mean에 속한 레코드만 저장
df_continent_under_mean = \
  drinks.loc[drinks.continent.isin(continent_under_mean)]
df_continent_under_mean
# df_continent_under_mean 데이터 중 spirit_servings값이 가장 큰 국가 조회
# df_continent_under_mean['spirit_servings'].idxmax() :
#  df_continent_under_mean 중  가장 큰 spirit_servings의 값을 가진 인덱스값리턴
max_spirit = df_continent_under_mean['spirit_servings'].idxmax()
df_continent_under_mean_max_spirit = df_continent_under_mean.loc[max_spirit]
df_continent_under_mean_max_spirit["country"]
