# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 09:01:39 2022

@author: KITCOOP
20220107.py
"""
import pandas as pd
chipo = pd.read_csv("data/chipotle.tsv", sep = '\t')
chipo.describe()
chipo.info()
chipo.head()
# order_id 컬럼의 자료형 문자열로
chipo["order_id"] = chipo["order_id"].astype(str)
chipo.describe()
chipo.info()
# item_price 컬럼의 자료형은 실수형으로 변경.
#chipo["item_price"] = chipo["item_price"].str.replace("$","").astype(float)
#chipo.info()
#apply(함수이름|람다식) : chipo['item_price'] 컬럼의 각각의 요소들의 함수 적용
#                                                           $12.3
chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x[1:]))
chipo.info()
'''
주문당 평균 주문금액 조회하기
  주문번호  주문금액
     1        1
     1        3
     2        1
     2        1
   전체주문금액 : 6
   주문건수     : 2
   주문당평균금액 : 3
'''
#전체주문금액
hap = chipo["item_price"].sum()
hap
#주문건수 
len(chipo.groupby("order_id")["item_price"].count())
cnt = len(chipo.groupby("order_id")["item_price"].count())
cnt
#주문당평균금액
hap/cnt

chipo.groupby("order_id")["item_price"].sum().mean()

#한번의 주문시 주문 금액이 50달러 이상인 주문id 출력하기
order_id_tot = chipo.groupby("order_id").sum()
result = order_id_tot[order_id_tot['item_price'] >= 50]
result.index
print("50달러 이상 주문 번호:",result.index.values)
print("50달러 이상 주문 건수:",len(result.index.values))

#데이터 출력시 5개컬럼까지 조회하도록 설정
pd.set_option("display.max_columns",5)
#한번의 주문시 주문 금액이 50달러 이상인 주문 정보 출력하기
chipo.loc[chipo["order_id"].isin(result.index.values)]

#item_name 별 단가를 조회하기
price_one = chipo.groupby("item_name").min()["item_price"]
price_one
# 단가의 분포를 히스토그램으로 출력하기
import matplotlib.pyplot as plt
plt.hist(price_one)
plt.ylabel("counts")
plt.title("Histogram of item price")
plt.show()

#item 중 단가가 높은 item 10개 출력하기
price_one.sort_values(ascending=False)[:10]

#주문번호별 가격이 가장 비싼 5건의 주문의 총수량을 출력하기
price5 = chipo.groupby("order_id").sum().\
    sort_values(by='item_price',ascending=False)[:5]
price5
price5["quantity"].sum()

# Veggie Salad Bowl 몇번 주문되었는지 출력하기
chipo_salad = chipo[chipo["item_name"]=='Veggie Salad Bowl']
len(chipo_salad)
len(chipo_salad.groupby("order_id").sum())

# Veggie Salad Bowl 주문 수량 출력하기
chipo_salad["quantity"].sum()

# 전세계 음주 데이터 분석하기 : drinks.csv 파일 분석
# 컬럼=피처=변수 
import pandas as pd
drinks = pd.read_csv("data/drinks.csv")
drinks.info()
drinks.shape
# 맥주,와인 상관관계 조회하기
# corr 함수 : 상관계수 리턴 (pearson,kendall,spearman)
beer_wine_corr = drinks[["beer_servings","wine_servings"]].corr(method="kendall")
beer_wine_corr
# 맥주,와인,spirit,alcohol 상관계수 조회하기
cols = ["beer_servings","wine_servings","spirit_servings","total_litres_of_pure_alcohol"]
cols = drinks.columns[1:-1]
corr = drinks[cols].corr()
corr
# 상관 계수 시각화 하기
# corr 행렬 히트맵(seaborn 모듈)을 시각화합니다.
import matplotlib.pyplot as plt
import seaborn as sns
cols_view = ['beer', 'spirit', 'wine', 'alcohol'] # 그래프 출력을 위한 cols 이름을 축약.
sns.set(font_scale=1.5)
hm = sns.heatmap(corr.values, #데이터 값
            cbar=True,        #컬러바 표시 
            annot=True,       #데이터값 표시
            square=True,      # 히트맵모양을 사각형으로 표현
            fmt='.2f',        #소숫점 이하 2자리까지 표현
            annot_kws={'size': 15}, #데이터값 표시 시 크기 지정 
            yticklabels=cols_view,  #y축에 표시될 라벨
            xticklabels=cols_view)  #y축에 표시될 라벨
plt.show()

# 피처간의 scatter plot 출력
sns.pairplot(drinks[cols],height=2.5)
plt.show()
# 각 피처의 결측값 조회하기
drinks.isnull().sum()

# fillna 함수 :대륙정보의 결측값 OT 문자열로 변경하기
drinks["continent"] = drinks["continent"].fillna("OT")
drinks.info()
# 대륙별 국가수 출력하기
drinks["continent"].value_counts() #건수의 역순으로 정렬 조회
drinks.groupby("continent").count()["country"] #대륙순으로 정렬조회

# 대륙별 국가의 갯수를 파이그래프로 출력하기
import matplotlib.pyplot as plt
labels = drinks['continent'].value_counts().index.tolist()
explode = (0, 0, 0, 0.1, 0, 0) #조각을 밖으로 표시 
#autopct='%.0f%%' : 비율을 %로 표시. %% : %문자표시
plt.pie(drinks['continent'].value_counts(),labels=labels,\
        autopct='%.0f%%', explode=explode,shadow=True)
plt.title('null data to \'OT\'')
plt.show()

# 대륙별 spirit_servings의 평균, 최소, 최대, 합계를 출력.
print("평균",drinks.groupby('continent').spirit_servings.mean())
print("최소",drinks.groupby('continent').spirit_servings.min())
print("최대", drinks.groupby('continent').spirit_servings.max())
print("합계",drinks.groupby('continent').spirit_servings.sum())

result = drinks.groupby('continent').\
    spirit_servings.agg(["mean","min","max","sum"])
result
# 대륙별 spirit_servings의 평균, 최소, 최대, 합계를 시각화.
import numpy as np
n_groups = len(result.index) 
means = result['mean'].tolist() #평균데이터들
mins = result['min'].tolist()   #최소데이터들
maxs = result['max'].tolist()
sums = result['sum'].tolist()
index = np.arange(n_groups) # 숫자형 배열 
index
bar_width = 0.1
rects1 = plt.bar(index, means, bar_width, color='r',label='Mean')
rects2 = plt.bar(index + bar_width, mins, bar_width,color='g',label='Min')
rects3 = plt.bar(index + bar_width * 2, maxs, bar_width,color='b', label='Max')
rects4 = plt.bar(index + bar_width * 3, sums, bar_width, color='y', label='Sum')
plt.xticks(index, result.index.tolist()) #x축의 레이블 변경 
plt.legend(loc="best")
plt.show()

# total_litres_of_pure_alcohol : 알콜량
# 알콜량이 전체 평균보다 많은 대륙을 출력하기
total_mean = drinks.total_litres_of_pure_alcohol.mean() #전체 알콜평균
total_mean
continent_mean = drinks.groupby('continent')['total_litres_of_pure_alcohol'].mean()
continent_mean  
continent_over_mean = continent_mean[continent_mean > total_mean]
print(continent_over_mean)
print(continent_over_mean.index.values)

# 평균 beer_servings이 가장 높은 대륙을 출력
cont_beer_mean = drinks.groupby('continent').beer_servings.mean()
cont_beer_mean
type(cont_beer_mean)
drinks.groupby('continent').beer_servings.mean().idxmax()
# 평균 beer_servings이 가장 적은 대륙을 출력
drinks.groupby('continent').beer_servings.mean().idxmin()

# 대륙별 total_litres_of_pure_alcohol 섭취량 평균 을 시각화
# 대륙별 알콜섭취량 평균
continent_mean=drinks.groupby('continent')['total_litres_of_pure_alcohol'].mean()
continent_mean
# 전체 알콜섭취량 평균
total_mean = drinks.total_litres_of_pure_alcohol.mean()
total_mean

continents = continent_mean.index.tolist() #대륙 정보 리스트 
continents.append("Mean") #Mean 추가
x_pos = np.arange(len(continents)) #0~ 6까지 배열값
alcohol = continent_mean.tolist() #대륙별 알콜평균 값 
alcohol 
total_mean = drinks.total_litres_of_pure_alcohol.mean()
total_mean
alcohol.append(total_mean) #전체평균추가
#막대그래프
#bar_list : 막대그래프 목록
bar_list = plt.bar(x_pos, alcohol, align='center',alpha=0.5)
bar_list
#bar_list[len(continents) - 1] : 마지막 막대그래프. 평균표시막대
bar_list[len(continents) - 1].set_color('r') #색상:r 빨강색
#"k--" : k:검정색, --:점선
plt.plot([0., 6], [total_mean, total_mean], "k--") #선그래프. 
plt.xticks(x_pos, continents) #x축레이블 변경
plt.ylabel('total_litres_of_pure_alcohol')
plt.title('total_litres_of_pure_alcohol by Continent')
plt.show()
'''
대륙별 beer_servings의 평균를 막대그래프로 시각화
가장 많은 맥주를 소비하는 대륙(EU)의 막대의 색상을 빨강색으로 변경하기 
전체 맥주 소비량 평균을 구해서 막대그래프에 추가
평균선을 출력하기. 막대 색상은 노랑색 
평균선은 검정색("k--")
'''
#대륙별 맥주 평균소비량
beer_mean = drinks.groupby("continent")["beer_servings"].mean()
beer_mean
total_mean = drinks.beer_servings.mean() #전체맥주 소비량 평균
total_mean
continents = beer_mean.index.tolist()
continents.append("Mean")
x_pos = np.arange(len(continents))
beer = beer_mean.tolist()
beer.append(total_mean)
beer
beer_mean.idxmax()
beer_mean.argmax()
bar_list = plt.bar(x_pos, beer, align='center',alpha=0.5)#막대그래프 출력
bar_list[beer_mean.argmax()].set_color("r")  #EU의 막대그래프의 색을 빨강색
#Mean의 막대그래프의 색을 노랑색
bar_list[continents.index("Mean")].set_color("y")
plt.xticks(x_pos, continents) #x축에 대륙코드설정
plt.plot([0.,6],[total_mean,total_mean],'k--') #평균선 
plt.ylabel('beer_servings')
plt.title('beer_servings by Continent')
plt.show()

from scipy import stats
#아프리카(AF) 대륙의 정보만 africa 변수에 저장하기
africa = drinks.loc[drinks['continent']=='AF']
africa
africa.shape
africa.mean()
#유럽(EU) 대륙의 정보만 europe 변수에 저장하기
europe = drinks.loc[drinks['continent']=='EU']
europe
europe.shape
europe.mean()
'''
 귀무가설 : 가설예상. 아프리카의 맥주소비량, 유럽의맥주소비량 동일한것으로 가설 설정
 t-statistic : 두개의 모집단의 차이. 음수인경우:뒤의 값이 크다.
 p-value : 설정한 가설이 맞을 확율. 귀무가설이 책택될확율
           0.1 ~ 0.5 된다면 인정. 귀무가설 채택.
           0         된다면 인정불가. 대립가설 채택.
           
 유럽의 맥주소비량은 아프리카의 맥주소비량보다 크다라는 값이 유의미한 값임을 증명.           
'''
# 분산의 값은 동일하다고 가정
tTestResult = stats.ttest_ind(africa['beer_servings'],europe['beer_servings'])
print("t-statistic:%.3f , p-value:%.3f" % tTestResult)

# 분산의 값은 동일하다지 않다고 가정
tTestResult = stats.ttest_ind\
    (africa['beer_servings'],europe['beer_servings'],equal_var=False)
print("t-statistic:%.3f , p-value:%.3f" % tTestResult)

# 대한민국은 얼마나 술을 독하게 마시는 나라일까?
# total_servings 피처 생성
drinks["total_servings"] = \
    drinks["beer_servings"] +drinks["spirit_servings"] + drinks["wine_servings"]
drinks.info()
# alcohol_rate = 알콜소비량 /총주류소비량
drinks["alcohol_rate"] = drinks["total_litres_of_pure_alcohol"]/drinks["total_servings"]
drinks.info()
# drinks 데이터중 alcohol_rate 컬럼이 결측값인 레코드 조회하기
drinks[drinks['alcohol_rate'].isnull()]
#alcohol_rate 컬럼은 total_servings 컬럼의 값이 0인 경우 결측값이 생성됨.
#fillna 함수 : alcohol_rate 컬럼의 결측값을 0으로 채우기
drinks['alcohol_rate'] = drinks['alcohol_rate'].fillna(0)
drinks.info()
# drinks 데이터의 country, alcohol_rate 컬럼만을 가지는 데이터 country_alcohol_rank 변수에 저장
country_alcohol_rank = drinks[['country', 'alcohol_rate']]
country_alcohol_rank
# country_alcohol_rank 알콜비율의 내림차순으로 정렬하기
country_alcohol_rank = country_alcohol_rank.sort_values(by=["alcohol_rate"], ascending=False )
country_alcohol_rank.head(15)

#알콜비율 기준으로 막대그래프 작성하기
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family ='Malgun Gothic')
country_list = country_alcohol_rank.country.tolist() #국가명
print(country_list[:20])
x_pos = np.arange(len(country_list)) #x축값초기화
rank = country_alcohol_rank.alcohol_rate.tolist() #데이터 
country_list.index("South Korea") #14. 15번째로 알콜비율이 높다.
bar_list = plt.bar(x_pos, rank) #막대 그래프 출력
bar_list[country_list.index("South Korea")].set_color('r')
plt.ylabel('alcohol rate')
plt.title('liquor drink rank by contry')
plt.axis([0, 200, 0, 0.3]) # (x축시작,x축종료,y축시작,y축종료)
korea_rank = country_list.index("South Korea") #한국의 인덱스값
korea_alc_rate = country_alcohol_rank\
[country_alcohol_rank['country'] == 'South Korea']\
    ['alcohol_rate'].values[0]
korea_alc_rate #한국의 알콜비율값
'''
annotate : 그래프에 설명선 추가

'''
plt.annotate('South Korea : ' + str(korea_rank + 1)+"번째",  #설명문장           
            xy=(korea_rank, korea_alc_rate),                 # x,y축 
            xytext=(korea_rank + 10, korea_alc_rate + 0.05), # 설명문장시작점
            arrowprops=dict(facecolor='red', shrink=0.05))   #화살표선 
plt.show()

'''
 total_servings 전체 술소비량을 막대그래프로 작성하고,
 대한민국의 위치를 빨강색으로 표시하기
'''

