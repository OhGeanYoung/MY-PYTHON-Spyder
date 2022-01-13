# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 13:28:03 2022

@author: KITCOOP
test0106-풀이.py
"""
'''
1. http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp 의 내용을 
   인터넷을 통해 데이터를 수신하고 다음 결과형태로 출력하시오
[결과]
+ 구름많음
 |-  서울
 |-  인천
 |-  수원
 |-  파주
 |-  이천
 |-  평택
 |-  강릉
 |-  대전
 |-  세종
 |-  홍성
 |-  청주
 |-  충주
 |-  영동
 |-  광주
 |-  목포
 |-  여수
 |-  순천
 |-  광양
 |-  나주
 |-  전주
 |-  군산
 |-  정읍
 |-  남원
 |-  고창
 |-  무주
 |-  부산
 |-  울산
 |-  창원
 |-  진주
 |-  거창
 |-  통영
 |-  대구
 |-  안동
 |-  포항
 |-  경주
 |-  울진
 |-  울릉도
 |-  제주
 |-  서귀포
+ 흐림
 |-  춘천
 |-  원주'''   
from bs4 import BeautifulSoup #html, xml 분석 도구
import urllib.request as req  #인터넷 접속 모듈
url="https://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"
res = req.urlopen(url) # 인터넷(url)에 연결. 요청.
info = {} #dictionary
# soup : res의 정보를 분석해서 저장 객체 
soup = BeautifulSoup (res, "html.parser")
#find_all("location") : location 태그 정보들. 
for location in soup.find_all("location") :
    #location.find("city").string : location 태그의 하위 태그 중 city태그 선택
    name = location.find("city").string  #도시명
    weather = location.find("wf").string #흐림, 맑음
    if not (weather in info) : #새로운 날씨정보.
        info[weather] = [] # 도시명을 저장하기 위한 리스트 객체 생성
    info[weather].append(name)
# info 딕셔너리 정보 출력
for weather in info.keys() :
    print("+",weather)
    for name in info[weather] :
        print(" |- ",name)
 
'''
2. 네이버에 본인들의 주문상품 목록을 조회하기
'''
import time
from selenium import webdriver
driver = webdriver.Chrome("C:/setup/chromedriver")
time.sleep(1)
driver.get("https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com")
id = input("네이버 아이디를 입력하세요 : ")
driver.execute_script ("document.getElementsByName('id')[0].value='"+id+"'")
pw = input("네이버 비밀번호를 입력하세요 : ")
driver.execute_script("document.getElementsByName('pw')[0].value='"+pw+"'")
driver.find_element_by_xpath('//*[@id="log.login"]').click()
time.sleep(1)
#쇼핑메뉴 클릭
driver.find_element_by_xpath\
    ('//*[@id="NM_FAVORITE"]/div[1]/ul[1]/li[5]/a').click()
time.sleep(1)
#쇼핑my
driver.find_element_by_xpath('//*[@id="_myPageWrapper"]/a').click()
time.sleep(1)
#주문확인
driver.find_element_by_xpath\
 ('//*[@id="_myPageWrapper"]/div/div[3]/ul[2]/li[2]/a').click()
time.sleep(1)
#주문내용을 조회.
products = driver.find_elements_by_css_selector(".goods_pay_section")
for product in products:
    print("-", product.text)
time.sleep(2)   
driver.quit()        

'''
3. item 별 판매 갯수 시각화하기.
   가장 많이 판매한 상품 10개만 막대그래프로 출력하기
   20220106-1.png 참조
'''   
import pandas as pd
import matplotlib.pyplot as plt

chipo = pd.read_csv("data/chipotle.tsv", sep = '\t')
chipo_chicken = chipo[chipo['item_name'] == "Chicken Bowl"]
item_qty = chipo.groupby("item_name")["quantity"].sum()
item_qty = item_qty.sort_values(ascending=False)[:10] #판매수량이 많은 10개 상품
item_name_list = item_qty.index.tolist() #그래프 출력할 item이름 목록
sell_cnt = item_qty.values.tolist()      #그래프 출력할 item 목록
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1) 
ax.bar(item_name_list, sell_cnt, align='center')
plt.ylabel('item_sell_count')
plt.xlabel('item Name')
ax.set_xticklabels(item_name_list, rotation=45)
plt.title('Distribution of all sell item')
plt.show()

'''
4. 멕시코풍 프랜차이즈 Chipotle의 주문 데이터를 이용하여 문제 풀기
    Chicken Bowl을 2개 이상 주문한 주문 횟수 구하기
    주문번호    Chicken Bowl 주문수량
       1                 2
       2                 3
       3                 1
         주문횟수 :   2    1,2,번주문만 횟수 
'''
import pandas as pd
chipo = pd.read_csv("data/chipotle.tsv", sep = '\t')
#Chicken Bowl 데이터만 저장
chipo_chicken = chipo[chipo['item_name'] == "Chicken Bowl"]
chipo_chicken
chipo_chicken_result = \
    chipo_chicken[chipo_chicken['quantity'] >= 2]
chipo_chicken_result
chipo_chicken_result.groupby("order_id").count()
len(chipo_chicken_result.groupby("order_id").count()) #행의 수
print(chipo_chicken_result.shape[0]) #행의 수

