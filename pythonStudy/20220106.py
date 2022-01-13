# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:23:41 2022

@author: KITCOOP
20220106.py
"""
import pandas as pd
df1 = pd.read_excel("data/stockprice.xlsx")
df2 = pd.read_excel("data/stockvaluation.xlsx")
df1.info()
df2.info()
# id 컬럼을 기준으로 같은 id 값을 가진 레코드 병합.
result1 = pd.merge(df1,df2) #on="id" 기본.
result1
result1 = pd.merge(df1,df2,on="id")
result1
#df1.stock_name 컬럼으로, df2.name 컬럼으로 연결하여 병합.
result2 = pd.merge(df1,df2, left_on='stock_name', right_on='name')
result2
#df1.stock_name 컬럼으로, df2.name 컬럼으로 연결하여 병합. 연결 컬럼의 값이 다른
#               레코드도 조회
# how='outer' : sql 용어. full outer join 
result3 = pd.merge(df1,df2, left_on='stock_name', right_on='name',how='outer')
result3
#df1.stock_name 컬럼으로, df2.name 컬럼으로 연결하여 병합. df1 데이터만 연결 컬럼의 값이 다른
#               레코드도 조회
# how='left' : sql 용어. left outer join 
result4 = pd.merge(df1,df2, left_on='stock_name', right_on='name',how='left')
result4
#df1.stock_name 컬럼으로, df2.name 컬럼으로 연결하여 병합. df2 데이터만 연결 컬럼의 값이 다른
#               레코드도 조회
# how='right' : sql 용어. right outer join 
result5 = pd.merge(df1,df2, left_on='stock_name', right_on='name',how='right')
result5

### 
from bs4 import BeautifulSoup #태그 분석 기능 모듈
import urllib.request as req  # 인터넷 접속 기능 모듈
url="https://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp"
res=req.urlopen(url)  #인터넷 접속. url이 전달한 응답 객체 
print(res)
soup = BeautifulSoup(res,"html.parser") #기본분석기. DOM의 root 노드 리턴
title=soup.find("title").string
wf=soup.find("wf").string
print(title)
'''
   <![CDATA[ 내용 ]]>  : CDATA 섹션. 순수 문자열.  
'''
print(wf)
# <br /> 를 기준으로 한줄씩 출력하기 
for w in wf.split("<br />") :
    print(w)

# 인터넷에서 수신된 내용을 forecast.xml 파일 저장하기
import os.path
if not os.path.exists("forecast.xml") : #forecast.xml 파일이 없으면
   #req.urlretrieve : url로 연결하여, 결과값을 data/forecast.xml 파일로 저장하기 
    req.urlretrieve(url,"data/forecast.xml")

#forecast.xml 파일을 읽어서 BeautifulSoup객체로 분석하기
fp = open("forecast.xml",encoding="utf-8")
soup = BeautifulSoup(fp,"html.parser")
#select_one : 태그 한개 선택
# rss pubDate : rss 태그의 하위 태그 중 pubDate 태그 선택
pubdate = soup.select_one("rss pubDate").string
print(pubdate)

#select : rss 태그들.
rss = soup.select("rss")[0]
pubdate = rss.select_one("pubDate").string
print(pubdate)

#모든 location 태그의 하위태그 중 city,wf 태그를 조회하기
for location in soup.select("location") :
    name = location.select_one("city").string
    weather = location.select_one("wf").string
    print(name,weather)

for location in soup.find_all("location") :
    name = location.find("city").string
    weather = location.find("wf").string
    print(name,weather)

### 네이버에 공시된 환율정보 출력하기
from bs4 import BeautifulSoup
import urllib.request as req
url = "https://finance.naver.com/marketindex/"
res = req.urlopen(url)
#
soup = BeautifulSoup (res,"html.parser",from_encoding="euc-kr")
sel = lambda q : soup.select(q)
#hlist : div.head_info 태그들.
#        환율 정보 표시하는 태그들 
hlist = sel("div.head_info")
print(hlist)
#htitle : 통화국가명
htitle = sel("h3.h_lst")
print(htitle)

taglist=[]
titlelist=[]
for tag, title in zip(hlist, htitle) :
    print(title.select_one("span.blind").string, end="\t") 
    value = tag.select_one("span.value").string # 환율데이터
    print(value, end=" ")
    change = tag.select_one("span.change").string  #상승,하락 폭
    print(change, end="\t")
    blinds = tag.select("span.blind") #상승,하락
    b = tag.select("span.blind")[0].string
    b = tag.select("span.blind")[-1].string
    print(b, end="*******\n")    
    print(blinds[-1].string, end="*******\n")    
    if b == '하락' :
       taglist.append(float(change) * -1)
    else :
       taglist.append(float(change))
    titlelist.append(title.select_one("span.blind").string)

titlelist = titlelist[:-2]
taglist = taglist[:-2]

import matplotlib.pyplot as plt
from matplotlib import  rc

plt.rcParams['axes.unicode_minus']=False
rc('font', family='Malgun Gothic')
xlab = range(len(titlelist))
plt.bar(xlab,taglist)
plt.plot(xlab,taglist)
plt.xticks(xlab,titlelist,rotation='vertical')

# 셀레니움 예제 
#pip install selenium
from selenium import webdriver
import time

'''
   chromedriver.exe 파일 다운 받기
   1. http://chromedriver.chromium.org/downloads
   2. 크롬의 버전 확인하기 ( 97.0.4692.71)
      => 크롬 브라우저 도움말 > 크롬정보
   3. 1번 사이트에서 크롬브라우저의 가장 가까운 버전 선택
   4. 운영체에에 맞는 드라이버 다운받기 
   5. 압축풀기
   6. chromedriver.exe 파일의 위치 저장
'''
driver = webdriver.Chrome("C:/setup/chromedriver")
driver.get("http://python.org")
'''
#top ul.menu li
  #top : 태그 중 id="top" 인 태그. 여기에서는 div 태그.
  ul.menu : #top 태그의 하위 태그 이면서 class="menu"인 ul 태그.
  li   : ul.menu 태그의 하위 태그인 li 태그
'''
menus = driver.find_elements_by_css_selector('#top ul.menu li')
print(type(menus))
menus
pypi = None #pypi 객체값이 없음.
for m in menus: 
    #m.text : li 태그의 문자열값
    if m.text == "PyPI":
        pypi = m #pypi 변수: 내용이 PyPi인 태그 <li ...><a>PyPi</a></li>
    print(m.tag_name,m.text)
pypi.click() #클릭하기
time.sleep(5) #5초대기.
driver.quit() #브라우저 종료.

### 네이버에 로그인 하기
driver = webdriver.Chrome("C:/setup/chromedriver")
driver.get("https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com")
id = input("네이버 아이디를 입력하세요:")
#execute_script : 자바스크립트 명령 전달.
# document.getElementsByName('id')[0].value :
#        문서 중 name='id'인 태그들 중 첫번째 태그의 value
driver.execute_script\
("document.getElementsByName('id')[0].value='"+id+"'")
pw = input("네이버 비밀번호를 입력하세요 : ")
time.sleep(1) #1초대기
driver.execute_script\
("document.getElementsByName('pw')[0].value='"+pw+"'")
time.sleep(1)
#find_element_by_xpath : xml에서 정의한 노드를 찾아가는 방식.
'''
   //*[@id="log.login"]
   // : 루트노드
   *  : 모든태그
   [] : 옵션
   @  : 속성
   @id="log.login" : id='log.login' 속성값을 가진 태그
'''
driver.find_element_by_xpath('//*[@id="log.login"]').click()

# daum 페이지에서 이미지 다운받아 저장하기
from selenium import webdriver
import time
import urllib.request as req 
import os 
driver = webdriver.Chrome("c:/setup/chromedriver")
driver.get("https://search.daum.net/search?w=img&nil_search=btn&DA=NTB&enc=utf8&q=%ED%98%B8%EB%9E%91%EC%9D%B4" )
time.sleep(3)

# #imgList : id='imgList'
# >  : 한단계 하위 태그
# images : img 태그 들
images = driver.find_elements_by_css_selector("#imgList > div > a > img")
img_url = []

for image in images :
    #image : img 태그 하나.
    # image.get_attribute('src') :img 태그의  src 속성의 값을 리턴. 
    url = image.get_attribute('src')
    img_url.append(url)
# img_url : daum의 이미지 접속 url 정보들    
print(img_url)
driver.quit()  #브라우저 닫기 

img_folder = './img' #이미지를 저장할 폴더 선택.
if not os.path.isdir(img_folder) : #img_folder 파일이 폴더가 아니니?
    os.mkdir(img_folder) #폴더 생성.
    
# enumerate(리스트) => 인덱스,리스트요소 리턴
for index, link in enumerate(img_url) :
    #index : img_url의 요소한개의 인덱스값
    #link : img_url의 요소한개. 이미지가 저장된 url정보
    #f'./img/{변수}.jpg' : format화된 문자열
    #req.urlretrieve : url에서 전달해준 내용을 파일로 저장
	req.urlretrieve(link, f'./img/{index}.jpg')

# 다음의 호랑이 화면을 이미지 파일로 저장하기
from selenium import webdriver 
driver = webdriver.Chrome("c:/setup/chromedriver")
driver.get("https://search.daum.net/search?w=img&nil_search=btn&DA=NTB&enc=utf8&q=%ED%98%B8%EB%9E%91%EC%9D%B4" )
driver.save_screenshot("img/tigerpage.png")
driver.quit()

############
#멕시코풍 프랜차이즈 chipotle의 주문 데이터
# chipotle.tsv : csv : 셀의 구분 ,
#                tsv : 셀의 구분 \t(탭). sep = '\t' 설정
############
'''
데이터 속성 설명
order_id : 주문번호
quantity : 아이템의 주문수량
item_name : 주문한 아이템의 이름
choice_description : 주문한 아이템의 상세 선택 옵션
item_price : 주문 아이템의 가격 정보
'''
import pandas as pd
chipo = pd.read_csv("data/chipotle.tsv", sep = '\t')
chipo.info()
# chipo 행과열의 건수를 출력하기
chipo.shape
# 10건 조회하기
chipo.head(10)
# column명을 조회하기
chipo.columns
# index을 조회하기
chipo.index
# 수치데이터 조회하기
chipo.describe()
# order_id 주문번호이므로 숫자형 분석의 의미가 없다.
# order_id 피처의 자료형을 문자열형으로 변경하기
chipo["order_id"]=chipo["order_id"].astype(str)
chipo.info()

# 1.판매상품명과 갯수를 출력하기
print("판매상품명")
print(chipo["item_name"].unique())
print("판매상품갯수 :",len(chipo["item_name"].unique()))
# 2. item 별 가장 많이 주문한 상품의 이름 10개를 출력하기
print(chipo['item_name'].value_counts()[:10])
print(chipo['item_name'].value_counts().index[:10])

chipo[["order_id",'item_name']].head(10)
# 3. item 별 주문 갯수 조회하기 
order_count = chipo.groupby("item_name")["order_id"].count()
order_count
type(order_count)
# 4. item 별 주문 갯수 시각화하기.
#    가장 많은 주문 10개만 막대그래프로 출력하기
#  - order_count : 주문건수의 내림차순정렬
#                  sort_values 함수 
order_count = order_count.sort_values(ascending=False)
order_count = order_count[:10]
order_count
import matplotlib.pyplot as plt
#order_count.index : 상품명들
#tolist() : 리스트화
item_name_list = order_count.index.tolist()
item_name_list
#order_count.values : 주문건수
order_cnt = order_count.values.tolist()
plt.style.use("ggplot")
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1) 
#ax.bar : 막대그래프
ax.bar(item_name_list, order_cnt, align='center')
plt.ylabel('ordered_item_count')
plt.xlabel('item Name')
ax.set_xticklabels(item_name_list, rotation=45)
plt.title('Distribution of all orderd item')
plt.show()

# 5. item 별 판매 갯수 시각화하기.
#    가장 많이 판매한 상품 10개만 막대그래프로 출력하기
chipo[["item_name","quantity"]].head(10)
