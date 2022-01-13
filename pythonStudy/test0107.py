# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:46:19 2022

@author: KITCOOP
test0107.py
"""

'''
0.total_servings 전체 술소비량을 막대그래프로 작성하고,
 대한민국의 위치를 빨강색으로 표시하기
'''

'''
1. drinks.csv 데이터를 읽고,대륙별 술 소비량 대비 알콜 비율 컬럼
    (alcohol_rate_continent) 추가하기
    alcohol_rate_continent =대륙별total_litres_of_pure_alcohol/대륙별전체술소비량  
   으로 계산한다.

[결과]
  country continent alcohol_rate_continent
0 Afghanistan AS    0.020293
1 Bahrain       AS    0.020293
2 Bangladesh  AS    0.020293
3 Bhutan        AS    0.020293
4 Brunei         AS    0.020293
'''

'''
 2. 전체 평균보다 적은 알코올을 섭취하는 대륙 중에서, 
    spirit을 가장 많이 마시는 국가 구하기
    
[결과]
Russian Federation
'''
