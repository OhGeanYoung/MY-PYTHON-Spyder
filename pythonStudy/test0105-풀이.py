# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:56:59 2022

@author: KITCOOP
test0105-풀이.py
"""
'''
1. supplier_data.csv 파일을 pandas를 이용하여 읽고 
 ["1/20/14","1/30/14"] 일자 데이터만 화면에 출력하기
'''
import pandas as pd

infile='data/supplier_data.csv'
df = pd.read_csv(infile)
print(df);
print(df.info());
importdate = ["1/20/14","1/30/14"]
df_inset = df.loc[df["Purchase Date"].isin(importdate),:]
print(df_inset)

'''
2.  supplier_data.csv 파일 데이터에서 Invoice Number가 920으로
 시작하는 레코드만 화면에 출력하기
''' 
infile='data/supplier_data.csv'
df = pd.read_csv(infile)
print(df["Invoice Number"].str.startswith("920"))
df_inset = df.loc[df["Invoice Number"].str.startswith("920"),:]
print(df_inset)

'''
3. sales_2013.xlsx 파일 중 Purchase Date 컬럼의 값이 
"01/24/2013"과 "01/31/2013" 인 행만 sales_2013_01.xlsx 파일로 저장하기
 isin 함수 사용.
'''
import pandas as pd
infile="data/sales_2013.xlsx"
outfile = "data/sales_2013_01.xlsx"
df = pd.read_excel(infile,"january_2013")
print(df.info())
print(df.head())
select_date = ['01/24/2013','01/31/2013']
#select_date = ['2013-01-24','2013-01-31']
df_value = df[df['Purchase Date'].isin(select_date)]
df_value
print(df_value.info())
writer = pd.ExcelWriter(outfile)
df_value.to_excel(writer,sheet_name="jan_13_output",index=False)
writer.save()

'''
4. sales_2013.xlsx 파일의 모든 sheet의  열이 
"Customer Name", "Sale Amount" 컬럼만 
sales_2013_allamt.xlsx 파일의 sales_2013_allamt sheet 에 저장하기
'''
import pandas as pd
infile="data/sales_2013.xlsx"
outfile = "data/sales_2013_allamt.xlsx"
writer = pd.ExcelWriter(outfile)
df = pd.read_excel(infile,sheet_name=None,index_col=None)
row_output = []
for worksheet_name,data in df.items() :
    print("===",worksheet_name,"===")
    data_value = data.loc[:,["Customer Name","Sale Amount"]]
    row_output.append(data_value)
filtered_row = pd.concat(row_output,axis=0,ignore_index=True)    
filtered_row.to_excel\
    (writer,sheet_name="sales_2013_allamt",index=False)
writer.save() 

'''
5. seaborn 모듈의 titanic 데이터를 이용하여 클래스별 
 생존 인원을 출력하시오
''' 
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.groupby(['class']).survived.sum()
titanic.groupby(['class']).survived.count()

