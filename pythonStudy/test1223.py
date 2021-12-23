# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 14:02:29 2021

@author: KITCOOP
test1223.py
"""
'''
1. module폴더의 mod2.py 파일을 읽어서 mod2.bak 파일로 복사하기.
'''
import os
os.mkdir("module")
infp = open("mod2.py","r",encoding="UTF-8")
outfp = open("mod2.bak","w",encoding="UTF-8")
while True :
    indata = infp.read()
    if not indata :
        break
    outfp.write(indata)
infp.close()
outfp.close()
'''
2. 현재폴더에 temp폴더를 생성하고, 생성된 폴더에 indata.txt 파일을
   생성하여 생성된파일에 키보드에서 입력된 정보 저장하는 프로그램 
   구현하기
'''
import os
os.mkdir("temp")
outfp = open("indata.txt","w",encoding="UTF-8")
while True:
    outstr = input("내용입력 : ")
    if outstr == "" :
        break
    outfp.writelines(outstr + "\n")
outfp.close()
'''
3.원본 파일의 이름을 입력받고, 입력받은 복사본 파일이름으로 복사하는 
  프로그램 작성하기
  원본파일이 없는 경우 원본 파일이 존재하지 않습니다. 출력하기

[결과]
원본파일의 이름을 입력하세요 : aaa
원본파일이 존재하지 않습니다.

원본파일의 이름을 입력하세요 : data.txt
복사본파일의 이름을 입력하세요 : databak.txt
복사완료
'''
import os
origin = input("원본파일의 이름을 입력하세요 :")
if os.path.exists(origin) :
    infp = open(origin,"r",encoding="UTF-8")
    copy = input("복사본파일의 이름을 입력하세요 : ")
    outfp = open(copy,"w",encoding="UTF-8")
    print("복사완료")
else :
    print("원본파일이 존재하지 않습니다.")
infp.close()
outfp.close()


    
'''
4. 모오스기호가 다음과 같을때 
    .-   'A'      -... 'B'     -.-. 'C'     -..: 'D'    .    'E'
    ..-. 'F'      --.  'G'     .... 'H'     ..   'I'    .--- 'J'
    -.-  'K'      .-.. 'L'     --   'M'     -.   'N'    ---  'O'
    .--. 'P'      --.- 'Q'     .-.  'R'     ...  'S'    -    'T'
    ..-  'U'      ...- 'V'     .--  'W'     -..- 'X'    -.-- 'Y'
    --.. 'Z'
'''

#영문으로 출력
print(alpha('.... . ... .-.. . . .--. ... . .- .-. .-.. -.--'))
#모오스부호로 출력
print(morse('HELLO'))
