# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 14:02:29 2021

@author: KITCOOP
test1223.py
"""
'''
1. module폴더의 mod2.py 파일을 읽어서 mod2.bak 파일로 복사하기.
'''
infp = open("module/mod2.py", "r", encoding='UTF8')
outfp = open("module/mod2.bak","w",encoding="UTF8")
while True :
    inStr = infp.readline()
    if inStr == '' : 
        break
    outfp.writelines(inStr)
infp.close()
outfp.close()
print("프로그램 종료")

'''
2. 현재폴더에 temp폴더를 생성하고, 생성된 폴더에 indata.txt 파일을
   생성하여 생성된파일에 키보드에서 입력된 정보 저장하는 프로그램 
   구현하기
'''
import os
wpath = os.getcwd()+"/temp"
if  not os.path.exists(wpath) :
    os.mkdir("temp")
   
outfp = open("temp/indata.txt",'w',encoding='UTF8')
while True :
    inline = input("저장 내용=>")
    if inline == '' :
        break
    outfp.writelines(inline+"\n")
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
infile = input("원본파일의 이름을 입력하세요 : ")
try :
     inFp=open(infile, "r",  encoding='utf-8')
     outfile = input("복사본파일의 이름을 입력하세요 : ")
     outFp=open(outfile, "w",  encoding='utf-8')
     inList = inFp.readlines() #text모드에서만 사용가능한 함수.
     for inStr in inList :
          outFp.writelines(inStr)
     inFp.close()
     outFp.close()
     print("\n복사완료")
except :
     print("원본파일이 존재하지 않습니다.")   


'''
4. 모오스기호가 다음과 같을때 
    .-   'A'      -... 'B'     -.-. 'C'     -..: 'D'    .    'E'
    ..-. 'F'      --.  'G'     .... 'H'     ..   'I'    .--- 'J'
    -.-  'K'      .-.. 'L'     --   'M'     -.   'N'    ---  'O'
    .--. 'P'      --.- 'Q'     .-.  'R'     ...  'S'    -    'T'
    ..-  'U'      ...- 'V'     .--  'W'     -..- 'X'    -.-- 'Y'
    --.. 'Z'
'''
def alpha(src) :
   result = []
   for char in src.split(" "):
        try :
           result.append(dic1[char])
        except KeyError :
            pass
   return "".join(result)
        
def morse(src) :
   result = []
   for char in src:
        try :
           result.append(dic2[char])
        except KeyError :
            pass
   return " ".join(result)
    
dic1 = {
    '.-':'A','-...':'B','-.-.':'C','-..':'D','.':'E','..-.':'F',
    '--.':'G','....':'H','..':'I','.---':'J','-.-':'K','.-..':'L',
    '--':'M','-.':'N','---':'O','.--.':'P','--.-':'Q','.-.':'R',
    '...':'S','-':'T','..-':'U','...-':'V','.--':'W','-..-':'X',
    '-.--':'Y','--..':'Z'
}
dic2 = {}
for k,v in dic1.items() :
    dic2[v] = k

#영문으로 출력
print(alpha('.... . ... .-.. . . .--. ... . .- .-. .-.. -.--'))
#모오스부호로 출력
print(morse('HELLO'))
