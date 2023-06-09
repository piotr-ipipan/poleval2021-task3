import re

gt_txt = None
file_fullpath_gt = "/home/pborkowski/storage/ebooks/txt_from_pdf3/test.txt"
with open(file_fullpath_gt, 'r') as f:
    gt_txt = f.read()

input_file = open(file_fullpath_gt,"r").read()    

for i, hyp_line in enumerate(open(file_fullpath_gt,"r")):
    print (hyp_line)

# x = re.sub("(?<!\n)\n(?!\n)", " ", gt_txt)
# print(gt_txt)
# print()
# 


thresh_dots = 0.3
#a = "ala ma kota  2spac   3cpacje"


# txt_oryginalny, txt_corrected

txt_origin = "Wbrew prognozom dotyczącym postępującego triumfu sekularyzacji”, McGuire bezkompromisowo przekonuje, że „religia jest jedną z najpotężniejszych, najgłębiej odczuwanych i najbardziej wpływowych sił w społeczeństwie”. To za sprawą religii"
txt_origin_list = list(txt_origin)
alignment = "|..|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
txt_correct = "WXXew prognozom dotyczącym postępującego triumfu sekularyzacji”, McGuire bezkompromisowo przekonuje, że „religia jest jedną z najpotężniejszych, najgłębiej odczuwanych i najbardziej wpływowych sił w społeczeństwie”. To za sprawą religii"
txt_correct_list =list(txt_correct)


txt_origin_split=txt_origin.split(" ")
splited_words_len = [len(x) for x in txt_origin_split]  # długości kawałków po podzieleniu


global_pos = 0
rownlogla_tabela = []
for i in splited_words_len:
    # pociete_na_slowa = a[global_pos:(global_pos+i)]
    # print(pociete_na_slowa)
    kreski_kropki = alignment[global_pos:(global_pos+i)]
    
    procent_kropek = kreski_kropki.count(".") / len(kreski_kropki)
    if(procent_kropek>thresh_dots): #cofam zmiany
        txt_correct_list[global_pos:(global_pos+i)]=txt_origin_list[global_pos:(global_pos+i)]

    # print(procent_kropek)
    global_pos += i + 1

txt_correct = "".join(txt_correct_list)
print(txt_correct)

