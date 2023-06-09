data_graniczna = 1960


in_data = "/home/pborkowski/PycharmProjects/poleval2021-task3/2021-ocr-correction/dev-0/in.tsv"
in_filtered = "/home/pborkowski/PycharmProjects/poleval2021-task3/2021-ocr-correction/dev-0/in_filtered.tsv"
expected_data = "/home/pborkowski/PycharmProjects/poleval2021-task3/2021-ocr-correction/dev-0/expected.tsv"
expected_filtered = "/home/pborkowski/PycharmProjects/poleval2021-task3/2021-ocr-correction/dev-0/expected_filtered.tsv"

in_filtered_to_write = ""
expected_filtered_to_write = ""
parsed_data = 0

for hyp_line, ref_line in zip(open(in_data), open(expected_data)):
    hyp_table = hyp_line.split("\t")
    
    try:
        parsed_data = int(hyp_table[2])
    except:
        print("Nie parsuje się data: ",hyp_table[2] )
        parsed_data = 0

    if(parsed_data>data_graniczna): #czyli dobra ksiazka
        print(parsed_data)
        in_filtered_to_write += hyp_line
        expected_filtered_to_write += ref_line
    
#zapis 
f1 = open(in_filtered,"w")
f1.write(in_filtered_to_write)
f1.close()

f2 = open(expected_filtered,"w")
f2.write(expected_filtered_to_write)
f2.close()