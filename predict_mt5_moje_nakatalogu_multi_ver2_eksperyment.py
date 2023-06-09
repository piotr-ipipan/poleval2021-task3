import logging
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import re
import sys
import shlex #do parsowania argumentów ze stringu
import pandas as pd

import time
from argparse import ArgumentParser

import multiprocessing
import warnings
import tqdm
from chunk_generator_context import chunk

from genalog.text import anchor
from genalog.text import alignment

warnings.simplefilter(action='ignore', category=FutureWarning)


from simpletransformers.t5 import T5Model, T5Args

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


data_dir = "/home/pborkowski/storage/ebooks/output_ebooksample_400dpi_6layoutow/"
in_path = data_dir + "3k_in.tsv"
# in_path_witht5 = data_dir + "3k_in_witht5_allegbase.tsv"
# in_path_witht5_cofanie = data_dir + "3k_in_witht5_allegbase_cofanie.tsv"
# in_path_witht5 = data_dir + "3k_in_witht5_alleglarge.tsv"
# in_path_witht5_cofanie = data_dir + "3k_in_witht5_alleglarge_cofanie.tsv"
# in_path = data_dir + "in.tsv"
in_path_witht5 = data_dir + "L_3k_in_witht5_allegbase.tsv"
in_path_witht5_cofanie = data_dir + "L_3k_in_witht5_allegbase_cofanie.tsv"
# in_path_witht5 = data_dir + "in_witht5_alleglarge.tsv"
# in_path_witht5_cofanie = data_dir + "in_witht5_alleglarge_cofanie.tsv"



# data_processed_path_dir = "/home/pborkowski/storage/pdfy_task/200_do_ewaluacji/corrected_txt_part"

# # data_path_dir = "/home/pborkowski/storage/pdfy_task/1000_anotowanych_txt/"
# # data_path_dir = "/home/pborkowski/storage/pdfy_task/1000_anotowanych_sample/"
# # save_path_dir = "/home/pborkowski/storage/pdfy_task/1000_post/"

# data_path_dir = "/home/pborkowski/storage/pdfy_task/5_do_ewaluacji/txt/"
# # save_path_dir = "/home/pborkowski/storage/pdfy_task/5_do_ewaluacji/corrected_txt_large/"
# save_path_dir = "/home/pborkowski/storage/pdfy_task/5_do_ewaluacji/corrected_txt_base/"
# # save_path_dir = "/home/pborkowski/storage/pdfy_task/5_do_ewaluacji/corrected_txt_small/"


s=" "
############# WYTRENOWANE NA TYGODNIKU POW
s = s + "--model_name /home/pborkowski/storage/polevaltask3_cache/model_bin_allegro_plt5-large_1679310187.9930806/best_model/" #ALLEGRO WYTRENOWANY Tygodnik
# s = s + "--model_name /home/pborkowski/storage/polevaltask3_cache/model_bin_allegro_plt5-small_1680189176.8394053/best_model/" #ALLEGRO SMALL
# s = s + "--model_name /home/pborkowski/storage/polevaltask3_cache/model_bin_allegro_plt5-base_1680294097.983086/best_model/" #ALLEGRO BASE



###### PONIZEJ MUSI BYC SPACJA NA POCZATKU (PRZED --) ZEBY SIE PARSOWALO
s = s + " --model_type t5" #TYP MODELU ALLEGRO
# s = s + " --model_type mt5" #TYP MODELU GOOGLE
s = s + " --chunk 50 --stride 0 --num_beams 8 --batch_size 8"

n_procesow = 1
thresh_dots = 0.3
min_dl_slowa_do_cofania_zmian=3
gap_char="✖"

processed = set()

def format_alignment_modified(align1, align2, score, begin, end,
                     full_sequences=False):
    """Format the alignment prettily into a string.

    IMPORTANT: Gap symbol must be "-" (or ['-'] for lists)!

    """
    align_begin = begin
    align_end = end
    start1 = start2 = ""
    start_m = begin  # Begin of match line (how many spaces to include)
    # For local alignments:
    if not full_sequences and (begin != 0 or end != len(align1)):
        # Calculate the actual start positions in the un-aligned sequences
        # This will only work if the gap symbol is '-' or ['-']!
        start1 = str(len(align1[:begin]) - align1[:begin].count("-") + 1) + " "
        start2 = str(len(align2[:begin]) - align2[:begin].count("-") + 1) + " "
        start_m = max(len(start1), len(start2))
    elif full_sequences:
        start_m = 0
        begin = 0
        end = len(align1)

    if isinstance(align1, list):
        # List elements will be separated by spaces, since they can be
        # of different lengths
        align1 = [a + " " for a in align1]
        align2 = [a + " " for a in align2]

    s1_line = ["{:>{width}}".format(start1, width=start_m)]  # seq1 line
    m_line = [" " * start_m]  # match line
    s2_line = ["{:>{width}}".format(start2, width=start_m)]  # seq2 line

    for n, (a, b) in enumerate(zip(align1[begin:end],
                                   align2[begin:end])):
        # Since list elements can be of different length, we center them,
        # using the maximum length of the two compared elements as width
        m_len = max(len(a), len(b))
        s1_line.append("{:^{width}}".format(a, width=m_len))
        s2_line.append("{:^{width}}".format(b, width=m_len))
        if full_sequences and (n < align_begin or n >= align_end):
            m_line.append("{:^{width}}".format(" ", width=m_len))  # space
            continue
        if a == b:
            m_line.append("{:^{width}}".format("|", width=m_len))  # match
        elif a.strip() == "-" or b.strip() == "-":
            m_line.append("{:^{width}}".format(" ", width=m_len))  # gap
        else:
            m_line.append("{:^{width}}".format(".", width=m_len))  # mismatch

    s2_line.append("\n  Score=%g\n" % score)
    # return "\n".join(["".join(s1_line), "".join(m_line), "".join(s2_line)])
    return "".join(m_line)



def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def multithread(lista_paragrafow):
    
    parser = ArgumentParser(description='')
    parser.add_argument('--model_type', default="mt5", help='Name the model')
    parser.add_argument('--model_name', default="google/mt5-base", help='Name the model')
    parser.add_argument('--batch_size', default=32, type=int, help='mini batch size')
    parser.add_argument('--max_seq_length', default=256, type=int, help='max_seq_length')
    parser.add_argument('--max_length', default=256, type=int, help='')
    parser.add_argument('--num_beams', default=1, type=int, help='')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, help='')

    parser.add_argument('--chunk', default=20, type=int, help='chunk size')
    parser.add_argument('--stride', default=5, type=int, help='stride size')
    
    parser.add_argument('--top_k', default=50, type=int, help='top_k')
    parser.add_argument('--top_p', default=0.95, type=float, help='top_p')
    parser.add_argument('--no_sample', action='store_true', help='sample')
    
    cache_dir = os.getenv('SCRATCH', '.') + '/cache/'
    
    #args = parser.parse_args()
    args = parser.parse_args(shlex.split(s)) #podmianka na parsowanie ze stringu


    # Configure the model
    model_args = T5Args()


    model_args.fp16 = False  # dont work with True
    model_args.max_length = args.max_length  # 20
    model_args.num_beams = args.num_beams  # 1
    model_args.repetition_penalty = args.repetition_penalty  # 1.0

    model_args.train_batch_size = args.batch_size
    model_args.eval_batch_size = args.batch_size
    model_args.use_multiprocessed_decoding = False

    model_args.max_seq_length = args.max_seq_length

    # gramformer
    model_args.do_sample = not args.no_sample
    model_args.top_k = args.top_k
    model_args.top_p = args.top_p
    
    ######## moje dodatkowe parametry ###########
    model_args.silent = True# True ######SILENT
    use_cuda = True
    ##########
    model = T5Model(args.model_type, args.model_name, args=model_args,use_cuda=use_cuda)

    korekta_t5 = []
    korekta_t5_z_cofaniem = []     
    # for paragraf in lista_paragrafow:   

    
        # print (paragraf)
        # try:
        #     txt_file_path = os.path.join(data_path_dir, paragraf)
        #     filename_without_ext = paragraf.rsplit(".",1)[0] #czyli dzielimy robiąc maksymalnie 1 dzielenie, szukając puntu podziału od prawej strony
        #     output_file_name = os.path.join(save_path_dir, filename_without_ext + '_corrected.' + "txt")
    
        #     f = open(output_file_name)
        #     lines=f.readlines()
        #     lines=lines[:-2]
        #     f.close()
        # except UnicodeDecodeError:
        #     print('Unicode error - delete last line')
        #     sys.exit()
        # except:
        #     lines=[]
        
        # print('Lines:', len(lines))
        
        # f=open(output_file_name, 'w')
        # f.writelines(lines)
        # f.flush()
        
    for i, hyp in enumerate(tqdm.tqdm(lista_paragrafow)): #enumerate(tqdm.tqdm(open(txt_file_path))): 
        # if i<len(lines): continue #ta operacja jest po to, zeby wyswitlal sie progres za sprawa metody z tqdm
        
        # hyp_line = hyp_line[:-1].replace('-\\n', '').replace('\\n', '\n\\n').replace('\\\\', '\\')
        
        try: #zdarza sie ze w oryginalnej lini jest pojedynczy znak, ktory nie zostal rozpoznany i mamy pusty rekord
            hyp = hyp.replace('-\\n', '').replace('\\n \\n', '').replace('\\n', ' ')
        except:
            hyp = ""


        # doc_id, page_no, year, hyp = hyp_line.split('\t')
        # hyp = hyp_line.split('\t')[-1] #[-1] oznacza wyrzucenie kilku pierwszych eltow z listy (ktora jest po podziale tabulatorami /t), bo tylko w ostatnim rekordzie jest tekst

        # hyp = re.sub(r'(\\n| )+', ' ', hyp)
        hyp = hyp.split()

        xs=[]
        for context_left, start, end, context_right in chunk(hyp, args.chunk, args.stride):
            if args.stride > 0:
                x = ' '.join(hyp[context_left:start]).strip() + ' <left_context> ' + ' '.join(
                    hyp[start:end]).strip() + ' <right_context> ' + ' '.join(
                    hyp[end:context_right]).strip()
            else:
                x = ' '.join(hyp[start:end]).strip()

            x = re.sub(' +', ' ', x)
            xs.append(x)

        # print([f'ocr: {x}' for x in xs])
        preds = model.predict([f'ocr: {x}' for x in xs])
        korekta_t5.append(preds)


        ### PORÓWNUJEMY TEKSTY I COFAMY EW zbyt znaczące zmiany

        if len(xs)>0:
            txt_origin, txt_correct = anchor.align_w_anchor(" ".join(xs)," ".join(preds),gap_char=gap_char) ## align_w_anchor(gt_txt, noise_txt)
            txt_origin_list = list(txt_origin)
            txt_correct_list =list(txt_correct)
            alignment= format_alignment_modified(txt_origin, txt_correct,0, 0, len(txt_origin), full_sequences=True) 
            
            txt_origin_split=txt_origin.split(" ")
            splited_words_len = [len(x) for x in txt_origin_split]  # długości kawałków po podzieleniu

            global_pos = 0
            for i in splited_words_len:
                if (i>=min_dl_slowa_do_cofania_zmian): # jesli slowa za zbyt krotkie (np 1 litera), to nie cofamy zmian
                    kreski_kropki = alignment[global_pos:(global_pos+i)]
                    
                    procent_kropek = kreski_kropki.count(".") / len(kreski_kropki)
                    if(procent_kropek>thresh_dots): #cofam zmiany
                        txt_correct_list[global_pos:(global_pos+i)]=txt_origin_list[global_pos:(global_pos+i)]

                global_pos += i + 1

            # print("ORYG:")
            # print(txt_origin)
            # print("CORR:")
            # print(txt_correct)
            # print("ALI:")
            # print(alignment)
            # print("###\n")
            txt_correct = ("".join(txt_correct_list)).replace(gap_char,"") # zamieniamy liste charow na napis, a potem usuwamy "gap_chary"
            # print("po podmianie:")
            # print(txt_correct)
            # print("##################\n")
        else:
            txt_correct=' '.join(preds)

        korekta_t5_z_cofaniem.append(txt_correct)
        # # f.write(' '.join(preds))
        # f.write(txt_correct)
        # f.write('\n')
        # f.flush()
    # print ("przetworzone: " + paragraf)
    return (korekta_t5,korekta_t5_z_cofaniem)



def main():
    st = time.time()

    # exp_txt = None
    # with open(exp_path, 'r') as f:
    #     exp_txt = f.read()

    in_txt = None
    with open(in_path, 'r') as f:
        in_txt = f.read()

    in_df = pd.read_csv(in_path,sep = "\t",header=None) #,dtype = {'v1': str, 'v2': float,'v3': str }
    in_df.columns=['v1','v2','v3','txt']
    # exp_df = pd.read_csv(exp_path,sep = "\t",header=None)
    # exp_df.columns=['txt']


    lista_paragrafow = list(in_df.txt)
    dlugosci_paragrafow = [len(str(x).replace("\\n \\n","")) for x in lista_paragrafow]
    print(f"Liczba znakow: {sum(dlugosci_paragrafow)} ")
    
    splitted_lista_plikow = list(split(lista_paragrafow,n_procesow))
        
    a_pool = multiprocessing.Pool(n_procesow)
    # REMOVE # w = list(tqdm.tqdm(a_pool.imap(multithread,splitted_lista_plikow)))
    w = a_pool.map(multithread,splitted_lista_plikow)
    a_pool.close()
    a_pool.join()

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    all_korekta_t5 = []
    all_korekta_t5_z_cofaniem = []
    for korekta_t5_list, korekta_t5_z_cofaniem_list in w:
        for paragr in korekta_t5_list:
            all_korekta_t5.append("".join(paragr)) #"".join(paragr) -- robi unlisting
        
        for paragr in korekta_t5_z_cofaniem_list:
            all_korekta_t5_z_cofaniem.append("".join(paragr)) 

    df_do_zapisu = pd.DataFrame(all_korekta_t5)
    df_do_zapisu.to_csv(in_path_witht5,header=False,index=False)
    df_do_zapisu = pd.DataFrame(all_korekta_t5_z_cofaniem)
    df_do_zapisu.to_csv(in_path_witht5_cofanie,header=False,index=False)

if __name__ == '__main__':
    main()


