import fitz #trzeba instalowac tak: pip install PyMuPDF
import cv2
import numpy as np
import pytesseract
import os

pdfy_dir = "/home/pborkowski/storage/ebooks/pdf2/"
output_txt_dir = "/home/pborkowski/storage/ebooks/txt_from_pdf2/"

def pix2np(pix): #zamian formaow grafiki
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im

for f in os.listdir(pdfy_dir):
    filepath = pdfy_dir + f
    print(filepath)
    filepath_out = output_txt_dir + f +".txt"
    doc = fitz.open(filepath)
    i=0
    output_text = ""
    for page in doc:
            pix = page.get_pixmap(dpi=600) # do postaci mapy pixeli, zeby skonwerowac na format pakietu cv2
            img = pix2np(pix)
            croped = img
            i=i+1
                    
            print("\n ######################PAGE", str(i) ,"######################## \n")

            img_rgb = cv2.cvtColor(croped, cv2.COLOR_BGR2RGB)
            strona = pytesseract.image_to_string(img_rgb, lang='pol+eng')
            output_text += strona
            # print(strona)

    f=open(filepath_out, 'w')
    f.write(output_text)
    f.close()



#print(text_all)        
