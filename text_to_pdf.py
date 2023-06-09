import fitz #trzeba instalowac tak: pip install PyMuPDF
import cv2
#import easyocr
import numpy as np
import pytesseract

dpi = 100
# dir = "/home/pborkowski/storage/pdfy_task/text_to_pdf/pirx/"
# files = ["pirx_2.pdf"]

dir = "/home/pborkowski/storage/ZIL/przetwarzanie_pdf/test_roznedpi_pdfu/"
files = ["bn-395030.pdf"]


def pix2np(pix): #zamian formaow grafiki
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im

for f in files:
    filepath = dir + f
    filepath_out = filepath+"DPI"+str(dpi)+".txt"
    doc = fitz.open(filepath)
    
    i=0

text_all = ""
for page in doc:
        pix = page.get_pixmap(dpi=dpi) # do postaci mapy pixeli, zeby skonwerowac na format pakietu cv2
        img = pix2np(pix)
        croped = img
        i=i+1
                
        print("\n ######################PAGE", str(i) ,"######################## \n")

        img_rgb = cv2.cvtColor(croped, cv2.COLOR_BGR2RGB)
        strona = pytesseract.image_to_string(img_rgb, lang='pol+eng')
        text_all += strona
        print(strona)

f=open(filepath_out, 'w')
f.write(text_all)
f.close()

#print(text_all)        