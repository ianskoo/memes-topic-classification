from cgitb import text
from curses.textpad import Textbox
from distutils.log import error
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
from spellchecker import SpellChecker


def _preprocess_white_text(im):
    im= cv2.bilateralFilter(im,5, 55,60)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(im, 245, 255, cv2.THRESH_BINARY_INV)
    return im

def _preprocess_black_text(im):
    im= cv2.bilateralFilter(im,5, 55,60)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)
    return im

def _check_return_best_transcr(text_white:list, text_black:list, remove_misspelled:bool=False):
    spell = SpellChecker()
    misspelled_white = spell.unknown(text_white)
    misspelled_black = spell.unknown(text_black)
    
    if text_white:
        error_white = len(misspelled_white)/len(text_white)
    else:
        error_white = 1
    
    if text_black:
        error_black = len(misspelled_black)/len(text_black)
    else:
        error_black = 1
    
    if error_black == error_white == 1:
        return []
    elif error_white <= error_black:
        # Remove misspelled words
        if remove_misspelled:
            text_white = [good for good in text_white if good not in misspelled_white]
        return text_white
    else:
        # Remove misspelled
        if remove_misspelled:
            text_black = [good for good in text_black if good not in misspelled_black]
        return text_black
    

def print_image(img:np.array, cmap:str='gray'):
    # img=np.array(Image.open(img_path + img))
    plt.figure(figsize=(4,4))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=cmap)
    plt.show()

def extract_text_wprocessing(img_path:str, print_processed_img:bool=False, print_text:bool=False, remove_misspelled=False, debug=False):
    custom_config = r"--oem 1 --psm 12 -c tessedit_char_whitelist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!?. '"
    img=np.array(Image.open(img_path))
        
    img_white=_preprocess_white_text(img)
    img_black=_preprocess_black_text(img)
        
    if print_processed_img:
        print_image(img_white)
        print_image(img_black)
    
    text_white = pytesseract.image_to_string(img_white, lang='eng', config=custom_config).split()
    text_black = pytesseract.image_to_string(img_black, lang='eng', config=custom_config).split()
    
    if debug:
        print(f"transcription candidates:\n{text_white}\n{text_black}\n")
    
    final_text = _check_return_best_transcr(text_white, text_black, remove_misspelled)
    
    if print_text and final_text:
        print(" ".join(final_text))#.replace('\n', ' '))
    return " ".join(final_text)