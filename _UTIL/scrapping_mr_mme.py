# %% IMPORTS
import argparse
import sys
from os import getcwd
from os.path import join
from util_file import write_file
from scrapping_util import get_page, get_selenium_firefox_driver, save_article_in_bdd
import re

parser = argparse.ArgumentParser(description='Scrap', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v', '--verbosity', default='0', type=int, choices=[0, 1, 2, 3], help='verbosity level')
parser.add_argument('-t', '--test', default='False', help='make tests')

# http://monsieur.madame.ont.free.fr/plan-du-site/

# http://monsieur.madame.ont.free.fr/blague/?humour&nom-de-famille=A
BASE_URL = 'http://monsieur.madame.ont.free.fr/blague/?humour&nom-de-famille='
urls_to_scrapt = [BASE_URL+chr(ord('A') + i) for i in range(0, 26)]

def get_page_jokes(url, verbose=0):
    """
    Return all page or section links
    Args:
        url (str, optional): the url base. Defaults to "".
        verbose (int, optional): log level. Defaults to 0.

    Raises:
        AttributeError: if page is None

    Returns:
        set(str): link list
    """
    page = get_page(url)
    if page is None:
        raise AttributeError("page expected")
    
    joke_list = []
    
    line_list = page.findAll('td', {'width': '80%'})
    
    for line in line_list:
        question = line.find('h1')
        if question is not None and len(question)>0:
            question = question.get_text()
            joke = line.get_text()
            if joke is not None and len(joke)>0:
                joke = joke.replace(question, "")
                joke = joke.split("(")[0].strip()
                family_name = question.split(" ont")[0].strip()
                family_name = family_name.split("adame ")[-1]
                family_name = family_name.split("mme ")[-1]
                family_name = family_name.split("Mme ")[-1]
                family_name = family_name.strip()
                if family_name is not None and len(family_name)>0:
                    if family_name not in joke:
                        joke += " "+ family_name.upper()
                    joke_list.append(joke)
        
    if verbose>0:
        print(f"[get_page_jokes] \tINFO : {len(joke_list)} jokes found")
    return joke_list


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%                                              MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main(argv):
    print("---------------------------- BENCH EXTRACTION ---------------------------- START")
    args = parser.parse_args(argv)
    args.test = True # TODO à supprimer lorsqu'il ne s'agit plus de tests
    
    args.verbosity             = args.verbosity      if not args.test       else 2
    path_dir                   = getcwd()
    if "_UTIL" not in path_dir:
        path_dir = join(path_dir, "_UTIL")
        
    path_txt_dest   = join(path_dir, "mr_et_mme_list.txt")
    
    joke_list = []
    for url in urls_to_scrapt:
        jk_temp = get_page_jokes(url, verbose=args.verbosity)
        joke_list.extend(jk_temp)
    
    write_file(path_txt_dest, joke_list, remove_if_exist=True, verbose=args.verbosity)
    print("-------------- END --------------")
  
    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#####################################
if __name__ == "__main__":
    main(sys.argv[1:])