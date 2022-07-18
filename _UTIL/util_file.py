
from os.path import join, exists
from os import remove, rename
from pathlib import Path

from tqdm import tqdm

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%                                              FONCTIONS UTILES GENERIQUES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_dir(dest_path, verbose=0):
    """
    Create dir

    Args:
        dest_path (str): destination path (directory)
        verbose (int, optional): log level. Defaults to 0.
    """
    # Création du répertoire s'il n'existe pas
    if dest_path is None or len(dest_path.strip()) > 0:   
        base = Path(dest_path)
        base.mkdir(exist_ok=True)
        
    return dest_path


def read_file(file_path, verbose=0):
    """
    Read the file

    Args:
        dest_path (str): destination path (directory)
        file_name (str): json file name
        verbose (int, optional): log level. Defaults to 0.
    Return :
        lines (list(bytes)) : file lines
    """       
    # charge fichier log en bite pour traiter les différents encodage
    with open(file_path, "r") as input_file:
        lines = input_file.readlines()
    return lines

   
def write_file(dest_path, lines, file_name="",remove_if_exist=True, verbose=0):
    """
    Write the file

    Args:
        dest_path (str): destination path (directory)
        file_name (str): json file name
        lines (iterable) : list
        remove_if_exist (bool, optional): if True check if the file ever exist and remove it. Defaults to True.
        verbose (int, optional): log level. Defaults to 0.
    """
    # Création du répertoire s'il n'existe pas
    if file_name is None or len(file_name.strip()) > 0:   
        base = Path(dest_path)
        base.mkdir(exist_ok=True)
        
        # Directly from dictionary
        dest_file_path = join(base,file_name)
    else:
        dest_file_path = dest_path
        
    # Suppression si existe
    if remove_if_exist:
        remove_file_if_exist(file_path=dest_file_path, backup_file=verbose)
    
    end_line_need = '\n' if '\n' not in lines[0] else '' 
        
    with open(dest_file_path, 'w') as outfile:
        outfile.write(end_line_need.join(lines))
    return dest_file_path

def remove_file_if_exist(file_path, backup_file=False, verbose=0):
    """
    Remove file

    Args:
        file_path (str): the file path (inlude file name)
        backup_file (bool, optional): if True save the previous file with .backup. Defaults to False.
        verbose (int, optional): Log level. Defaults to 0.
    """
    if (exists(file_path)):
        if backup_file:
            if (exists(str(file_path)+".backup")):
                remove(str(file_path)+".backup")
            rename(str(file_path), str(file_path)+".backup")
        else:
            remove(file_path)

def write_a_file_by_line(dest_path, lines, file_name_prefix="phrase",file_name_extension=".txt", verbose=0):
    line_by_id = {}
    function_short_name = "write_file_by_line"
    if dest_path is None or len(dest_path) ==0:
        if verbose > 0:
            print(f"\n[{function_short_name}]\tINFO : no input dest_path")
        return None
    
    if lines is None or len(lines) ==0:
        if verbose > 0:
            print(f"\n[{function_short_name}]\tINFO : no input lines to write")
        return None
    
    for i in tqdm(range(0, len(lines)), desc=f"[{function_short_name}] lines", disable=verbose<1):
        line = lines[i]
        # Creation du nom de fichier
        file_name=file_name_prefix+f"{i:0>6d}"
        if line is not None and len(line)>0:
            res = write_file(dest_path, [line], file_name=file_name+file_name_extension,remove_if_exist=True, verbose=verbose)
            if res is not None:
                line_by_id[file_name] = line
    return line_by_id
        
