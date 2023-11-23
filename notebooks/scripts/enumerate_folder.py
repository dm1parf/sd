"""
enumerates jpg filen in folders
"""

import shutil
import os

FILES_FOLDER = '../data/images'

for idx, file in enumerate(os.listdir(FILES_FOLDER)):
    print(file)
    rel_file_path = os.path.join(FILES_FOLDER, file)
    rel_new_file_path = os.path.join(FILES_FOLDER, f'{idx}.jpg')
    shutil.move(rel_file_path, rel_new_file_path)
