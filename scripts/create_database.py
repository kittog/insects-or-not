# create dataframe
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import os

def get_filenames(f):
	path = "corpus/" + f
	# f for folder
	filenames = []
	labels = []
	documentIds = []
	for root, dirs, files in os.walk(path, topdown=False):
	  for name in files:
	  	label = name.split("_")[0]
	  	documentId = re.findall("[0-9]{1,3}", name)[0]
	  	labels.append(label)
	  	documentIds.append(int(documentId))
	  	_abs_path = os.path.abspath(name)
	  	filenames.append(name)
	return filenames, labels, documentIds


#### main


filenames_insects, labels_insects, documentsIds_insects = get_filenames("insects")
filenames_ninsects, labels_ninsects, documentsIds_ninsects = get_filenames("non-insects")

filenames = filenames_insects + filenames_ninsects
labels = labels_insects + labels_ninsects
documentIds = documentsIds_insects + documentsIds_ninsects

file_contents = []
count = 0
for file in tqdm(filenames):
	count += 1
	directory = file.split("_")[0]
	path = f'''corpus/{directory}/{file}'''
	with open(path, "r") as f:
		file_contents.append(f.read())

doc_df = pd.DataFrame(list(zip(filenames, labels, documentIds)),
	columns=['filename', 'label', 'documentId'])
doc_df['text'] = file_contents

doc_df.to_csv("insects_or_not.csv")