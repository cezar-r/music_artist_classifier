#!/usr/bin/env python
# coding: utf-8

'''
This file can be used to load the data into a format that can be used for the BERT model
BERT can be ran here: https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/classify_text_with_bert.ipynb#scrollTo=6IwI_2bcIeX8
'''

import os
import random
from pathlib import Path
from artist_classifier import DataManager


def load_data():
	"""Loads data in
	
	Returns
	-------
	clean_data: pd.DataFrame
	"""
	dm = DataManager()
	clean_data = dm.load_data(explode = 20)
	return clean_data


def df_to_tf_dirs():
	"""Converts data into format that is readable for BERT model 
	Two folders -> train and test
		Train and test have folders for each category
			Each category folder has .txt files of lyrics pertaining to that category
	"""
	data = load_data()

	for i in range(len(data)):
		lyric = data.iloc[i, 0]
		artist = data.iloc[i, 1]

		artist = "_".join(artist.split(' '))

		print(f"{artist} {i+1}/{len(data)}")

		filename = f'{i+1:07}'

		if random.randint(1, 10) <= 2:

			Path(f"../bert_data/test/{artist}").mkdir(parents=True, exist_ok=True)
			
			with open(f"../bert_data/test/{artist}/{filename}.txt", "w") as f:
				f.write(lyric)

		else:
			Path(f"../bert_data/train/{artist}").mkdir(parents=True, exist_ok=True)
			with open(f"../bert_data/train/{artist}/{filename}.txt", "w") as f:
				f.write(lyric)	

if __name__ == '__main__':
	df_to_tf_dirs()
