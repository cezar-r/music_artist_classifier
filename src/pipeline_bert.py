# for i in range len(df)
	# if np.random <= 2:
		# go to test directory, check if folder for that class exists
		# if it exists -> add a text file of the lyrics to that folder
		# if it doesn't -> create new folder and add lyrics to that folder
	# else:
		# go to train directory, check if folder for that class exists
		# if it exists -> add a text file of the lyrics to that folder
		# if it doesn't -> create new folder and add lyrics to that folder



import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
from artist_classifier import DataManager


def load_data():
	dm = DataManager()
	clean_data = dm.load_data(explode = 20)
	return clean_data



def df_to_tf_dirs():
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
			# add to train

		

df_to_tf_dirs()