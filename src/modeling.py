

# class Recommender:
# get all words in data
# vectorize, create pca, create embedding -> classes = songs and classes = artist
# fetch_artists(artist, n_artists) -> returns n_artists neighbors for artist
# fetch_songs(song, n_songs) -> returns n_songs neighbors for song where song.artist != neighbor.artist

# r = Recommender()
# r.fit(data) -> create two dfs, one with target = artist other with target = song. vectorize both and fit both into pca embedding. use this for neighbors
# r.fetch_artists('Drake', 5) -> uses pca where target = artist and find 5 closest neighbors of node when node = "Drake"
# r.fetch_songs('Gods Plan', 5) -> uses pca where target = song and find 5 closest neighbors of node when node = "Gods Plan"


# in main() of main.py:
# 	fit_data() -> fitting Recommender

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import pickle
<<<<<<< HEAD
import json

import nltk
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()


'''
TODO
add absolute sentiment/2 difference (make sentiment 0 to 1 instead of -1 to 1) to euc distance
sum(|sent_i - sent_n|/2, euc(pca_i, pca_n))
'''
class SongRecommender:

	def __init__(self, fit = False):
		if fit:
			self.fit()



	def _text_to_score(self, text):
		return sid.polarity_scores(text)['compound']



	def _clean_col(self, x):
		if type(x) == float:
			return 0
		else:
			return x.strip(f'\r')



	def _vectorizer(self, data):
		tfidf = TfidfVectorizer()
		X = tfidf.fit_transform(data).toarray()
		return X, np.array(tfidf.get_feature_names())



	def _get_data(self, df):
		data = list(df.words.values)
		labels = df.target
		le = LabelEncoder()
		y = le.fit_transform(labels)
		seen = []
		self._label_map = {}
		for label, _y in list(zip(labels, y)):
			if label not in seen:
				self._label_map[str(_y)] = label
				seen.append(label)
		return data, y, self._label_map



	def _fit(self, df):
		data, y, label_map = self._get_data(df)
		vect, vocab = self._vectorizer(data)
		ss = StandardScaler()
		X = ss.fit_transform(vect)
		pca = PCA(n_components = 10)
		X_pca = pca.fit_transform(X)
		return X_pca, y
		# plot_mnist_embedding(X_pca, y, label_map)



	def fit(self):
		df = pd.read_csv('../data/test.txt', delimiter = '|', lineterminator='\n')
		df.columns = ['artist', 'song', 'lyrics', 'genre']
		df['song'] = df[['song', 'artist']].agg(' by '.join, axis=1)
		df['genre'] = df.loc[:, 'genre'].apply(self._clean_col)
		print('getting sentiment')
		start_time = datetime.now()
		df['sentiment'] = df['lyrics'].apply(self._text_to_score)

		self.df = df
		print('got sentiment')
		print(datetime.now() - start_time)

		song_target_df = df.drop(['artist', 'genre', 'sentiment'], axis=1)
		song_target_df.columns = ['target', 'words']

		artist_target_df = df.drop(['song', 'genre', 'sentiment'], axis=1)
		artist_target_df.columns = ['target', 'words']

		self._artist_X, self._artist_y = self._fit(artist_target_df)
		self._song_X, self._song_y = self._fit(song_target_df)

		self._song_neighbors()
		self._artist_neighbors()



	def _sentiment_diff(self, song, other_song):
		song_sent = self.df['sentiment'][self.df['song'] == song].values[0]
		other_song_sent = self.df['sentiment'][self.df['song'] == other_song].values[0]
		return abs(song_sent - other_song_sent)/2



	def _find_nearest_songs(self, song, coords, n_songs):
		label_list_dict = {}
		for i in range(len(self._song_X)):
			euc_dist = np.sqrt((self._song_X[i, 0] - coords[0])**2 + (self._song_X[i, 1] - coords[1])**2)
			sent_dif = self._sentiment_diff(song, other_song = self._label_map[str(self._song_y[i])])
			total_dist = euc_dist + sent_dif
			label_list_dict[self._label_map[str(self._song_y[i])]] = total_dist
		sorted_songs = [(k, v) for k, v in sorted(list(label_list_dict.items()), key=lambda x: x[1])]
		return [i for i, j in sorted_songs if i != song ][:n_songs]



	def _fetch_songs(self, song, n_songs=100, verbose = False):
		labels = self._song_y.tolist()
		num = list(self._label_map.keys())[list(self._label_map.values()).index(song)]
		idx = labels.index(int(num))
		coords = self._song_X[idx, 0], self._song_X[idx, 1]
		nearest_songs = self._find_nearest_songs(song, coords, n_sonif verbose:
			print(f'Recommended songs for "{song.split(" by ")[0].title()}" by {song.split(" by ")[1].title()}\n')
			for song in nearest_songs:
				print(f'"{song.split(" by ")[0].title()}" by {song.split(" by ")[1].title()}')gs)
		
		return nearest_songs



	def _to_json(self):
		with open("song_neighbors.json", "w") as outfile: 
    		json.dump(self._song_neighbors, outfile)



	def _song_neighbors(self):
		self._song_neighbors = {}
		for i in range(len(self._song_X)):
			song_name = self._label_map[str(self._song_y[i])]
			print(song_name)
			start_time = datetime.now()
			neighbors = self._fetch_songs(song_name)
			print(datetime.now() - start_time)
			self._song_neighbors[song_name] = neighbors
		self._to_json()



	def _artist_neighbors(self):
		self._artist_neighbirs = {}
		for i in range(len(self._artist_X)):
			artist_name = self._label_map[str(self._artist_y[i])]
			print(artist_name)
			start_time = datetime.now()
			neighbors = self._fetch_artist(artist_name)


	def fetch_artists(self, artist, n_artists):
		pass
		# take mean coordinates for each target label (artist)
		# compare mean coord of artist_input to other artists




	def fetch_songs(self, song, n_songs=100):
		neighbors = self._song_neighbors[song]
		for i, song in enumerate(neighbors):
			if i <= n_songs:
				print(f'"{song.split(" by ")[0].title()}" by {song.split(" by ")[1].title()}')







if __name__ == '__main__':
	sr = SongRecommender()
	now = datetime.now()
	# sr.fetch_songs("SICKO MODE by travis scott", n_songs = 20)
	save_model = open('../models/songrecommender.pickle', 'wb')
	pickle.dump(sr, save_model)
	save_model.close()
	
sr = SongRecommender()
print("saving model")
save_model = open('../models/songrecommender.pickle', 'wb')
pickle.dump(sr, save_model)
save_model.close()
