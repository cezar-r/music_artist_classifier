import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import pickle
import json

sid = SentimentIntensityAnalyzer()



class ArtistRecommender:

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



	def fit(self):
		df = pd.read_csv('../data/test.txt', delimiter = '|', lineterminator='\n')
		df.columns = ['artist', 'song', 'lyrics', 'genre']
		df['song'] = df[['song', 'artist']].agg(' by '.join, axis=1)
		df['genre'] = df.loc[:, 'genre'].apply(self._clean_col)

		artist_target_df = df.drop(['song', 'genre'], axis=1)
		artist_target_df.columns = ['target', 'words']

		artist_target_df['words'] = artist_target_df.groupby(['target'])['words'].transform(lambda x : ' '.join(x))
		artist_target_df = artist_target_df.drop_duplicates()

		self._artist_X, self._artist_y = self._fit(artist_target_df)

		print('getting sentiment')
		now = datetime.now()
		artist_target_df['sentiment'] = artist_target_df['words'].apply(self._text_to_score)
		print('got sentiment\n')
		print(datetime.now() - now)
		self.df = artist_target_df

		self._artist_neighbors()



	def _sentiment_diff(self, artist, other_artist):
		artist_sent = self.df['sentiment'][self.df['target'] == artist].values[0]
		other_artist_sent = self.df['sentiment'][self.df['target'] == other_artist].values[0]
		return abs(artist_sent - other_artist_sent)/2



	def _find_nearest_artists(self, artist, coords, n_artists):
		label_list_dict = {}
		for i in range(len(self._artist_X)):
			euc_dist = np.sqrt((self._artist_X[i, 0] - coords[0])**2 + (self._artist_X[i, 1] - coords[1])**2)
			sent_dif = self._sentiment_diff(artist, other_artist = self._label_map[str(self._artist_y[i])])
			total_dist = euc_dist + sent_dif
			label_list_dict[self._label_map[str(self._artist_y[i])]] = total_dist
		sorted_artists = [(k, v) for k, v in sorted(list(label_list_dict.items()), key=lambda x: x[1])]
		return [i for i, j in sorted_artists if i != artist ][:n_artists]



	def _fetch_artists(self, artist, n_artists=100, verbose = False):
		labels = self._artist_y.tolist()
		num = list(self._label_map.keys())[list(self._label_map.values()).index(artist)]
		idx = labels.index(int(num))
		coords = self._artist_X[idx, 0], self._artist_X[idx, 1]
		nearest_artists = self._find_nearest_artists(artist, coords, n_artists)
		if verbose:
			print(f'Recommended artists for "{artist.title()}" \n')
			for a in nearest_artists:
				print(f'"{a.title()}')

		return nearest_artists



	def _to_json(self):
		with open("../data/artist_neighbors.json", "w") as outfile: 
			json.dump(self._artist_neighbors, outfile)



	def _artist_neighbors(self):
		self._artist_neighbors = {}
		for i in range(len(self._artist_X)):
			artist_name = self._label_map[str(self._artist_y[i])]
			print(artist_name)
			start_time = datetime.now()
			neighbors = self._fetch_artists(artist_name, verbose = True)
			self._artist_neighbors[artist_name] = neighbors
		self._to_json()


	def recommend(self, artist, n_artists=20):
		df = pd.read_json('../data/artist_neighbors.json')
		neighbors = df[artist]
		print(f'Artist recommendations for {artist.title()}\n')
		for i, a in enumerate(neighbors):
			if i <= n_artists:
				print(f'{a.title()}')



a = ArtistRecommender()
a.recommend('trippie redd')