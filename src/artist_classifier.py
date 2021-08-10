#!/usr/bin/env python
# coding: utf-8

"""
Contains 3 objects, all of which are sequantially used to cast a prediction in the web app.
    DataManager(): Loads in data.
    Classifier(): Trains on data and pickles itself for later use
    LyricPredictor(): Uses pickled Classifier() model to predict artist
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import re 
import nltk 
import heapq
import pickle
import warnings
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

plt.style.use("dark_background")
warnings.filterwarnings("ignore")

REGEX = re.compile('[^a-zA-Z]')


class DataManager:
	"""Loads in data, with various, optional cleaning tools"""

	def _clean_col(self, x):
		"""Cleans each row in column that has /r at the end of each string due to how the CSV is read in
		
		Parameters
		----------
		x: string
		
		Returns
		-------
		x: string
		"""
		if type(x) == float: # in case there are any nan values
				return 0
		else:
			return x.strip(f'\r')


	def _clean_text(self, x):
		"""Cleans text of any unwanted characters
		
		Parameters
		----------
		x: string
		
		Returns
		-------
		x: string
		"""
		return REGEX.sub(' ', x)


	def _snippets(self, x):
		"""Only uses the first 50 words in a song lyric (not recommeneded)
		
		Parameters
		----------
		x: string
		
		Returns
		-------
		x: string
		"""
		return ' '.join(x.split(' ')[:50])


	def _explode(self, df, snip_len):
		"""Explodes each lyric into snippets of size snip_len into multiple rows
		Ex: (snip_len = 2) 
		"i walked the dog today" -> "i walked" "the dog" "today"
		
		Parameters
		----------
		df: pd.DataFrame
		snip_len: int
		
		Returns
		-------
		df: pd.DataFrame
		"""
		df = pd.concat([pd.Series(row['artist'], 
			[' '.join(row['lyrics'].split(' ')[i: i+snip_len]) for i in range(0, len(row['lyrics'].split(' ')), snip_len)]) for _, row in df.iterrows() 
		]).reset_index()
		df.columns = ['lyrics', 'artist']
		return df


	def load_data(self, filename="test", snippets = False, explode = False):
		"""Loads data into pandas DataFrame from desired CSV/TXT file
		
		Parameters
		----------
		filename: string
		snippets: bool
		explode: int or bool
		
		Returns
		-------
		df: pd.DataFrame
		"""
		df = pd.read_csv(f'../data/{filename}.txt', delimiter = '|', lineterminator='\n')
		df.columns = ['artist', 'song', 'lyrics', 'genre']
		df['genre'] = df.loc[:, 'genre'].apply(self._clean_col)
		df['lyrics'] = df.loc[:, 'lyrics'].apply(self._clean_text)
		self.df = df
		if snippets:
			df['lyrics'] = df.loc[:, 'lyrics'].apply(self._snippets)
		elif explode:
			df = self._explode(df, explode)

		return df



class Classifier:
	"""
	Used to train and test model. Has methods for PCA plots as well. Also used to save and load pickle files.
	"""

	def __init__(self, data = None, model = SGDClassifier):
		"""Initalizes data as well as model that is being used
		
		Parameters
		----------
		data: pd.DataFrame
		model: 
			default: SGDClassifier. Can use most SKLearn classifier models
		"""
		self.data = data
		self.model = model(loss = 'log')
		if self.data is not None:
			self._split_data()


	def _split_data(self):
		"""Splits data into train and test data"""
		X = self.data.lyrics
		y = self.data.artist
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = .2)

		
	def _arr_to_str(self, x):
		"""Converts an array to string
		
		Parameters
		----------
		x: list or string
		
		Returns
		-------
		x: string
		"""
		if type(x) == list:
			return " ".join(x)
		else:
			return x

		
	def fit(self,  vector_params, over_sample = False, under_sample = False):
		"""Trains model. Has parameters that dictate what parameters are used for the CountVectorizer
		as well as parameters for over sampling and under sampling
		
		Parameters
		----------
		vector_params: dict
		over_sample: bool
		under_sample: bool
		"""
		if over_sample:
			over_sampler = RandomOverSampler()
			self.X_train, self.y_train = over_sampler.fit_resample(self.X_train.to_numpy().reshape(-1, 1), self.y_train)

		elif under_sample:
			under_sampler = RandomUnderSampler()
			self.X_train, self.y_train = under_sampler.fit_resample(self.X_train.to_numpy().reshape(-1, 1), self.y_train)

		self.X_train = pd.Series(self.X_train.tolist())

		self.X_train = self.X_train.apply(self._arr_to_str)

		self.pipe = Pipeline([('vect', CountVectorizer(**vector_params)),
						('tfidf', TfidfTransformer()),
						('clf', self.model),
						])

		self.pipe.fit(self.X_train, self.y_train)


	def predict(self):
		"""Uses trained model to predict unseen data"""
		self.y_hat = self.pipe.predict(self.X_test)
		

	def accuracy(self, verbose = False):
		"""Prints/returns the models accuracy
		
		Parameters
		----------
		verbose: bool
		
		Returns
		-------
		self.accuracy: int
		"""
		self.accuracy = accuracy_score(self.y_test, self.y_hat)
		if verbose:
			print(self.accuracy)
		return self.accuracy


	def report(self, verbose = True):
		"""Prints/returns the models classification report
		
		Parameters
		----------
		verbose: bool
		
		Returns
		-------
		classification_report: string
		"""
		if verbose:
			print(classification_report(self.y_test, self.y_hat))
		return classification_report(self.y_test, self.y_hat)


	def _get_data(self, df):
		"""Labels data in DataFrame using LabelEncoder object

		Parameters
		----------
		df: pd.DataFrame
			
		Returns
		-------
		data: list
		y: np.array
		genre_label_map: dict
		"""
		data = list(df.lyrics.values)
		labels = df.artist
		le = LabelEncoder()
		y = le.fit_transform(labels)
		seen = []
		genre_label_map = {}
		for label, _y in list(zip(labels, y)):
			if label not in seen:
				genre_label_map[str(_y)] = label
				seen.append(label)
		return data, y, genre_label_map


	def _vectorizer(self, data):
		"""Creates a tfidf and returns it as an np.array, as well as the feature names (vocabulary)
	    
		Parameters
		----------
		data: list
		
		Returns
		-------
		X: np.array
		np.array
		"""
		tfidf = TfidfVectorizer()
		X = tfidf.fit_transform(data).toarray()
		return X, np.array(tfidf.get_feature_names())


	def _plot_embedding(self, X, y, label_map):
	    """Plot two dimensional PCA embedding
	    
	    Parameters
	    ----------
	    X: numpy.array
	    y: numpy.array
	    label_map: dict
	    """
	    x_min, x_max = np.min(X, 0), np.max(X, 0)
	    X = (X - x_min) / (x_max - x_min)

	    plt.rcParams["figure.figsize"] = (18, 7)

	    color_map = {'0' : 'purple',
	    			'1' : 'red',
	    			'2' : 'blue',
	    			'3' : 'orange',
	    			'4' : 'teal',
	    			'5' : 'yellow',
	    			'6' : 'green',
	    			'7' : 'lime',
	    			'8' : 'hotpink',
	    			'9' : 'darkgoldenrod'}

	    for i in range(X.shape[0]):
	    	if label_map[str(y[i])] == 'other':
		    	plt.text(X[i, 0], X[i, 1], str(y[i]), color='grey', alpha = .2, fontdict={'weight': 'bold', 'size': 12})
	    	else:
		    	plt.text(X[i, 0], X[i, 1], str(y[i]), color=color_map[str(y[i])], alpha = .6, fontdict={'weight': 'bold', 'size': 12})

	    y_range = [np.mean(X[:, 1]) - .001, np.mean(X[:, 1]) + .001]
	    x_range = [np.mean(X[:, 0]) - .0002, np.mean(X[:, 0])]

	    plt.ylim(y_range)
	    plt.xlim(x_range)

	    patches = []
	    for i in range(len(np.unique(y))):
	    	if label_map[str(i)] == 'other':
	    		patch = mpatches.Patch(color =  'grey', label = label_map[str(i)] + '-' + str(i))
	    	else:
		    	patch = mpatches.Patch(color=color_map[str(i)], label = label_map[str(i)] + '-' + str(i))
	    	patches.append(patch)

	    plt.title('PCA Embedding')
	    plt.xlabel('PCA Dimension 1')
	    plt.ylabel('PCA Dimension 2')
	    plt.legend(handles = patches)
	    plt.savefig('../images/pca_embedding_spaced_out_med4.png')

		
	def _clean_report(self, report):
		"""Converts the classification report (type str) into a dictionary
		
		Parameters
		----------
		report: string
		
		Returns
		-------
		sorted_scores_dict: dict
		"""
		report = report.replace('\n', '')
		new_str = [report[0+i:66+i] for i in range(0, len(report), 66)]
		scores_dict = {}
		for string in new_str[1:-3]:
			line = string.split('   ')
			for char in line:
				if char != '':
					artist = char.lstrip()
					break
			for char in line[::-1][1:]:
				if char != '':
					accuracy = char.lstrip()
					break
			scores_dict[artist] = float(accuracy)
		sorted_scores_dict = {k : v for k, v in sorted(list(scores_dict.items()), key = lambda x: x[1])[::-1]}
		return sorted_scores_dict


	def _relabel_rows(self, x):
		"""Relabels artists who are not in the top 10 accuracy scores with "other"
		
		Parameters
		----------
		x: string
		
		Returns
		-------
		x: string
		"""
		if x in self.top_ten_artists:
			return x
		else:
			return 'other'


	def _relabel_df(self):
		"""Relabels the data such that artists who are not one of the top 10 predicted artists
		(in terms of accuracy) are labelled as "other"
		
		Returns
		-------
		new_df: pd.DataFrame
		"""
		report = self.report()
		report = self._clean_report(report)
		self.top_ten_artists = list(report.keys())[:9]
		new_df = self.data
		new_df['artist'] = new_df['artist'].apply(self._relabel_rows)
		return new_df


	def plot_artist_labels(self):
		"""Parent function to plotting genre cluster graph"""
		new_df = self._relabel_df()
		data, y, label_map = self._get_data(new_df)
		print('vectorizing')
		vect, vocab = self._vectorizer(data)
		ss = StandardScaler()
		X = ss.fit_transform(vect)
		print('fitting into pca')
		pca = PCA(n_components = 3)
		X_pca = pca.fit_transform(X)
		print('going into plot function')
		self._plot_embedding(X_pca, y, label_map)


	def predict_one(self, lyric, show = 10, verbose = False):
		"""Predicts artists for one lyric
		
		Parameters
		----------
		lyric: string
		show: int
		verbose: bool
		
		Returns
		-------
		artists: list
		"""
		lyric = lyric.lower()
		pred = self.pipe.predict_proba(pd.Series([lyric]))
		pred = pred.tolist()[0]
		sorted_artists = sorted(set(list(self.data.artist.values)))
		elems = heapq.nlargest(show, pred)
		if verbose:
			print(f'Artist who would most likely say "{lyric}":\n')
		artists = []
		for i, elem in enumerate(elems):
			idx = pred.index(elem)
			if verbose:
				print(f'{i+1}) {sorted_artists[idx]}')
			artists.append(sorted_artists[idx])
			if i == show:
				break
		return artists


	@classmethod
	def load(cls, file = '../models/artist_classifier.pickle'):
		"""Loads the pickled model
		
		Parameters
		----------
		cls: Classifier
		file: string
		
		Returns
		-------
		instance: Classifier
		"""
		file = open(file, "rb")
		instance = pickle.load(file)
		file.close()
		return instance

	
	def save(self, file = '../models/artist_classifier.pickle'):
		"""Saves the model as pickle
		
		Parameters
		----------
		file: string
		"""
		file = open(file, "wb")
		pickle.dump(self, file)
		file.close()


		
class LyricPredictor:
	"""Loads pickled model and casts predictions"""

	def __init__(self):
		self.load_model()

	def predict(self, lyric):
		"""Predicts artists given a lyric
		
		Parameters
		----------
		lyric: string
		"""
		artist_predictions = self.model.predict_one(lyric)
		return artist_predictions

	def load_model(self):
		"""Loads model using Classifier.load() method"""
		sg = Classifier()
		self.model = sg.load()
		
		
		
def refit_model(print_accuracy = False, 
				print_report = False, 
				explode = True,
				snip_len = 30, 
				plot_pca = False, 
				verbose = True, 
				over_sample = True,
				under_sample = False, 
				save_model = False,
				vector_params = {'ngram_range' : (1, 4)}):
	"""Retrains the model
		
	Parameters
	----------
	print_accuracy: bool
	print_report: bool
	explode: bool
	snip_len: int
	plot_pca: bool
	verbose: bool
	over_sample: bool
	under_sample: bool
	save_model: bool
	vector_params: dict
	"""
	
	# loading in data
	dm = DataManager()
	print("loading data" if verbose else "", end = "")
	now = datetime.now()
	if explode:
		clean_data = dm.load_data(explode = snip_len)
	else:
		clean_data = dm.load_data()
	print(f" - {datetime.now() - now}" if verbose else "")

	# fitting and testing model
	print("fitting model" if verbose else "", end = "")
	now = datetime.now()
	sg = Classifier(clean_data, model = SGDClassifier)
	sg.fit(vector_params, over_sample, under_sample)
	sg.predict()
	print(f" - {datetime.now() - now}" if verbose else "")

	# optional print statements for metrics
	if print_accuracy:
		sg.accuracy()
	elif print_report:
		sg.report()
	elif plot_pca and explode is False:
		sg.plot_artist_labels()

	# saving model as pickle
	if save_model:
		print("saving model" if verbose else "", end = "")
		now = datetime.now()
		sg.save()
		print(f" - {datetime.now() - now}" if verbose else "")

		print("saved model!" if verbose else "")


def load_model():
	"""Retrains the model
		
	Returns
	----------
	model: Classifier
	"""
	sg = Classifier()
	model = sg.load()
	return model


