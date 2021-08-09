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


	def _clean_col(self, x):
		if type(x) == float:
				return 0
		else:
			return x.strip(f'\r')


	def _clean_text(self, x):
		return REGEX.sub(' ', x)


	def _snippets(self, x):
		return ' '.join(x.split(' ')[-50:])


	def _explode(self, df, snip_len):
		df = pd.concat([pd.Series(row['artist'], 
			[' '.join(row['lyrics'].split(' ')[i: i+snip_len]) for i in range(0, len(row['lyrics'].split(' ')), snip_len)]) for _, row in df.iterrows() 
		]).reset_index()
		df.columns = ['lyrics', 'artist']
		return df


	def load_data(self, filename="test", snippets = False, explode = False):
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

	def __init__(self, data = None, model = SGDClassifier):
		self.data = data
		self.model = model(loss = 'log')
		if self.data is not None:
			self._split_data()


	def _split_data(self):
		X = self.data.lyrics
		y = self.data.artist
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = .2)

	def _arr_to_str(self, x):
		if type(x) == list:
			return " ".join(x)
		else:
			return x

	def fit(self,  vector_params, over_sample = False, under_sample = False):
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
		self.y_hat = self.pipe.predict(self.X_test)
		

	def accuracy(self):
		self.accuracy = accuracy_score(self.y_test, self.y_hat)
		if verbose:
			print(self.accuracy)
		return self.accuracy


	def report(self, verbose = True):
		if verbose:
			print(classification_report(self.y_test, self.y_hat))
		return classification_report(self.y_test, self.y_hat)



	def _get_data(self, df):
		"""Labels data in DatFrame

		Parameters
		----------
		df: pd.DataFrame
			all data
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
	    data: arr
	    	list of all lyrics
	    """
		tfidf = TfidfVectorizer()
		X = tfidf.fit_transform(data).toarray()
		return X, np.array(tfidf.get_feature_names())



	def _plot_embedding(self, X, y, label_map, title=None):
	    """Plot an embedding of the mnist dataset onto a plane.
	    
	    Parameters
	    ----------
	    ax: matplotlib.axis object
	      The axis to make the scree plot on.
	      
	    X: numpy.array, shape (n, 2)
	      A two dimensional array containing the coordinates of the embedding.
	      
	    y: numpy.array
	      The labels of the datapoints.  Should be digits.
	      
	    title: str
	      A title for the plot.
	    """
	    x_min, x_max = np.min(X, 0), np.max(X, 0)
	    X = (X - x_min) / (x_max - x_min)

	    plt.rcParams["figure.figsize"] = (18, 7)
	    # plt.set_size_inches(15, 8)

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
	    # plt.show()



	def _clean_report(self, report):
		report = report.replace('\n', '')
		new_str = [report[0+i:66+i] for i in range(0, len(report), 66)]
		scores_dict = {}
		for string in new_str[1:-3]:
			line = string.split('   ')
			# accuracy = line[-4]
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
		if x in self.top_ten_artists:
			return x
		else:
			return 'other'


	def _relabel_df(self):
		# get top artists
		# if artist != top artist, label as other
		report = self.report()
		report = self._clean_report(report)
		self.top_ten_artists = list(report.keys())[:9]
		new_df = self.data
		new_df['artist'] = new_df['artist'].apply(self._relabel_rows)
		return new_df


	def plot_artist_labels(self):
		"""Parent function to plotting genre cluster graph

		Parameters
		----------
		df: pd.DataFrame
			all data
		"""
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
		file = open(file, "rb")
		instance = pickle.load(file)
		file.close()

		return instance

	def save(self, file = '../models/artist_classifier.pickle'):
		file = open(file, "wb")
		pickle.dump(self, file)
		file.close()


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
	sg = Classifier()
	model = sg.load()
	return model



# lyric_samples = ['drank in my cup']


# if __name__ == '__main__':
# 	# refit_model(print_report = True, save_model = True)

# 	model = load_model()
# 	for lyric in lyric_samples:
# 		artists = model.predict_one(lyric, show = 10)
# 		print(artists)
# 		print(' ')


'''

PARAMS

snip_len = None, explode = False,  over_ampling = True, ngrams = (1, 4), max_features = None, min_df = None
26% accuracy 6.5/10 

snip_len = 15, over_sampling = True, ngrams = (1, 4), max_features = None, min_df = None
34% accuracy = 6/10

snip_len = 15, over_sample = True, ngrams = (1, 5),max_featues = None, min_df = None
40% accuracy 6.5/10

snip_len = 15, over_sample = True, ngrams = (1, 10), max_feature = None, min_df = None
42% accuracy

snip_len = 20, over_sample = True, ngram = (1, 4)
45% accuracy 7/10 9gb model

snip_len = 20, over_sample = True, ngram = (1, 5)
45% accuracy 6/10 12 gb model

snip_len 25, over_sample = True, ngram = (1, 4)
49% accuracy 7/10

snip_len = 25, over_sample = True, ngram = (1, 5)
50% accuracy 7/10

snip_len = 30, over_sample = True, ngram = (1, 4)
52% acccuracy 7/10

snip_len - 30, over_sample = True, ngram = (1, 5)
53% accuracy 7/10

snip_len = 50, over_sample = True, ngram = (1, 4)
60% accuracy 


'''

# TODO
# start building flask app

'''
lyric_samples = ['sit down be humble',
				'man i feel just like a rockstar',
				'vomit on his sweater already moms spaghetti',
				'i just fucked your bitch in some gucci flip flops',
				'drown my pain in all these drugs',
				'im a motherfuckin starboy',
				'codeine in my piss got pills in my shit',
				'cant keep my dick in my pants',
				'if cops pull up i put that crack in my crack',
				'blueface baby aight',
				'money got longer speaker got louder car got faster turn to a savage',
				'young savage why you trappin so hard',
				'pullin out the coupe at the lot',
				'i just hit a lick with the box had to put the stick in the box',
				'got a beer in my hand guitar in the other',
				'yea i love my tractor im from alabama',
				'gonna take my horse from my old town road',
				'dont you run from me',
				'ill kill all of you',
				'my bitch dont love me no more she kick me out im like bro',
				'fucked up fucked up fucked up fucked up',
				'in new york i millie rock hide it in my sock',
				'said they all wanna live but i just wanna die',
				'i still see your shadows in my room',
				'lil pussy know he done fucked up',
				'she like how i smell cologne',
				'gucci gang gucci gang gucci gang',
				'slaughter gang shit murder gang shit',
				'got pills in my shit dont think i can quit',
				'ridin on a tractor through alabama sippin on beer',
				'copped a bmw new deposit',
				'ric flair drip go woo on a bitch',
				'christian dior dior when it rains it pours im up in all the stores',
				'bust down thotiana i wanna see you bust down',
				'woke up to niggas talkin like me',
				'i love darkskins love her melanin',
				'i told my bitch to go to the store cuz we ran outta soda',
				'green gun particle turn you into molecules',
				'told that lil bitch that im with it',
				'i just be dancin right up on that bitch what you be stuntin',
				'wasnt no food but rice i give these bitches one night',
				'hundred round drum on the bottom of that shit',
				'pull up on yo bitch she sucked my dick and then you kissed her',
				'i just smoked a wood bitch im from the hood',
				'man you washed up young savage man i got my car washed up']
'''
