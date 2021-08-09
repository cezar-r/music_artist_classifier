from flask import Flask, render_template, request
from artist_classifier import Classifier
from datetime import datetime

app = Flask(__name__)

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


clf = LyricPredictor()

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
	lyric = str(request.form['lyric'])
	lyric = lyric.capitalize()
	lyric = '"' + lyric + '"'
	artists = clf.predict(lyric)
	artists = [i.title() for i in artists]
	return render_template('index_pred.html', data = (artists, lyric))


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, debug=True)
