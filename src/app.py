#!/usr/bin/env python
# coding: utf-8

"""
Runs the Flask web application.
"""

from flask import Flask, render_template, request
from artist_classifier import LyricPredictor
from datetime import datetime

app = Flask(__name__)


clf = LyricPredictor()

def clean_input(lyric):
	lyric = lyric.lower()
	clean_lyric = ""
	for word in lyric.split(' '):
		if word not in stop_words:
			for char in bad_chars:
				word = word.replace(char, "")
			clean_lyric += word + ' '
	clean_lyric = " ".join(clean_lyric.split())
	return clean_lyric[:-1]



@app.route('/')
def index():
	return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
	og_lyric = str(request.form['lyric'])
	lyric = clean_input(og_lyric)
	lyric = lyric.capitalize()
	artists = clf.predict(lyric)
	artists = [i.title() for i in artists]
	og_lyric = '"' + og_lyric + '"'
	return render_template('index_pred.html', data = (artists, og_lyric))


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, debug=True)
