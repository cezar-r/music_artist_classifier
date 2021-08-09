import pickle
from datetime import datetime
from modeling import SongRecommender

def test_model():
	model_file = open('../models/songrecommender.pickle', 'rb')
	model = pickle.load(model_file)
	model_file.close()
	return model


model = test_model()
print(len(model._song_neighbors.keys()))

print(model)
save_model = open('../models/songrecommender2.pickle', 'wb')
pickle.dump(model, save_model)
save_model.close()