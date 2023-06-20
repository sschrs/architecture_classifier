from keras import models
from keras.utils import load_img, img_to_array
import pandas as pd
import numpy as np
import sys
import warnings

classes = ['Achaemenid architecture', 
           'American Foursquare architecture', 
           'American craftsman style', 
           'Ancient Egyptian architecture', 
           'Art Deco architecture', 
           'Art Nouveau architecture', 
           'Baroque architecture', 
           'Bauhaus architecture', 
           'Beaux-Arts architecture', 
           'Byzantine architecture', 
           'Chicago school architecture', 
           'Colonial architecture', 
           'Deconstructivism', 
           'Edwardian architecture', 
           'Georgian architecture', 
           'Gothic architecture', 
           'Greek Revival architecture', 
           'International style', 
           'Novelty architecture', 
           'Palladian architecture', 
           'Postmodern architecture', 
           'Queen Anne architecture', 
           'Romanesque architecture', 
           'Russian Revival architecture', 
           'Tudor Revival architecture']

def predict(path):
    model = models.load_model('/Users/suleyman/Downloads/arc_tuned.h5')
    img = load_img(path, target_size = (224, 224))
    img_tensor = img_to_array(img)
    img_tensor = np.array([img_tensor]) # set shape as (1, 224, 224, 3)

    prediction = model.predict(img_tensor)

    # create a dataframe of prediction
    prediction_frame = pd.DataFrame(prediction)
    prediction_frame.columns = classes
    prediction_frame = prediction_frame.T.reset_index()
    prediction_frame.columns = ['Class','Proba']

    prediction_frame['Proba'] = round(prediction_frame['Proba'],2) * 100 # adapt values to percent representation
    prediction_frame = prediction_frame.sort_values(by='Proba', ascending=False) # sort classes according to their probabilities
    
    # if proba==100 return one column
    if prediction_frame.iloc[0,1] == 100:
       return prediction_frame.iloc[:1,:]
    
    # return the three most likely classes
    return prediction_frame[:3]

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    if (len(sys.argv) >= 2):
        print(predict(sys.argv[1]))
