import numpy as np
from tensorflow.keras.models import Model, load_model
from load_data import load_dataset

#Load the trained model.
loaded_model = load_model("Saved_Model/Model.h5")
loaded_model.set_weights(loaded_model.get_weights())
#Discard the Softmax layer, Second last layer provides the latent feature representation.

matrix_size = loaded_model.layers[-2].output.shape[1]
new_model = Model(loaded_model.inputs, loaded_model.layers[-2].output)
print(new_model.summary())

images, labels = load_dataset(verbose=1, mode="Test")
images = np.expand_dims(images, axis=1)
#print(len(images))
#print(len(labels))
print(images.shape)
print(len(labels))
#Normalize the image.
images = images / 255.

#display the list of available song 

decodeDict = {'151404': 'Seconds To Mars - Night of the hunter', '152103': 'Afrojack - The spark', '152253': 'Alesso - Heros', 
              '152254': 'Awolnation - Sail', '152258': 'Boyce Avenue - Wonderwall', '152261': 'Bruno Mars - Just the way you are', 
              '152262': 'Bruno Mars - Locked out of heaven', '152324': 'Calvin Harris - Summer', '152418': 'Calvin Harris - Sweet Nothing',
              '152425': 'Coldplay - Magic', '152480': 'Coldplay - Paradise', '152543': 'Coldplay - Viva La Vida', '152545': 'Coldplay - The Scientist', 
              '152568': 'Daft Punk - Instant crush', '152569': 'Daft Punk - Lose yourself to dance', '152570': 'Don Omar - Danza Kuduro', 
              '153337': 'Enrique Iglesias - Bailando', '153383': 'Imagine Dragons - Demons', '153452': 'Imagine Dragons - Its Time', 
              '153946': "Jennifer Lopez - On the floor ", '153955': 'John Mayer - Say', '153956': 'Kanye West - Stronger', 
              '154303': 'Katy Perry - Dark Horse', '154305': 'Katy Perry - Fireworks', '154306': 'Khalid - Location', '154307': 'Lana Del Rey - Young and Beautiful', 
              '154308': 'Maroon5 - Moves Like Jagger', '154309': 'Passenger - Let Her Go', '154413': 'Wiz Khalifa - Black and Yellow', '154414': 'Wiz Khalifa - Young, Wild and Free'}
for i in np.unique(labels):
    #print(decodeDict[i])
    print(decodeDict.get('i'))

decodeDictReverse = {x: y for y, x in decodeDict.items()}
recommend_wrt = input("Enter Song name:\n")
recommend_wrt = decodeDictReverse[recommend_wrt]

prediction_anchor = np.zeros((1, matrix_size))
count = 0
predictions_song = []
predictions_label = []
counts = []
distance_array = []

# Calculate the latent feature vectors for all the songs.
for i in range(0, len(labels)):
    if labels[i] == recommend_wrt:
        test_image = images[i]
        test_image = np.expand_dims(test_image, axis=0)
        prediction = new_model.predict(test_image)
        prediction_anchor = prediction_anchor + prediction
        count = count + 1
    elif labels[i] not in predictions_label:
        predictions_label.append(labels[i])
        test_image = images[i]
        test_image = np.expand_dims(test_image, axis=0)
        prediction = new_model.predict(test_image)
        predictions_song.append(prediction)
        counts.append(1)
    elif labels[i] in predictions_label:
        index = predictions_label.index(labels[i])
        test_image = images[i]
        test_image = np.expand_dims(test_image, axis=0)
        prediction = new_model.predict(test_image)
        predictions_song[index] = predictions_song[index] + prediction
        counts[index] = counts[index] + 1
# Count is used for averaging the latent feature vectors.
prediction_anchor = prediction_anchor / count
for i in range(len(predictions_song)):
    predictions_song[i] = predictions_song[i] / counts[i]
    # Cosine Similarity - Computes a similarity score of all songs with respect
    # to the anchor song.
    distance_array.append(np.sum(prediction_anchor * predictions_song[i]) / (np.sqrt(np.sum(prediction_anchor**2)) * np.sqrt(np.sum(predictions_song[i]**2))))

distance_array = np.array(distance_array)
recommendations = 0

print("Recommendation is:")

# Number of Recommendations is set to 2.
while recommendations < 2:
    index = np.argmax(distance_array)
    value = distance_array[index]
    print("Song Name: " + decodeDict[predictions_label[index]] + " with value = %f" % value)
    distance_array[index] = -np.inf
    recommendations = recommendations + 1

""" print(np.unique(labels))
# Enter a song name which will be an anchor song.
recommend_wrt = input("Enter Song name:\n")
prediction_anchor = np.zeros((1, matrix_size))
count = 0
predictions_song = []
predictions_label = []
counts = []
distance_array = []

# Calculate the latent feature vectors for all the songs.
for i in range(0, len(labels)):
    if labels[i] == recommend_wrt:
        test_image = images[i]
        test_image = np.expand_dims(test_image, axis=0)
        prediction = new_model.predict(test_image)
        prediction_anchor = prediction_anchor + prediction
        count = count + 1
    elif labels[i] not in predictions_label:
        predictions_label.append(labels[i])
        test_image = images[i]
        test_image = np.expand_dims(test_image, axis=0)
        prediction = new_model.predict(test_image)
        predictions_song.append(prediction)
        counts.append(1)
    elif labels[i] in predictions_label:
        index = predictions_label.index(labels[i])
        test_image = images[i]
        test_image = np.expand_dims(test_image, axis=0)
        prediction = new_model.predict(test_image)
        predictions_song[index] = predictions_song[index] + prediction
        counts[index] = counts[index] + 1
# Count is used for averaging the latent feature vectors.
prediction_anchor = prediction_anchor / count
for i in range(len(predictions_song)):
    predictions_song[i] = predictions_song[i] / counts[i]
    # Cosine Similarity - Computes a similarity score of all songs with respect
    # to the anchor song.
    distance_array.append(np.sum(prediction_anchor * predictions_song[i]) / (np.sqrt(np.sum(prediction_anchor**2)) * np.sqrt(np.sum(predictions_song[i]**2))))

distance_array = np.array(distance_array)
recommendations = 0

print("Recommendation is:")

# Number of Recommendations is set to 2.
while recommendations < 2:
    index = np.argmax(distance_array)
    value = distance_array[index]
    print("Song Name: " + predictions_label[index] + " with value = %f" % value)
    distance_array[index] = -np.inf
    recommendations = recommendations + 1
 """