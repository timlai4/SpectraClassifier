import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from keras import regularizers

def generate_data():
    index = 0
    while True:
        if index > 4341:
            index = 0
        input_path = 'ham/inputs' + str(index)
        input_batch = np.load(input_path,allow_pickle=True)
        label_path = 'ham/labels' + str(index)
        label_batch = np.load(label_path,allow_pickle=True)
        index += 1
        yield (input_batch, label_batch)


trainGen = generate_data()

model = tf.keras.Sequential()
# Adds a densely-connected layer with 8000 units to the model:
model.add(layers.Dense(8000, input_dim = 16000, activation='relu'))
# Add another:
model.add(layers.Dense(8000, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(8000, activation='relu'))

model.add(layers.Dense(8000, activation='relu'))
model.add(layers.Dropout(0.25))

model.add(layers.Dense(8000, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1, activation='sigmoid'))
class_weights = {0: 1, 1: 76/20}
model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

model.fit_generator(trainGen, steps_per_epoch = 4341, epochs = 15,
                    class_weight = class_weights, use_multiprocessing = True)

model.save('my_model.h5')

ham_ptest = np.load('ham_ptest',allow_pickle=True)
with open('ham_ntest','rb') as f:
    ham_ntest = pickle.load(f)

ham_ntest = np.array(ham_ntest)

test_plabels = [np.array([1]) for i in range(len(ham_ptest))]
test_nlabels = [np.array([0]) for i in range(len(ham_ntest))]
test_labels = np.array(test_plabels + test_nlabels)

test_inputs = np.vstack((ham_ptest,ham_ntest))

score = model.evaluate(test_inputs, test_labels, batch_size = 32,
                       use_multiprocessing = True)

print(score)
