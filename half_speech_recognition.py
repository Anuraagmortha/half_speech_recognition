import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as pltfrom
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout,Flatten
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2Dfrom
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

# Commented out IPython magic to ensure Python compatibility.

# %matplotlib inline
audios_dir = r'your_audio_files_directory_path'

words=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

data = []
labels = []
dirs = os.listdir(audios_dir)
for d in dirs:
    if d in words:
        print(d)
        files = os.listdir(os.path.join(audios_dir, d))
        audios = [f for f in files if f.endswith('.wav')]
        audios.sort()
        count = 0  # Track the number of files processed for this directory
        for file in audios:

            if count >= 700:
                break  # Stop processing files if we've reached the limit
            sample_rate, samples = wavfile.read(os.path.join(audios_dir, d, file))
            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
            if spectrogram.shape[0] >= 128 and spectrogram.shape[1] >= 48:
                data.append(spectrogram[:128, :48])
                labels.append(d)
                count += 1  # Increment the count of processed files for this directory

data = np.array(data)

values,count = np.unique(labels,return_counts=True)

#plot
plt.figure(figsize=(10,5))
index = np.arange(len(words))
plt.bar(index, count)
plt.xlabel('Commands', fontsize=12)
plt.ylabel('No of samples', fontsize=12)
plt.xticks(index, words, fontsize=15, rotation=60)
plt.title('No. of samples for each command')
plt.show()

labels_backup = labels

le = LabelEncoder()
ls = le.fit_transform(labels)
labels_categoricals = to_categorical(ls)

# partition the data into training,cross-validation and testing splits using 60%,20% and 20% of data
(trainX, testX, trainY, testY) = train_test_split(data, labels_categoricals,test_size=0.20, stratify=labels_categoricals, random_state=42)
(trainX, cvX, trainY, cvY) = train_test_split(trainX, trainY,test_size=0.25, stratify=trainY, random_state=42)

# input image dimensions
img_rows, img_cols = 128, 48

if K.image_data_format() == 'channels_first':
    trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
    testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
    cvX = cvX.reshape(cvX.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    trainX = trainX.reshape(trainX.shape[0],img_rows, img_cols,1)
    testX = testX.reshape(testX.shape[0],img_rows, img_cols,1)
    cvX = cvX.reshape(cvX.shape[0],img_rows, img_cols,1)
    input_shape = (img_rows, img_cols,1)

#initialize the initial learning rate, number of epochs to train for and batch size
INIT_LR = 1e-3
EPOCHS = 100
batch_size = 64
num_class = 10

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_class, activation='softmax'))

model.summary()

#from tensorflow.keras.optimizers import Adam
#opt = Adam(lr=INIT_LR)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.0001)
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit(trainX, trainY, batch_size=batch_size, epochs=EPOCHS, verbose=1, callbacks=[es,mc],validation_data=(cvX, cvY))

# make predictions on the testing set
predIdxs = model.predict(testX, batch_size=batch_size)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

def plot_Confusion_Matrix(actual_labels, predict_labels, title, words):
    """This function plots the confusion matrix"""
    cm = confusion_matrix(actual_labels, predict_labels)
    classNames = words
    cm_data = pd.DataFrame(cm, index=classNames, columns=classNames)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm_data, annot=True, fmt="d")
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

plot_Confusion_Matrix(testY.argmax(axis=1), predIdxs, "Confusion Matrix", words)

# Calculate accuracy
accuracy = accuracy_score(testY.argmax(axis=1), predIdxs)
print("Accuracy:", accuracy)

# Define a function to create the model
def create_model(optimizer='adam', dropout_rate=0.5):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_class, activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create Keras classifier for grid search
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define hyperparameters to tune
param_dist = {
    'optimizer': ['adam', 'rmsprop'],  # optimizer options
    'dropout_rate': [0.2, 0.3, 0.4, 0.5]  # dropout rate options
}

# Create GridSearchCV object
random_search=RandomizedSearchCV(estimator=model,param_distributions=param_dist,n_iter=3,cv=3)
# Perform grid search
random_search.fit(trainX,trainY)

# Summarize results
print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))

# Get the best hyperparameters from the grid search results
best_params = random_search.best_params_
best_optimizer = best_params['optimizer']
best_dropout_rate = best_params['dropout_rate']

# Create a new model with the best hyperparameters
best_model = create_model(optimizer=best_optimizer, dropout_rate=best_dropout_rate)

# Train the model with the best hyperparameters
history = best_model.fit(trainX, trainY, batch_size=batch_size, epochs=EPOCHS, verbose=1, callbacks=[es, mc], validation_data=(cvX, cvY))

# Load the best model saved during training
best_model = load_model('best_model.hdf5')

# Evaluate the best model on the test set
test_loss, test_accuracy = best_model.evaluate(testX, testY, verbose=0)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# make predictions on the testing set
predIdxs = best_model.predict(testX, batch_size=batch_size)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# Calculate accuracy
accuracy = accuracy_score(testY.argmax(axis=1), predIdxs)
print("Accuracy:", accuracy)
