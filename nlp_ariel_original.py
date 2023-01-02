import warnings

warnings.filterwarnings('ignore')
import os
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D  # , AveragePooling1D
from keras.layers import Flatten, Dropout, Activation  # Input,
from keras.layers import Dense  # , Embedding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

"""### Setup the Basic Paramter"""

dataset_path = os.path.abspath('./Dataset')
destination_path = os.path.abspath('./')
# To shuffle the dataset instances/records
randomize = True
# for spliting dataset into training and testing dataset
split = 0.8
# Number of sample per second e.g. 16KHz
sampling_rate = 20000
emotions = ["positive", "negative", "neutral"]

from utils import dataset

df, train_df, test_df = dataset.create_and_load_meta_csv_df(dataset_path, destination_path, randomize, split)
print('Dataset samples  : ', len(df), "\nTraining Samples : ", len(train_df), "\ntesting Samples  : ", len(test_df))

"""
### Labels Assigned for emotions : 
- 0 : anger
- 1 : disgust
- 2 : fear
- 3 : happy
- 4 : neutral 
- 5 : sad
- 6 : surprise
"""

"""#Data Pre-Processing
### Getting the features of audio files using librosa
Calculating MFCC, Pitch, magnitude, Chroma features.
"""

from utils.feature_extraction import *

# trainfeatures_noised, trainlabel_noised = get_features_dataframe(train_df, sampling_rate,add_noise=True)
# trainfeatures_noised.to_pickle('./features_dataframe/trainfeatures_noise')
# trainlabel_noised.to_pickle('./features_dataframe/trainlabel_noise')
#
# trainfeatures, trainlabel = get_features_dataframe(train_df, sampling_rate)
# trainfeatures.to_pickle('./features_dataframe/trainfeatures')
# trainlabel.to_pickle('./features_dataframe/trainlabel')
#
# testfeatures, testlabel = get_features_dataframe(test_df, sampling_rate)
# testfeatures.to_pickle('./features_dataframe/testfeatures')
# testlabel.to_pickle('./features_dataframe/testlabel')


# I have ran above 2 lines and get the featured dataframe.
# and store it into pickle file to use it for later purpose.
# it take too much time to generate features(around 30-40 minutes).


trainfeatures = pd.read_pickle('./features_dataframe/trainfeatures')
trainlabel = pd.read_pickle('./features_dataframe/trainlabel')


testfeatures = pd.read_pickle('./features_dataframe/testfeatures')
testlabel = pd.read_pickle('./features_dataframe/testlabel')

trainfeatures = trainfeatures.fillna(0)
testfeatures = testfeatures.fillna(0)

# By using .ravel() : Converting 2D to 1D e.g. (512,1) -> (512,). To prevent DataConversionWarning

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel).ravel()
X_test = np.array(testfeatures)
y_test = np.array(testlabel).ravel()

# One-Hot Encoding
lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

"""### Changing dimension for CNN model"""

x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)

"""# 6. Model Creation"""

model = Sequential()
model.add(Conv1D(256, 5, padding='same',
                 input_shape=(x_traincnn.shape[1], x_traincnn.shape[2])))
model.add(Activation('relu'))
model.add(Conv1D(128, 5, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5, padding='same', ))
model.add(Activation('relu'))
model.add(Conv1D(128, 5, padding='same', ))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

## train:
cnnhistory = model.fit(x_traincnn, y_train, batch_size=32, epochs=50, validation_data=(x_testcnn, y_test))

# evaluate model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = model.evaluate(x_testcnn, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))


