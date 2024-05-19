import nltk
import numpy as np
import pandas as pd
import re
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Define the path to your dataset
data_path = '../training_set_rel3.tsv'

df = pd.read_csv(data_path, delimiter='\t', encoding='latin1')

# Display the first few rows of the dataset
print(df.head())

# Lists to hold essays based on their scores
list_1 = []
list_2 = []
list_3 = []
list_4 = []
list_5 = []
list_6 = []
list_7 = []
list_8 = []
list_9 = []
list_10 = []

# Iterate through the dataframe and append essays to the corresponding lists
for i, j in zip(range(len(df)), df["essay"]):
    if df["domain1_score"][i] == 1:
        list_1.append(j)
    elif df["domain1_score"][i] == 2:
        list_2.append(j)
    elif df["domain1_score"][i] == 3:
        list_3.append(j)
    elif df["domain1_score"][i] == 4:
        list_4.append(j)
    elif df["domain1_score"][i] == 5:
        list_5.append(j)
    elif df["domain1_score"][i] == 6:
        list_6.append(j)
    elif df["domain1_score"][i] == 7:
        list_7.append(j)
    elif df["domain1_score"][i] == 8:
        list_8.append(j)
    elif df["domain1_score"][i] == 9:
        list_9.append(j)
    elif df["domain1_score"][i] == 10:
        list_10.append(j)

# Combine all lists into a single data list
import itertools
data = list(itertools.chain(list_1, list_2, list_3, list_4, list_5, list_6, list_7, list_8, list_9, list_10))

# Check the length of the combined data list
print(len(data))

def gen_num(num, length):
    return [num] * length

# Generate score lists
score_1 = gen_num(1, len(list_1))
score_2 = gen_num(2, len(list_2))
score_3 = gen_num(3, len(list_3))
score_4 = gen_num(4, len(list_4))
score_5 = gen_num(5, len(list_5))
score_6 = gen_num(6, len(list_6))
score_7 = gen_num(7, len(list_7))
score_8 = gen_num(8, len(list_8))
score_9 = gen_num(9, len(list_9))
score_10 = gen_num(10, len(list_10))

# Combine all score lists into a single score list
score = list(itertools.chain(score_1, score_2, score_3, score_4, score_5, score_6, score_7, score_8, score_9, score_10))

# Check the length of the combined score list
print(len(score))

# Create a dictionary of lists
data_dict = {'essay': data, 'score': score}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data_dict)

# Display the first few rows of the new DataFrame
print(df.head())

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess text data
corpus = []
for i in range(len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['essay'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

print(corpus[:5])  # Display the first 5 preprocessed essays


# Initialize the tokenizer and fit on the cleaned essays
tokenizer = Tokenizer(num_words=5000)  # Adjust the number of words as needed
tokenizer.fit_on_texts(corpus)

# Convert the cleaned essays to sequences
sequences = tokenizer.texts_to_sequences(corpus)

# Pad the sequences to ensure uniform length
padded_sequences = pad_sequences(sequences, maxlen=500)  # Adjust maxlen as needed

# Display the shape of the padded sequences
print(padded_sequences.shape)


def essay_to_wordlist(essay_v, remove_stopwords):
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words

def essay_to_sentences(essay_v, remove_stopwords):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def make_feature_vec(words, model, num_features):
    feature_vec = np.zeros((num_features,), dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            num_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if num_words > 0:
        feature_vec = np.divide(feature_vec, num_words)
    return feature_vec

def get_avg_feature_vecs(essays, model, num_features):
    counter = 0
    essay_feature_vecs = np.zeros((len(essays), num_features), dtype="float32")
    for essay in essays:
        essay_feature_vecs[counter] = make_feature_vec(essay, model, num_features)
        counter = counter + 1
    return essay_feature_vecs

# Tokenize essays into sentences
sentences = []
for essay in df['essay']:
    sentences += essay_to_sentences(essay, remove_stopwords=True)

# Initialize and train the Word2Vec model
num_features = 300  # Size of the word vectors
min_word_count = 40  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

print("Training Word2Vec model...")
word2vec_model = Word2Vec(sentences, workers=num_workers, vector_size=num_features, min_count=min_word_count, window=context, sample=downsampling)
word2vec_model.init_sims(replace=True)

# Save the model for later use
word2vec_model.wv.save_word2vec_format('../word2vec.bin', binary=True)

# Create average feature vectors for the essays
clean_essays = []
for essay in df['essay']:
    clean_essays.append(essay_to_wordlist(essay, remove_stopwords=True))
essay_feature_vecs = get_avg_feature_vecs(clean_essays, word2vec_model, num_features)

# Display the shape of the feature vectors
print(essay_feature_vecs.shape)


# First split into train and temp sets
X_train, X_temp, y_train, y_temp = train_test_split(essay_feature_vecs, df['score'], test_size=0.3, random_state=42)

# Split temp into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Display the shapes of the datasets
print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)

import tensorflow as tf

# Check available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)
else:
    print("No GPU available")


# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def get_model():
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy', 'mae'])
    model.summary()

    return model

# Reshape the feature vectors for LSTM
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Get the model
lstm_model = get_model()

# Train the model
history = lstm_model.fit(X_train_reshaped, y_train, epochs=300, batch_size=64, validation_data=(X_val_reshaped, y_val))

# Display the training history
print(history.history)


# Make predictions on the test set
y_pred = lstm_model.predict(X_test_reshaped)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
y_pred_rounded = np.round(y_pred).astype(int)
kappa = cohen_kappa_score(y_test, y_pred_rounded, weights='quadratic')

# Print evaluation metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Cohen\'s Kappa Score: {kappa}')


# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def get_model():
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy', 'mae'])
    model.summary()

    return model

# Reshape the feature vectors for LSTM
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Get the model
lstm_model = get_model()

# Define early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('../best_lstm_model.h5', save_best_only=True, monitor='val_loss')

# Train the model with early stopping and model checkpoint
history = lstm_model.fit(X_train_reshaped, y_train, epochs=400, batch_size=64, validation_data=(X_val_reshaped, y_val),
                         callbacks=[early_stopping, model_checkpoint])

# Display the training history
print(history.history)


# Load the best model
best_model = tf.keras.models.load_model('../best_lstm_model.h5')

# Make predictions on the test set
y_pred = best_model.predict(X_test_reshaped)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
y_pred_rounded = np.round(y_pred).astype(int)
kappa = cohen_kappa_score(y_test, y_pred_rounded, weights='quadratic')

# Print evaluation metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Cohen\'s Kappa Score: {kappa}')



def model_builder(hp):
    model = Sequential()

    # Tune the number of units in the first LSTM layer
    hp_units1 = hp.Int('units1', min_value=32, max_value=512, step=32)
    model.add(LSTM(units=hp_units1, dropout=hp.Float('dropout1', min_value=0.2, max_value=0.5, step=0.1),
                   recurrent_dropout=hp.Float('recurrent_dropout1', min_value=0.2, max_value=0.5, step=0.1),
                   input_shape=[1, 300], return_sequences=True))

    # Tune the number of units in the second LSTM layer
    hp_units2 = hp.Int('units2', min_value=32, max_value=512, step=32)
    model.add(LSTM(units=hp_units2, dropout=hp.Float('dropout2', min_value=0.2, max_value=0.5, step=0.1),
                   recurrent_dropout=hp.Float('recurrent_dropout2', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(Dropout(hp.Float('dropout3', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='relu'))

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate),
                  loss='mean_squared_error',
                  metrics=['accuracy', 'mae'])

    return model

tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=30,
                     factor=3,
                     directory='keras_tuner',
                     project_name='lstm_tuning')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

tuner.search(X_train_reshaped, y_train, epochs=50, validation_data=(X_val_reshaped, y_val), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of units in the first LSTM layer is {best_hps.get('units1')} with a dropout rate of {best_hps.get('dropout1')} and recurrent dropout rate of {best_hps.get('recurrent_dropout1')}.
The optimal number of units in the second LSTM layer is {best_hps.get('units2')} with a dropout rate of {best_hps.get('dropout2')} and recurrent dropout rate of {best_hps.get('recurrent_dropout2')}.
The optimal dropout rate for the final layer is {best_hps.get('dropout3')}.
The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# Build the best model and train it
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(X_train_reshaped, y_train, epochs=400, batch_size=64, validation_data=(X_val_reshaped, y_val),
                         callbacks=[early_stopping, model_checkpoint])

# Display the training history
print(history.history)

# Load the best model
best_model.save('../best_lstm_model_tuned.h5')
best_model = tf.keras.models.load_model('../best_lstm_model_tuned.h5')

# Make predictions on the test set
y_pred = best_model.predict(X_test_reshaped)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
y_pred_rounded = np.round(y_pred).astype(int)
kappa = cohen_kappa_score(y_test, y_pred_rounded, weights='quadratic')

# Print evaluation metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Cohen\'s Kappa Score: {kappa}')

# Evaluate on training set
y_train_pred = best_model.predict(X_train_reshaped)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
y_train_pred_rounded = np.round(y_train_pred).astype(int)
train_kappa = cohen_kappa_score(y_train, y_train_pred_rounded, weights='quadratic')

# Evaluate on validation set
y_val_pred = best_model.predict(X_val_reshaped)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_val, y_val_pred)
y_val_pred_rounded = np.round(y_val_pred).astype(int)
val_kappa = cohen_kappa_score(y_val, y_val_pred_rounded, weights='quadratic')

# Evaluate on test set
y_test_pred = best_model.predict(X_test_reshaped)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
y_test_pred_rounded = np.round(y_test_pred).astype(int)
test_kappa = cohen_kappa_score(y_test, y_test_pred_rounded, weights='quadratic')

# Print evaluation metrics
print("Training set evaluation:")
print(f'Mean Squared Error (MSE): {train_mse}')
print(f'Root Mean Squared Error (RMSE): {train_rmse}')
print(f'Mean Absolute Error (MAE): {train_mae}')
print(f'Cohen\'s Kappa Score: {train_kappa}')

print("\nValidation set evaluation:")
print(f'Mean Squared Error (MSE): {val_mse}')
print(f'Root Mean Squared Error (RMSE): {val_rmse}')
print(f'Mean Absolute Error (MAE): {val_mae}')
print(f'Cohen\'s Kappa Score: {val_kappa}')

print("\nTest set evaluation:")
print(f'Mean Squared Error (MSE): {test_mse}')
print(f'Root Mean Squared Error (RMSE): {test_rmse}')
print(f'Mean Absolute Error (MAE): {test_mae}')
print(f'Cohen\'s Kappa Score: {test_kappa}')

"""Training vs. Validation Metrics:

The training and validation metrics are relatively close, suggesting that the model is not significantly overfitting. The Cohen's Kappa Scores for both the training and validation sets are high and close to each other, indicating good model performance and generalization.
Validation vs. Test Metrics:

The validation and test metrics are also close to each other, which further suggests that the model is not overfitting and is generalizing well to unseen data.
"""


# Metrics values
metrics = {
    'MSE': [train_mse, val_mse, test_mse],
    'RMSE': [train_rmse, val_rmse, test_rmse],
    'MAE': [train_mae, val_mae, test_mae]
}

# Labels for the sets
labels = ['Training', 'Validation', 'Test']

# Plotting MSE
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.bar(labels, metrics['MSE'], color=['blue', 'orange', 'green'])
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')

# Plotting RMSE
plt.subplot(1, 3, 2)
plt.bar(labels, metrics['RMSE'], color=['blue', 'orange', 'green'])
plt.title('Root Mean Squared Error (RMSE)')
plt.ylabel('RMSE')

# Plotting MAE
plt.subplot(1, 3, 3)
plt.bar(labels, metrics['MAE'], color=['blue', 'orange', 'green'])
plt.title('Mean Absolute Error (MAE)')
plt.ylabel('MAE')

plt.tight_layout()
plt.show()

"""Analysis of Bar Plots
Mean Squared Error (MSE):

Training MSE is lower than validation and test MSE.
Validation and test MSE are close to each other.
Root Mean Squared Error (RMSE):

Training RMSE is lower than validation and test RMSE.
Validation and test RMSE are close to each other.
Mean Absolute Error (MAE):

Training MAE is slightly lower than validation and test MAE.
Validation and test MAE are close to each other.
Conclusions
Overfitting:

The training metrics (MSE, RMSE, MAE) are better than the validation and test metrics. This indicates that the model performs slightly better on the training data, which is expected. However, the differences are not very large.
The closeness of validation and test metrics suggests that the model is generalizing well to unseen data.
Generalization:

Since the validation and test metrics are close to each other, it indicates that the model is not overfitting significantly and is generalizing well.
"""


def get_bilstm_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(300, dropout=0.4, recurrent_dropout=0.4, return_sequences=True), input_shape=[1, 300]))
    model.add(Bidirectional(LSTM(64, recurrent_dropout=0.4)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy', 'mae'])
    model.summary()

    return model

# Reshape the feature vectors for Bidirectional LSTM
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val_reshaped = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('../best_bilstm_model.h5', save_best_only=True, monitor='val_loss')

# Train the model with early stopping and model checkpoint
bilstm_model = get_bilstm_model()
history = bilstm_model.fit(X_train_reshaped, y_train, epochs=400, batch_size=64, validation_data=(X_val_reshaped, y_val),
                         callbacks=[early_stopping, model_checkpoint])

# Display the training history
print(history.history)

# Load the best model
best_bilstm_model = tf.keras.models.load_model('../best_bilstm_model.h5')

# Evaluate on training, validation, and test sets
y_train_pred = best_bilstm_model.predict(X_train_reshaped)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_kappa = cohen_kappa_score(y_train, np.round(y_train_pred).astype(int), weights='quadratic')

y_val_pred = best_bilstm_model.predict(X_val_reshaped)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_kappa = cohen_kappa_score(y_val, np.round(y_val_pred).astype(int), weights='quadratic')

y_test_pred = best_bilstm_model.predict(X_test_reshaped)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_kappa = cohen_kappa_score(y_test, np.round(y_test_pred).astype(int), weights='quadratic')

# Print evaluation metrics
print("Training set evaluation:")
print(f'Mean Squared Error (MSE): {train_mse}')
print(f'Root Mean Squared Error (RMSE): {train_rmse}')
print(f'Mean Absolute Error (MAE): {train_mae}')
print(f'Cohen\'s Kappa Score: {train_kappa}')

print("\nValidation set evaluation:")
print(f'Mean Squared Error (MSE): {val_mse}')
print(f'Root Mean Squared Error (RMSE): {val_rmse}')
print(f'Mean Absolute Error (MAE): {val_mae}')
print(f'Cohen\'s Kappa Score: {val_kappa}')

print("\nTest set evaluation:")
print(f'Mean Squared Error (MSE): {test_mse}')
print(f'Root Mean Squared Error (RMSE): {test_rmse}')
print(f'Mean Absolute Error (MAE): {test_mae}')
print(f'Cohen\'s Kappa Score: {test_kappa}')

"""Analysis
Discrepancy in Metrics:

The training and validation/test metrics are relatively close, which suggests the model might not be overfitting significantly.
Cohen's Kappa Score is high for all sets, which indicates good agreement.
Learning Curves:

From the provided screenshots and learning curves, the training and validation losses converge, and there are no signs of a significant gap.
The accuracy plot shows stable trends for both training and validation sets, without diverging significantly.
"""

# Import necessary libraries

# Define the model-building function for Keras Tuner
def build_bilstm_model(hp):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                                 dropout=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1),
                                 recurrent_dropout=hp.Float('recurrent_dropout', min_value=0.1, max_value=0.5, step=0.1),
                                 return_sequences=True), input_shape=[1, 300]))
    model.add(Bidirectional(LSTM(units=hp.Int('units2', min_value=32, max_value=512, step=32),
                                 recurrent_dropout=hp.Float('recurrent_dropout2', min_value=0.1, max_value=0.5, step=0.1))))
    model.add(Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error',
                  optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd']),
                  metrics=['accuracy', 'mae'])
    return model

# Initialize the tuner
tuner = kt.RandomSearch(
    build_bilstm_model,
    objective='val_loss',
    max_trials=10,  # Number of hyperparameter combinations to try
    executions_per_trial=2,  # Number of models to train for each combination of hyperparameters
    directory='bilstm_tuning',
    project_name='essay_scoring'
)

# Define early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('../best_bilstm_model_tuned.h5', save_best_only=True, monitor='val_loss')

# Start the hyperparameter search
tuner.search(X_train_reshaped, y_train, epochs=100, validation_data=(X_val_reshaped, y_val),
             callbacks=[early_stopping, model_checkpoint])

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Evaluate the best model on the training, validation, and test sets
y_train_pred = best_model.predict(X_train_reshaped)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_kappa = cohen_kappa_score(y_train, np.round(y_train_pred).astype(int), weights='quadratic')

y_val_pred = best_model.predict(X_val_reshaped)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_kappa = cohen_kappa_score(y_val, np.round(y_val_pred).astype(int), weights='quadratic')

y_test_pred = best_model.predict(X_test_reshaped)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_kappa = cohen_kappa_score(y_test, np.round(y_test_pred).astype(int), weights='quadratic')

# Print evaluation metrics
print("Training set evaluation:")
print(f'Mean Squared Error (MSE): {train_mse}')
print(f'Root Mean Squared Error (RMSE): {train_rmse}')
print(f'Mean Absolute Error (MAE): {train_mae}')
print(f'Cohen\'s Kappa Score: {train_kappa}')

print("\nValidation set evaluation:")
print(f'Mean Squared Error (MSE): {val_mse}')
print(f'Root Mean Squared Error (RMSE): {val_rmse}')
print(f'Mean Absolute Error (MAE): {val_mae}')
print(f'Cohen\'s Kappa Score: {val_kappa}')

print("\nTest set evaluation:")
print(f'Mean Squared Error (MSE): {test_mse}')
print(f'Root Mean Squared Error (RMSE): {test_rmse}')
print(f'Mean Absolute Error (MAE): {test_mae}')
print(f'Cohen\'s Kappa Score: {test_kappa}')

"""Analysis of the Learning Curve:
Training and Validation Loss Convergence:

Both the training and validation loss curves are decreasing steadily.
The gap between the training and validation loss is small and remains consistent, which indicates that the model is generalizing well.
No Significant Overfitting:

There is no sign of the validation loss increasing while the training loss continues to decrease, which would indicate overfitting.
The validation loss closely follows the training loss, which is a good indication of the model's ability to generalize.
Stabilization of Loss:

Both curves start to stabilize towards the end of the training, suggesting that the model has learned the underlying patterns in the data.
Conclusion:
Based on the learning curve and the evaluation metrics:

Expected Higher Validation/Test Error: It is expected because the model is not directly optimized on the validation/test data. The validation and test data are used to evaluate the model's performance on unseen data, which naturally results in slightly higher errors.
Good Generalization: The small and consistent gap between training and validation losses indicates that the model is not significantly overfitting and is generalizing well.
"""

# Plot evaluation metrics
metrics = {
    'MSE': [train_mse, val_mse, test_mse],
    'RMSE': [train_rmse, val_rmse, test_rmse],
    'MAE': [train_mae, val_mae, test_mae]
}
labels = ['Training', 'Validation', 'Test']

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.bar(labels, metrics['MSE'], color=['blue', 'orange', 'green'])
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')

plt.subplot(1, 3, 2)
plt.bar(labels, metrics['RMSE'], color=['blue', 'orange', 'green'])
plt.title('Root Mean Squared Error (RMSE)')
plt.ylabel('RMSE')

plt.subplot(1, 3, 3)
plt.bar(labels, metrics['MAE'], color=['blue', 'orange', 'green'])
plt.title('Mean Absolute Error (MAE)')
plt.ylabel('MAE')

plt.tight_layout()
plt.show()

"""Conclusion
A slightly higher error on validation and test sets compared to the training set is a sign of a well-generalizing model. It means the model has learned the underlying patterns without overfitting to the training data. The high Cohen's Kappa Score across all sets further confirms the model's strong agreement between predicted and actual scores.
"""


