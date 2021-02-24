from keras import Input, Model
from keras.datasets import imdb, reuters
from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import LSTM, Bidirectional
# from keras.layers import Attention, AdditiveAttention, Concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import backend as K
from transformers import TFBertForSequenceClassification, BertTokenizer
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
from transformers import glue_convert_examples_to_features
import numpy as np

from basics import *


class NeuralNetwork:
    def __init__(self, dataset):
        self.DATASET = dataset
        self.MODEL = Sequential()
        self.EMBEDDING = None

    def predict(self, sequence):
        if self.MODEL.name.__contains__("bert"):
            prediction = self.MODEL.predict(sequence)[0]
            label = np.argmax(prediction)
            confidence = np.max(prediction)
            return label, confidence

        sequence = np.expand_dims(sequence, axis=0)
        confidence = self.MODEL.predict(sequence)[0, 0]
        label = self.MODEL.predict_classes(sequence)[0, 0]
        # label = (confidence > 0.5).astype("int32")[0, 0]
        return label, confidence

    def train_network(self, model_type="cnn"):
        if self.DATASET == "imdb":
            # From Keras example - text classification.
            # set parameters:
            vocab_size = 10000
            max_length = 400
            embedding_dims = 50

            batch_size = 32
            epochs = 2

            print("Loading data...")
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
            print(len(x_train), "train sequences")
            print(len(x_test), "test sequences")

            print("Pad sequences (samples x length)")
            x_train = pad_sequences(x_train, maxlen=max_length)
            x_test = pad_sequences(x_test, maxlen=max_length)
            print("x_train shape:", x_train.shape)
            print("x_test shape:", x_test.shape)

            # model_type = "cnn"
            # model_type = "lstm"
            # model_type = "self-attention_cnn"
            # model_type = "self-attention_lstm"

            if model_type == "cnn":
                filters = 250
                kernel_size = 3
                hidding_dims = 250

                # https://keras.io/zh/examples/imdb_cnn/

                print("Build model...")
                model = Sequential(name="cnn")

                # we start off with an efficient embedding layer which
                # maps our vocab indices into embedding_dims dimensions
                model.add(Embedding(vocab_size,
                                    embedding_dims,
                                    input_length=max_length))
                model.add(Dropout(0.2))

                # we add a Convolution1D, which will learn filters
                # word group filters of size filter_length:
                model.add(Conv1D(filters,
                                 kernel_size,
                                 padding="valid",
                                 activation="relu",
                                 strides=1))
                # we use max pooling:
                model.add(GlobalMaxPooling1D())

                # we add a vanilla hidden layer:
                model.add(Dense(hidding_dims))
                model.add(Dropout(0.2))
                model.add(Activation("relu"))

                # we project onto a single unit output layer, and squash it with a sigmoid:
                model.add(Dense(1))
                model.add(Activation("sigmoid"))
                model.summary()

                model.compile(loss="binary_crossentropy",
                              optimizer="adam",
                              metrics=["accuracy"])
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test))

                score = model.evaluate(x_test, y_test, verbose=0)
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])

                self.MODEL = model

            elif model_type == "self-attention_cnn":
                filters = 250
                kernel_size = 3
                hidding_dims = 250

                # keras_self_attention package
                # https://pypi.org/project/keras-self-attention/#description

                print("Build model...")
                model = Sequential(name="self-attention_cnn")

                # we start off with an efficient embedding layer which
                # maps our vocab indices into embedding_dims dimensions
                model.add(Embedding(vocab_size,
                                    embedding_dims,
                                    input_length=max_length))
                model.add(Dropout(0.2))

                # we add a Convolution1D, which will learn filters
                # word group filters of size filter_length:
                model.add(Conv1D(filters,
                                 kernel_size,
                                 padding="valid",
                                 activation="relu",
                                 strides=1))

                model.add(SeqSelfAttention(attention_activation="sigmoid"))
                # model.add(SeqWeightedAttention())

                # we use max pooling:
                model.add(GlobalMaxPooling1D())

                # we add a vanilla hidden layer:
                model.add(Dense(hidding_dims))
                model.add(Dropout(0.2))
                model.add(Activation("relu"))

                # we project onto a single unit output layer, and squash it with a sigmoid:
                model.add(Dense(1))
                model.add(Activation("sigmoid"))
                model.summary()

                model.compile(loss="binary_crossentropy",
                              optimizer="adam",
                              metrics=["accuracy"])
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test))

                score = model.evaluate(x_test, y_test, verbose=0)
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])

                self.MODEL = model

            elif model_type == "lstm":
                lstm_units = 128

                # https://keras.io/examples/nlp/bidirectional_lstm_imdb/

                model = Sequential(name="lstm")
                model.add(Embedding(vocab_size,
                                    embedding_dims,
                                    input_length=max_length))
                model.add(Bidirectional(LSTM(units=lstm_units,
                                             return_sequences=True)))
                # model.add(Bidirectional(LSTM(units=lstm_units,
                #                              return_sequences=True)))
                model.add(Bidirectional(LSTM(units=lstm_units)))
                model.add(Dense(1, activation="sigmoid"))
                model.summary()

                model.compile(loss="binary_crossentropy",
                              optimizer="adam",
                              metrics=["accuracy"])
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test))

                score = model.evaluate(x_test, y_test, verbose=0)
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])

                self.MODEL = model

            elif model_type == "self-attention_lstm":
                lstm_units = 128

                # keras_self_attention package
                # https://github.com/joek47/keras-self-attention/blob/master/Self%20attention.ipynb

                model = Sequential(name="self-attention_lstm")
                model.add(Embedding(vocab_size,
                                    embedding_dims,
                                    input_length=max_length))
                model.add(Bidirectional(LSTM(units=lstm_units,
                                             return_sequences=True)))
                model.add(Bidirectional(LSTM(units=lstm_units,
                                             return_sequences=True)))
                # model.add(Bidirectional(LSTM(units=lstm_units)))
                # model.add(SeqSelfAttention(attention_activation="sigmoid"))
                model.add(SeqWeightedAttention())
                model.add(Dense(1, activation="sigmoid"))
                model.summary()

                model.compile(loss="binary_crossentropy",
                              optimizer="adam",
                              metrics=["accuracy"])
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test))

                score = model.evaluate(x_test, y_test, verbose=0)
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])

                self.MODEL = model

        elif self.DATASET == "reuters":
            # From Keras example - text classification.
            # set parameters:
            vocab_size = 10000
            max_length = 250
            embedding_dims = 50
            num_classes = 46

            batch_size = 32
            epochs = 5

            print("Loading data...")
            (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=vocab_size)
            print(len(x_train), "train sequences")
            print(len(x_test), "test sequences")

            print("Pad sequences (samples x length)")
            x_train = pad_sequences(x_train, maxlen=max_length)
            x_test = pad_sequences(x_test, maxlen=max_length)
            print("x_train shape:", x_train.shape)
            print("x_test shape:", x_test.shape)
            from keras.utils import to_categorical
            y_train = to_categorical(y_train, num_classes)
            y_test = to_categorical(y_test, num_classes)

            if model_type == "cnn":
                filters = 250
                kernel_size = 3
                hidding_dims = 250

                # https://keras.io/zh/examples/imdb_cnn/

                print("Build model...")
                model = Sequential(name="cnn")

                # we start off with an efficient embedding layer which
                # maps our vocab indices into embedding_dims dimensions
                model.add(Embedding(vocab_size,
                                    embedding_dims,
                                    input_length=max_length))
                model.add(Dropout(0.2))

                # we add a Convolution1D, which will learn filters
                # word group filters of size filter_length:
                model.add(Conv1D(filters,
                                 kernel_size,
                                 padding="valid",
                                 activation="relu",
                                 strides=1))
                # we use max pooling:
                model.add(GlobalMaxPooling1D())

                # we add a vanilla hidden layer:
                model.add(Dense(hidding_dims))
                model.add(Dropout(0.2))
                model.add(Activation("relu"))

                # we project onto a single unit output layer, and squash it with a sigmoid:
                model.add(Dense(num_classes))
                model.add(Activation("softmax"))
                model.summary()

                model.compile(loss="categorical_crossentropy",
                              optimizer="adam",
                              metrics=["accuracy"])
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test))

                score = model.evaluate(x_test, y_test, verbose=0)
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])

                self.MODEL = model

            elif model_type == "self-attention_cnn":
                filters = 250
                kernel_size = 3
                hidding_dims = 250

                # keras_self_attention package
                # https://pypi.org/project/keras-self-attention/#description

                print("Build model...")
                model = Sequential(name="self-attention_cnn")

                # we start off with an efficient embedding layer which
                # maps our vocab indices into embedding_dims dimensions
                model.add(Embedding(vocab_size,
                                    embedding_dims,
                                    input_length=max_length))
                model.add(Dropout(0.2))

                # we add a Convolution1D, which will learn filters
                # word group filters of size filter_length:
                model.add(Conv1D(filters,
                                 kernel_size,
                                 padding="valid",
                                 activation="relu",
                                 strides=1))

                model.add(SeqSelfAttention(attention_activation="sigmoid"))
                # model.add(SeqWeightedAttention())

                # we use max pooling:
                model.add(GlobalMaxPooling1D())

                # we add a vanilla hidden layer:
                model.add(Dense(hidding_dims))
                model.add(Dropout(0.2))
                model.add(Activation("relu"))

                # we project onto a single unit output layer, and squash it with a sigmoid:
                model.add(Dense(num_classes))
                model.add(Activation("softmax"))
                model.summary()

                model.compile(loss="categorical_crossentropy",
                              optimizer="adam",
                              metrics=["accuracy"])
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test))

                score = model.evaluate(x_test, y_test, verbose=0)
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])

                self.MODEL = model

            elif model_type == "lstm":
                lstm_units = 256

                # https://keras.io/examples/nlp/bidirectional_lstm_imdb/

                model = Sequential(name="lstm")
                model.add(Embedding(vocab_size,
                                    embedding_dims,
                                    input_length=max_length))
                model.add(Bidirectional(LSTM(units=lstm_units,
                                             return_sequences=True)))
                # model.add(Bidirectional(LSTM(units=lstm_units,
                #                              return_sequences=True)))
                model.add(Bidirectional(LSTM(units=lstm_units)))

                model.add(Dense(num_classes))
                model.add(Activation("softmax"))
                model.summary()

                model.compile(loss="categorical_crossentropy",
                              optimizer="adam",
                              metrics=["accuracy"])
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test))

                score = model.evaluate(x_test, y_test, verbose=0)
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])

                self.MODEL = model

            elif model_type == "self-attention_lstm":
                lstm_units = 256

                # keras_self_attention package
                # https://github.com/joek47/keras-self-attention/blob/master/Self%20attention.ipynb

                model = Sequential(name="self-attention_lstm")
                model.add(Embedding(vocab_size,
                                    embedding_dims,
                                    input_length=max_length))
                model.add(Bidirectional(LSTM(units=lstm_units,
                                             return_sequences=True)))
                model.add(Bidirectional(LSTM(units=lstm_units,
                                             return_sequences=True)))
                # model.add(Bidirectional(LSTM(units=lstm_units)))
                # model.add(SeqSelfAttention(attention_activation="sigmoid"))
                model.add(SeqWeightedAttention())

                model.add(Dense(num_classes))
                model.add(Activation("softmax"))
                model.summary()

                model.compile(loss="categorical_crossentropy",
                              optimizer="adam",
                              metrics=["accuracy"])
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test))

                score = model.evaluate(x_test, y_test, verbose=0)
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])

                self.MODEL = model

        elif self.DATASET == "sst":
            # From Tensorflow example - sentiment analysis.
            # set parameters:
            vocab_size = 10000
            max_length = 49
            embedding_dims = 50
            trunc_type = 'post'
            pad_type = 'post'
            oov_tok = "<OOV>"

            batch_size = 32
            epochs = 2

            # https://colab.research.google.com/github/tensorflow/examples/blob/master/
            # courses/udacity_intro_to_tensorflow_for_deep_learning/
            # l10c02_nlp_multiple_models_for_predicting_sentiment.ipynb#scrollTo=0TWLvXA1Oa_W
            print("Loading data...")
            sst, _ = tfds.load("glue/sst2", data_dir='./datasets/SST/', with_info=True)
            train, validation = sst["train"], sst["validation"]
            x_train, y_train, x_validation, y_validation = [], [], [], []
            for item in train.take(-1):
                x, y = item["sentence"], item["label"]
                x_train.append(x.numpy().decode())
                y_train.append(y.numpy())
            for item in validation.take(-1):
                x, y = item["sentence"], item["label"]
                x_validation.append(x.numpy().decode())
                y_validation.append(y.numpy())
            print(len(x_train), "train sequences")
            print(len(x_validation), "validation sequences")

            tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
            tokenizer.fit_on_texts(x_train)
            word_index = tokenizer.word_index
            print("Pad sequences (samples x length)")
            x_train = tokenizer.texts_to_sequences(x_train)
            x_train = pad_sequences(sequences=x_train,
                                    maxlen=max_length,
                                    truncating=trunc_type,
                                    padding=pad_type)
            x_validation = tokenizer.texts_to_sequences(x_validation)
            x_validation = pad_sequences(sequences=x_validation,
                                         maxlen=max_length,
                                         truncating=trunc_type,
                                         padding=pad_type)
            y_train = np.array(y_train)
            y_validation = np.array(y_validation)
            print("x_train shape:", x_train.shape)
            print("x_validation shape:", x_validation.shape)

            if model_type == "cnn":
                # filters = 16
                # kernel_size = 5
                # hidding_dims = 25

                filters = 250
                kernel_size = 3
                hidding_dims = 250

                # https://keras.io/zh/examples/imdb_cnn/

                print("Build model...")
                model = Sequential(name="cnn")

                # we start off with an efficient embedding layer which
                # maps our vocab indices into embedding_dims dimensions
                model.add(Embedding(vocab_size,
                                    embedding_dims,
                                    input_length=max_length))
                model.add(Dropout(0.2))

                # we add a Convolution1D, which will learn filters
                # word group filters of size filter_length:
                model.add(Conv1D(filters,
                                 kernel_size,
                                 padding="valid",
                                 activation="relu",
                                 strides=1))
                # we use max pooling:
                model.add(GlobalMaxPooling1D())

                # we add a vanilla hidden layer:
                model.add(Dense(hidding_dims))
                model.add(Dropout(0.2))
                model.add(Activation("relu"))

                # we project onto a single unit output layer, and squash it with a sigmoid:
                model.add(Dense(1))
                model.add(Activation("sigmoid"))
                model.summary()

                # learning_rate = 0.0001
                model.compile(loss="binary_crossentropy",
                              optimizer="adam",
                              # optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
                              metrics=["accuracy"])
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_validation, y_validation))

                score = model.evaluate(x_validation, y_validation, verbose=0)
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])

                self.MODEL = model

            elif model_type == "self-attention_cnn":
                # filters = 16
                # kernel_size = 5
                # hidding_dims = 25

                filters = 250
                kernel_size = 3
                hidding_dims = 250

                # keras_self_attention package
                # https://pypi.org/project/keras-self-attention/#description

                print("Build model...")
                model = Sequential(name="self-attention_cnn")

                # we start off with an efficient embedding layer which
                # maps our vocab indices into embedding_dims dimensions
                model.add(Embedding(vocab_size,
                                    embedding_dims,
                                    input_length=max_length))
                model.add(Dropout(0.2))

                # we add a Convolution1D, which will learn filters
                # word group filters of size filter_length:
                model.add(Conv1D(filters,
                                 kernel_size,
                                 padding="valid",
                                 activation="relu",
                                 strides=1))

                model.add(SeqSelfAttention(attention_activation="sigmoid"))
                # model.add(SeqWeightedAttention())

                # we use max pooling:
                model.add(GlobalMaxPooling1D())

                # we add a vanilla hidden layer:
                model.add(Dense(hidding_dims))
                model.add(Dropout(0.2))
                model.add(Activation("relu"))

                # we project onto a single unit output layer, and squash it with a sigmoid:
                model.add(Dense(1))
                model.add(Activation("sigmoid"))
                model.summary()

                # learning_rate = 0.0001
                model.compile(loss="binary_crossentropy",
                              optimizer="adam",
                              # optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
                              metrics=["accuracy"])
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_validation, y_validation))

                score = model.evaluate(x_validation, y_validation, verbose=0)
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])

                self.MODEL = model

            elif model_type == "lstm":
                lstm_units = 128

                # https://keras.io/examples/nlp/bidirectional_lstm_imdb/

                model = Sequential(name="lstm")
                model.add(Embedding(vocab_size,
                                    embedding_dims,
                                    input_length=max_length))
                model.add(Bidirectional(LSTM(units=lstm_units,
                                             return_sequences=True)))
                # model.add(Bidirectional(LSTM(units=lstm_units,
                #                              return_sequences=True)))
                model.add(Bidirectional(LSTM(units=lstm_units)))
                model.add(Dense(1, activation="sigmoid"))
                model.summary()

                # learning_rate = 0.0003
                model.compile(loss="binary_crossentropy",
                              optimizer="adam",
                              # optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
                              metrics=["accuracy"])
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_validation, y_validation))

                score = model.evaluate(x_validation, y_validation, verbose=0)
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])

                self.MODEL = model

            elif model_type == "self-attention_lstm":
                lstm_units = 128

                # keras_self_attention package
                # https://github.com/joek47/keras-self-attention/blob/master/Self%20attention.ipynb

                model = Sequential(name="self-attention_lstm")
                model.add(Embedding(vocab_size,
                                    embedding_dims,
                                    input_length=max_length))
                model.add(Bidirectional(LSTM(units=lstm_units,
                                             return_sequences=True)))
                model.add(Bidirectional(LSTM(units=lstm_units,
                                             return_sequences=True)))
                # model.add(Bidirectional(LSTM(units=lstm_units)))
                # model.add(SeqSelfAttention(attention_activation="sigmoid"))
                model.add(SeqWeightedAttention())
                model.add(Dense(1, activation="sigmoid"))
                model.summary()

                # learning_rate = 0.0003
                model.compile(loss="binary_crossentropy",
                              optimizer="adam",
                              # optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
                              metrics=["accuracy"])
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_validation, y_validation))

                score = model.evaluate(x_validation, y_validation, verbose=0)
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])

                self.MODEL = model

        elif self.DATASET == "sst-bert":
            model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

            # model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
            # tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

            data, _ = tfds.load("glue/sst2", data_dir='./datasets/SST/', with_info=True)
            # data = tfds.load("glue/mrpc", data_dir="./datasets/MRPC/")
            train_dataset = data["train"]
            validation_dataset = data["validation"]
            train_dataset = glue_convert_examples_to_features(examples=train_dataset,
                                                              tokenizer=tokenizer,
                                                              max_length=49,
                                                              task='sst-2'
                                                              )
            validation_dataset = glue_convert_examples_to_features(examples=validation_dataset,
                                                                   tokenizer=tokenizer,
                                                                   max_length=49,
                                                                   task='sst-2')
            train_dataset = train_dataset.batch(640)
            validation_dataset = validation_dataset.batch(64)

            model.summary()
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

            model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
            model.fit(train_dataset, epochs=3, validation_data=validation_dataset)

            self.MODEL = model

        elif self.DATASET == "mrpc":
            model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

            # model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
            # tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

            data = tfds.load("glue/mrpc", data_dir="./datasets/MRPC/")
            train_dataset = data["train"]
            validation_dataset = data["validation"]
            train_dataset = glue_convert_examples_to_features(examples=train_dataset,
                                                              tokenizer=tokenizer,
                                                              max_length=128,
                                                              task='mrpc')
            validation_dataset = glue_convert_examples_to_features(examples=validation_dataset,
                                                                   tokenizer=tokenizer,
                                                                   max_length=128,
                                                                   task='mrpc')
            train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
            validation_dataset = validation_dataset.batch(64)

            # for item in validation_dataset.take(1):
            #     x, y = item[0], item[1]
            #     print(x.get('input_ids'))
            #     print(tokenizer.convert_ids_to_tokens(x.get('input_ids')))
            #     print(y.numpy())

            model.summary()
            optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

            model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
            model.fit(train_dataset, epochs=3, validation_data=validation_dataset)

            self.MODEL = model

        else:
            print("Unsupported dataset %s." % self.DATASET)
            exit()

        self.save_network()
        self.EMBEDDING = K.function([self.MODEL.layers[0].input],
                                    [self.MODEL.layers[0].output])

    def save_network(self):
        directory = "models/{}/".format(self.DATASET)
        # directory = "attention_cnn/"
        assure_path_exists(directory)

        if self.MODEL.name.__contains__('bert'):
            self.MODEL.save("{}{}({})".format(directory, self.DATASET, self.MODEL.name))
            print("Neural network saved to disk.")

        elif self.DATASET == "imdb" or self.DATASET == "sst" or self.DATASET == "reuters":
            # self.MODEL.save("models/imdb.h5")
            self.MODEL.save("{}{}({}).h5".format(directory, self.DATASET, self.MODEL.name))
            # self.MODEL.save(directory + self.DATASET + "-" + self.MODEL.name + ".h5")
            print("Neural network saved to disk.")

        # elif self.DATASET == "mrpc":
        #     self.MODEL.save("{}{}({})".format(directory, self.DATASET, self.MODEL.name))
        #     print("Neural network saved to disk.")

        else:
            print("Save network: unsupported dataset.")
            exit()

    def load_network(self, model_type="cnn"):
        directory = "models/{}/".format(self.DATASET)
        model_path = "{}{}({}).h5".format(directory, self.DATASET, model_type)
        # model_path = "{}Anthony/{}({})_90.h5".format(directory, self.DATASET, model_type)

        if self.DATASET == "imdb" or self.DATASET == "sst" or self.DATASET == "sst-bert" or self.DATASET == "mrpc" or self.DATASET == "reuters":
            if model_type == "bert":
                model_type = "tf_bert_for_sequence_classification"
                # model_type = "tf_distil_bert_for_sequence_classification"
                model_path = "{}{}({})".format(directory, self.DATASET, model_type)
                self.MODEL = load_model(model_path)
            elif model_type == "cnn" or model_type == "lstm":
                print(model_path)
                self.MODEL = load_model(model_path)
            elif model_type == "self-attention_cnn":
                self.MODEL = load_model(model_path,
                                        custom_objects=SeqSelfAttention.get_custom_objects())
            elif model_type == "self-attention_lstm":
                self.MODEL = load_model(model_path,
                                        custom_objects=SeqWeightedAttention.get_custom_objects())
            else:
                print("Unrecognised model type. "
                      "Try 'cnn', 'lstm', 'self-attention_cnn' or 'self-attention_lstm'.")
                exit()
            print("Neural network loaded from disk.")
        else:
            print("Load network: unsupported dataset.")
            exit()

        self.EMBEDDING = K.function([self.MODEL.layers[0].input],
                                    [self.MODEL.layers[0].output])

    def obtain_embedding(self, sequence):
        # from keras import backend as K
        # func = K.function([model.layers[0].input],
        #                   [model.layers[0].output])
        # func = K.function([model.get_layer("embedding").input],
        #                   [model.get_layer("embedding").output])
        embedding = self.EMBEDDING([sequence])[0]
        return embedding
