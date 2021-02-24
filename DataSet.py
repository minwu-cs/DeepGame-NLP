from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import tensorflow_datasets as tfds
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from transformers import TFBertForSequenceClassification, BertTokenizer
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
from transformers import glue_convert_examples_to_features

# from manipulations import tag_map
from basics import gensim_model, tag_map


class DataSet:
    def __init__(self, dataset, training=False):
        self.DATASET = dataset
        self.VOCAB_SIZE = 10000
        self.SIMILARITY_BOUND = 0.4

        if self.DATASET == "imdb":
            # vocab_size = 10000
            max_length = 400
            # embedding_dims = 50

            print("Loading data...")
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.VOCAB_SIZE)
            print(len(x_train), "train sequences")
            print(len(x_test), "test sequences")

            print("Pad sequences (samples x length)")
            x_train = pad_sequences(x_train, maxlen=max_length)
            x_test = pad_sequences(x_test, maxlen=max_length)
            print("x_train shape:", x_train.shape)
            print("x_test shape:", x_test.shape)

            if training:
                self.X = x_train
                self.Y = y_train
            else:
                self.X = x_test
                self.Y = y_test

            self.WORD_TO_ID = imdb.get_word_index()
            self.ID_TO_WORD = {value: key for key, value in self.WORD_TO_ID.items()}

        elif self.DATASET == "sst":
            # From Tensorflow example - sentiment analysis.
            # set parameters:
            # vocab_size = 10000
            max_length = 49
            # embedding_dims = 16
            trunc_type = 'post'
            pad_type = 'post'
            oov_tok = "<OOV>"

            # https://colab.research.google.com/github/tensorflow/examples/blob/
            # master/courses/udacity_intro_to_tensorflow_for_deep_learning/
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

            tokenizer = Tokenizer(num_words=self.VOCAB_SIZE, oov_token=oov_tok)
            tokenizer.fit_on_texts(x_train)
            self.ID_TO_WORD = tokenizer.index_word
            self.WORD_TO_ID = tokenizer.word_index

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

            if training:
                self.X = x_train
                self.Y = y_train
            else:
                self.X = x_validation
                self.Y = y_validation

        elif self.DATASET == "sst-bert":
            max_length = 49
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            # tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

            sst, _ = tfds.load("glue/sst2", data_dir='./datasets/SST/', with_info=True)
            train_dataset, validation_dataset = sst["train"], sst["validation"]
            train_dataset = glue_convert_examples_to_features(examples=train_dataset,
                                                              tokenizer=tokenizer,
                                                              max_length=max_length,
                                                              task='sst-2')

            validation_dataset = glue_convert_examples_to_features(examples=validation_dataset,
                                                                   tokenizer=tokenizer,
                                                                   max_length=max_length,
                                                                   task='sst-2')
            train_dataset = train_dataset.batch(1)
            validation_dataset = validation_dataset.batch(1)

            x_train, y_train, x_validation, y_validation = [], [], [], []
            for item in train_dataset.take(-1):
                x, y = item[0], item[1]
                x_train.append(x)
                y_train.append(y.numpy()[0])
            for item in validation_dataset.take(-1):
                x, y = item[0], item[1]
                x_validation.append(x)
                y_validation.append(y.numpy()[0])
            print(len(x_train), "train sequences")
            print(len(x_validation), "validation sequences")

            self.TOKENIZER = tokenizer
            if training:
                self.X = x_train
                self.Y = y_train
            else:
                self.X = x_validation
                self.Y = y_validation

        elif self.DATASET == "mrpc":
            max_length = 128
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            # tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

            mrpc = tfds.load("glue/mrpc", data_dir="./datasets/MRPC/")
            train_dataset, validation_dataset = mrpc["train"], mrpc["validation"]
            train_dataset = glue_convert_examples_to_features(examples=train_dataset,
                                                              tokenizer=tokenizer,
                                                              max_length=max_length,
                                                              task='mrpc')
            validation_dataset = glue_convert_examples_to_features(examples=validation_dataset,
                                                                   tokenizer=tokenizer,
                                                                   max_length=max_length,
                                                                   task='mrpc')
            train_dataset = train_dataset.batch(1)
            validation_dataset = validation_dataset.batch(1)

            x_train, y_train, x_validation, y_validation = [], [], [], []
            for item in train_dataset.take(-1):
                x, y = item[0], item[1]
                x_train.append(x)
                y_train.append(y.numpy()[0])
            for item in validation_dataset.take(-1):
                x, y = item[0], item[1]
                x_validation.append(x)
                y_validation.append(y.numpy()[0])
            print(len(x_train), "train sequences")
            print(len(x_validation), "validation sequences")

            self.TOKENIZER = tokenizer
            if training:
                self.X = x_train
                self.Y = y_train
            else:
                self.X = x_validation
                self.Y = y_validation
            return
        else:
            print("Unsupported dataset {}. Try 'imdb' or 'sst'.".format(dataset))
            # print("Unsupported dataset %s. Try 'imdb' or 'sst'." % dataset)
            exit()

    def get_dataset(self):
        return self.X, self.Y

    def get_input(self, index):
        return self.X[index]

    def get_oracle(self, index):
        return self.Y[index]

    def get_word(self, input_id):
        if self.DATASET == "imdb":
            return self.ID_TO_WORD.get(input_id - 3, "#")
        elif self.DATASET == "sst":
            return self.ID_TO_WORD.get(input_id, "#")
        elif self.DATASET == "sst-bert" or self.DATASET == "mrpc":
            return self.TOKENIZER.convert_ids_to_tokens(input_id)
        else:
            print("Unsupported dataset.")
            exit()

    def get_word_id(self, input_word):
        if self.DATASET == "imdb":
            word_id = self.WORD_TO_ID.get(input_word)
            if word_id:
                return word_id + 3
            else:
                return None
        elif self.DATASET == "sst":
            word_id = self.WORD_TO_ID.get(input_word)
            if word_id:
                return word_id
            else:
                return None
        elif self.DATASET == "sst-bert" or self.DATASET == "mrpc":
            return self.TOKENIZER.convert_tokens_to_ids(input_word)
        else:
            print("Unsupported dataset.")
            exit()

    def get_text(self, input_ids):
        if self.DATASET == "imdb":
            # https://builtin.com/data-science/how-build-neural-network-keras
            text = [self.ID_TO_WORD.get(i - 3, '#') for i in input_ids]

            with open("review.txt", "w") as f:
                for item in text:
                    f.write("%s\n" % item)
            exit()
            return text
        elif self.DATASET == "sst":
            text = [self.ID_TO_WORD.get(i, '#') for i in input_ids]
            return text
        elif self.DATASET == "sst-bert" or self.DATASET == "mrpc":
            # input_ids = input_ids['input_ids'][0]
            # print(self.TOKENIZER.decode(input_ids))
            # return self.TOKENIZER.convert_ids_to_tokens(input_ids)
            return self.TOKENIZER.decode(input_ids)
        else:
            print("Unsupported dataset.")
            exit()

    def print_text(self, input_ids, index=None):
        if self.DATASET == "sst-bert" or self.DATASET == "mrpc":
            # input_ids = input_ids['input_ids'][0]
            text = self.TOKENIZER.decode(input_ids)
            print(text)
            tokens = self.TOKENIZER.convert_ids_to_tokens(input_ids)
            print(tokens)

            file_name = "{}-index{}-oracle{}".format(self.DATASET,
                                                     index,
                                                     self.get_oracle(index))
            with open(file_name + ".txt", "w") as f:
                for item in tokens:
                    f.write("%s\n" % item)
            exit()
            return
        else:
            text = self.get_text(input_ids)

        line_num = int(np.sqrt(len(text)))
        for i in range(line_num):
            temp = " ".join(text[i * line_num: i * line_num + line_num])
            print(temp)

    def get_neighbourhood(self, input_ids):
        text = self.get_text(input_ids)
        text_pos = nltk.pos_tag(text)
        # text_pos = [(item[0], tag_map[item[1]]) for item in text_pos if tag_map[item[1]]]
        text_pos_dict = {}
        for i in range(len(text_pos)):
            if text_pos[i][0] not in stop_words and not text_pos[i][0].__contains__("'") and tag_map[text_pos[i][1]]:
                text_pos_dict.update({i: (text_pos[i][0], tag_map[text_pos[i][1]])})

        neighbourhood = {}
        for location, word_POS in text_pos_dict.items():
            synonyms = []
            antonyms = []
            for synset in wn.synsets(word_POS[0], pos=word_POS[1][0]):
                for lemma in synset.lemmas():
                    synonyms.append(lemma.name())
                    if lemma.antonyms():
                        for ant in lemma.antonyms():
                            antonyms.append(ant.name())
            synonyms = set(synonyms)
            antonyms = set(antonyms)
            combined = synonyms | antonyms
            combined.discard(word_POS[0])
            # print(combined)

            combined_ids = []
            for substitution in combined:
                word_id = self.get_word_id(substitution)
                if word_id and word_id < self.VOCAB_SIZE and substitution in gensim_model.vocab:
                    if gensim_model.similarity(word_POS[0], substitution) > self.SIMILARITY_BOUND:
                        combined_ids.append(word_id)
            if combined_ids:
                neighbourhood.update({location: combined_ids})

        print("Finding substitution(s) for each word... \n"
              "Cosine similarity bound", self.SIMILARITY_BOUND)
        for location, substitutions in neighbourhood.items():
            print("location:", [location, text[location]])
            print("Neighbours:", [[substitution, self.get_word(substitution)] for substitution in substitutions])
        # print(neighbourhood)
        return neighbourhood
