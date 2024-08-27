from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import pandas as pd
import gensim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns


class LstmWithWord2Vec:
    def __init__(self):
        self.epochs = 150
        self.file_path = 'data.txt'
        self.max_len = 300
        self.input_dim = 500
        self.embedding_dim = 50
        self.classes = None
        self.batch_size = 64
        self.word2vec_path = 'glove.6B.50d.txt'
        self.target_size = 100  

    def load_word2vec_model(self):
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_path, binary=False, no_header=True)

    def get_embedding_matrix(self, word_index):
        embedding_matrix = np.zeros((self.input_dim, self.embedding_dim))
        for word, i in word_index.items():
            if i >= self.input_dim:
                continue
            try:
                embedding_vector = self.word2vec[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                pass
        return embedding_matrix

    def preprocess_data(self):
        df = pd.read_csv(self.file_path, delimiter=",")
        df = df.dropna()
        # print(df)
        class_counts = df['Category'].value_counts()
        classes_to_increase = class_counts[class_counts < self.target_size].index
        df_increased = pd.DataFrame()

        for cls in classes_to_increase:
            class_df = df[df['Category'] == cls]
            num_to_sample = self.target_size - len(class_df)
            sampled_df = class_df.sample(num_to_sample, replace=True)
            df_increased = pd.concat([df_increased, class_df, sampled_df])

        df_remaining = df[~df['Category'].isin(classes_to_increase)]
        df_balanced = pd.concat([df_increased, df_remaining])

        label_encoder = LabelEncoder()
        df_balanced['Category'] = label_encoder.fit_transform(df_balanced['Category'])
        self.classes = label_encoder.classes_
        labels = tf.keras.utils.to_categorical(df_balanced['Category'])
        labels_df = pd.DataFrame(labels)
        # print(labels_df)
        

        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.input_dim, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(df_balanced['Description'].values)
        X = tokenizer.texts_to_sequences(df_balanced['Description'].values)
        X =tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.max_len)
        Y = pd.get_dummies(df_balanced['Category'],columns=df_balanced["Category"]).values
        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
        self.embedding_matrix = self.get_embedding_matrix(tokenizer.word_index)

        return X_train, y_train, X_test, y_test


    def get_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.embedding_dim, input_length=self.max_len,
                                            weights=[self.embedding_matrix], trainable=False))
        model.add(tf.keras.layers.SpatialDropout1D(0.3))
        model.add(tf.keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3))
        model.add(tf.keras.layers.Dense(len(self.classes), activation='softmax'))
        
        print(model.summary())
        return model

    def classificate(self):
        self.load_word2vec_model()
        x_train, y_train, x_test, y_test = self.preprocess_data()
        model = self.get_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(x_test, y_test))

        print("Train Score :", model.evaluate(x_train, y_train))
        print("Test Score :", model.evaluate(x_test, y_test))

        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred_classes)

        self.plotGraphics(history)
        self.plotConfusion(cm, self.classes)

    def plotGraphics(self, history):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('loss-accuracy graph.png')
        plt.show()
       
        

    def plotConfusion(self, cm, classes):
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('confussion matrix.png')
        plt.show()
        

obj = LstmWithWord2Vec()
obj.classificate()
