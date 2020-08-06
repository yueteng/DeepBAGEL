#coding:utf-8
from imblearn.keras import BalancedBatchGenerator
from imblearn.metrics import classification_report_imbalanced
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dropout, Dense, Concatenate, Embedding
from position_encoding import Position_Embedding
import numpy as np
import pyexcel as pe

class MultiCNN(object):
    def __init__(self, segment_length):

        self.kernel_sizes = [9, 18, 27, 36, 45, 54]
        self.filters_nums = [900, 900, 900, 900, 900, 900]     #    #
        self.dropout = 0.3                     #
        self.learning_rate = 0.001
        self.batch_size = 64
        self.max_epoches = 300
        self.channels_num = 4
        self.best_model_path = 'the_best_model'+'.h5'
        self.segment_length = segment_length    #

    def build_embedding_layers(self, input):
    
        embedding_matrix = np.array([[0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        embedding_layer = Embedding(self.channels_num+1, self.channels_num, weights=[embedding_matrix], input_length=self.segment_length, trainable=False)(input)
        return embedding_layer

    def encode_sequences(self, sequences):
 
        new_sequences = list()
        for sequence in sequences:
            new_sequence = list()
            for letter in sequence:
                if letter in {'A', 'a'}:
                    new_sequence.append(1)
                elif letter in {'C', 'c'}:
                    new_sequence.append(2)
                elif letter in {'G', 'g'}:
                    new_sequence.append(3)
                elif letter in {'T', 't'}:
                    new_sequence.append(4)
                else:
                    new_sequence.append(0)
            new_sequences.append(new_sequence)
        return np.array(new_sequences)

    def cnn(self, input, filters_num, kernel_size):
        conv_layer = Conv1D(filters=filters_num, kernel_size=kernel_size, activation='tanh')(input)
        pool_layer = GlobalMaxPooling1D()(conv_layer)
        return pool_layer

    def make_model(self, with_position=False):

        input = Input(shape=(self.segment_length, ), dtype='int32')
        embedding = self.build_embedding_layers(input)
        print(embedding[0])
        print('Embedding layer: ', embedding)
        if with_position:
            encoding = Position_Embedding()(embedding)
        else:
            encoding = embedding
        assert (len(self.kernel_sizes) == len(self.filters_nums))
        cnn_layers = []
        for filters_num, kernel_size in zip(self.filters_nums, self.kernel_sizes):
            cnn_layers.append(self.cnn(encoding, filters_num, kernel_size))
        cnn_layers = Concatenate(axis=1)(cnn_layers)

        output = Dropout(self.dropout)(cnn_layers)
        output = Dense(1, activation='sigmoid')(output)


        self.model = Model(inputs=input, outputs=output)
        self.model.compile(Adam(lr=self.learning_rate), 'binary_crossentropy', metrics=['accuracy'])


    def fit(self, train_data, train_labels, valid_data, valid_labels):

        train_data = self.encode_sequences(train_data)
        valid_data = self.encode_sequences(valid_data)

        early_stopping = EarlyStopping(monitor='val_acc', patience=60, verbose=1, mode='max')
        check_point = ModelCheckpoint(self.best_model_path, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
        callbacks_list = [check_point, early_stopping]
        self.model.fit(x=train_data, y=train_labels, batch_size=self.batch_size, epochs=self.max_epoches, verbose=2,
                       callbacks=callbacks_list, validation_data=(valid_data, valid_labels))

    def fit_with_weighted(self, train_data, train_labels, valid_data, valid_labels):

        train_data = self.encode_sequences(train_data)
        valid_data = self.encode_sequences(valid_data)

        weights = dict()
        for label in train_labels:
            weights.setdefault(label, 0)
            weights[label] += 1
        max_value = max(weights.items(), key=lambda x:x[1])[1]
        weights = {label: float(max_value)/weights[label] for label in weights}

        print(weights)
        early_stopping = EarlyStopping(monitor='val_acc', patience=30, verbose=1, mode='max')
        check_point = ModelCheckpoint(self.best_model_path, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

        callbacks_list = [check_point, early_stopping]

        self.model.fit(x=train_data, y=train_labels, batch_size=self.batch_size, epochs=self.max_epoches, verbose=2,
                       callbacks=callbacks_list, validation_data=(valid_data, valid_labels), class_weight= weights)

    def fit_with_undersampling(self, train_data, train_labels, valid_data, valid_labels):

        train_data = self.encode_sequences(train_data)
        valid_data = self.encode_sequences(valid_data)
        
        print('encode:',train_data.shape)
        
        training_generator = BalancedBatchGenerator(train_data, train_labels, batch_size=self.batch_size, random_state=42)

        early_stopping = EarlyStopping(monitor='val_acc', patience=50, verbose=1, mode='max')
        check_point = ModelCheckpoint(self.best_model_path, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
        callbacks_list = [check_point, early_stopping]


        self.model.fit_generator(generator=training_generator, epochs=self.max_epoches, verbose=2,
                       callbacks=callbacks_list, validation_data=(valid_data, valid_labels))


    def boolMap(self,arr):
        if arr > 0.5:
            return 1
        else:
            return 0

    def evaluate(self, test_data, test_labels):
 

        test_data = self.encode_sequences(test_data)
        print(np.shape(test_data))
        print(test_data[1])
        saved_model = load_model(self.best_model_path)
        _, test_acc = saved_model.evaluate(test_data, test_labels, verbose=0)
        print('Test: %.3f' % (test_acc))
        val_predict = list(map(self.boolMap, saved_model.predict(test_data)))
        target_names = ['0', '1']
        print(classification_report_imbalanced(test_labels, val_predict,  target_names=target_names))
        print('Test: %.3f' % (test_acc))

        TP,TN,FP,FN =0,0,0,0
        PP,NN = 0,0
        for i in range(len(val_predict)):
            if test_labels[i] == 1:
                PP +=1
                if val_predict[i] == 1:
                    TP += 1
                if val_predict[i] == 0:
                    FP += 1

        for i in range(len(val_predict)):
            if test_labels[i] == 0:
                NN +=1
                if val_predict[i] == 1:
                    FN += 1
                if val_predict[i] == 0:
                    TN += 1
        print('PP:{}'.format(PP))
        print('NN:{}'.format(NN))
        print('TP:{}'.format(TP/PP))
        print('TN:{}'.format(TN/NN))
        print('FP:{}'.format(FP/PP))
        print('FN:{}'.format(FN/NN))