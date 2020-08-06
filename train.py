import pickle
from model import MultiCNN

data,label = list(), list()
min_length = 204
max_length = 3468


print('Load training data ...')
path = 'after_data/'
with open(path + 'train.pkl', 'rb') as fp:
    data = pickle.load(fp)
    train_data = data['data']
    train_label = data['labels']

print('Load valid data ...')
with open(path + 'valid.pkl', 'rb') as fp:
    data = pickle.load(fp)
    valid_data = data['data']
    valid_label = data['labels']

with open(path + 'test.pkl', 'rb') as fp:
    data = pickle.load(fp)
    test_data = data['data']
    test_label = data['labels']


print('Make model ...')
segment_length = len(train_data[0])
multi_cnn_model = MultiCNN(segment_length)
multi_cnn_model.make_model(with_position=False)

print('Training ....')
multi_cnn_model. fit_with_weighted(train_data, train_label, valid_data, valid_label)
multi_cnn_model.evaluate(test_data,test_label)




