import tensorflow as tf
import numpy as np
import csv
import pickle

number_of_classes = 7
dimension = 48
number_of_channels = 1
batch_size = 50
number_of_epochs = 150

x = tf.placeholder('float', [None, dimension*dimension])
y = tf.placeholder('float', [None, number_of_classes])


def load_dataset(file):
    dataset_features = []
    dataset_labels = []

    file = '/input' + file

    with open(file) as csvfile:
        csv_reader_object = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in csv_reader_object:
            if len(row) == 0:
                _0 = 0  # ignore
            else:
                dataset_features.append(row[1].split())
                # print(count)
                # count += 1
                temp = np.zeros(number_of_classes, dtype=int)
                temp[row[0]] = int(1)
                dataset_labels.append(temp)

    return np.array(dataset_features), np.array(dataset_labels)

def pickle_dump(what, name):
    pickle_out = open(name, 'wb')
    pickle.dump(what, pickle_out)
    pickle_out.close()

def pickle_retrieve(name):
    pickle_in = open(name, 'rb')
    file = pickle.load(pickle_in)
    return file

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape))

def new_biases(length):
    return tf.Variable(tf.truncated_normal(shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True, use_padding=True):
    if use_padding:
        input = tf.pad(input, [[0, 0], [2, 2], [1, 1], [0, 0]], "CONSTANT")

    shape = [filter_size, filter_size, num_input_channels, num_filters]         # shape of the filter-weights for the convolution
    weights = new_weights(shape=shape)          # create new weights i.e. filters of the given shape
    biases = new_biases(length=num_filters)     # create new biases, one for each filter

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')    # padding='SAME' means input image padded with 0 if needed
    layer += biases         # Add the biases to the results of the convolution. A bias-value is added to each filter-channel

    if(use_pooling):        # downsample to original resolution divided by 2 since we are using 2x2 max pooling
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    layer = tf.nn.relu(layer)   # relu
    return layer, weights

def flatten_layer(layer):                               # flatten tensor of 4 dimension to 2 dimension so that they can be used for fully connected layer
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be: layer_shape == [num_images, img_height, img_width, num_channels]
    # The number of features is: img_height * img_width * num_channels, we can use a function from TensorFlow to calculate this
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])  # Reshape the layer to [num_images, num_features].

    return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def neural_net_model(x):
    # modified version of VGG-D with batch normalization and droupout layers

    dropout = 0.5

    x = tf.reshape(x, [-1, dimension, dimension, number_of_channels])

    layer_conv1, weights_conv1 = new_conv_layer(input=x, num_input_channels=number_of_channels, filter_size=3, num_filters=64, use_pooling=False, use_padding=True)
    layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=64, filter_size=3, num_filters=64, use_pooling=True, use_padding=True)
    layer_conv2 = tf.contrib.layers.batch_norm(layer_conv2)
    layer_conv2 = tf.nn.dropout(layer_conv2, dropout)

    layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2, num_input_channels=64, filter_size=3, num_filters=128, use_pooling=False, use_padding=True)
    layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3, num_input_channels=128, filter_size=3, num_filters=128, use_pooling=True, use_padding=True)
    layer_conv4 = tf.contrib.layers.batch_norm(layer_conv4)
    layer_conv4 = tf.nn.dropout(layer_conv4, dropout)

    layer_conv5, weights_conv5 = new_conv_layer(input=layer_conv4, num_input_channels=128, filter_size=3, num_filters=256, use_pooling=False, use_padding=True)
    layer_conv6, weights_conv6 = new_conv_layer(input=layer_conv5, num_input_channels=256, filter_size=3, num_filters=256, use_pooling=False, use_padding=True)
    layer_conv7, weights_conv7 = new_conv_layer(input=layer_conv6, num_input_channels=256, filter_size=3, num_filters=256, use_pooling=True, use_padding=True)
    layer_conv7 = tf.contrib.layers.batch_norm(layer_conv7)
    layer_conv7 = tf.nn.dropout(layer_conv7, dropout)

    layer_conv7 = tf.contrib.layers.batch_norm(layer_conv7)
    layer_conv8, weights_conv8 = new_conv_layer(input=layer_conv7, num_input_channels=256, filter_size=3, num_filters=512, use_pooling=False, use_padding=True)
    layer_conv9, weights_conv9 = new_conv_layer(input=layer_conv8, num_input_channels=512, filter_size=3, num_filters=512, use_pooling=False, use_padding=True)
    layer_conv10, weights_conv10 = new_conv_layer(input=layer_conv9, num_input_channels=512, filter_size=3, num_filters=512, use_pooling=True, use_padding=True)
    layer_conv10 = tf.contrib.layers.batch_norm(layer_conv10)
    layer_conv10 = tf.nn.dropout(layer_conv10, dropout)

    layer_conv10 = tf.contrib.layers.batch_norm(layer_conv10)
    layer_conv11, weights_conv11 = new_conv_layer(input=layer_conv10, num_input_channels=512, filter_size=3, num_filters=512, use_pooling=False, use_padding=True)
    layer_conv12, weights_conv12 = new_conv_layer(input=layer_conv11, num_input_channels=512, filter_size=3, num_filters=512, use_pooling=False, use_padding=True)
    layer_conv13, weights_conv13 = new_conv_layer(input=layer_conv12, num_input_channels=512, filter_size=3, num_filters=512, use_pooling=True, use_padding=True)
    layer_conv13 = tf.contrib.layers.batch_norm(layer_conv13)

    layer_flat, num_features = flatten_layer(layer_conv10)
    layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=4096, use_relu=True)
    layer_fc3 = new_fc_layer(input=layer_fc1, num_inputs=4096, num_outputs=number_of_classes, use_relu=False)
    return layer_fc3

def get_next_batch(dataset_features, dataset_labels, batch_index, batch_size):
    return dataset_features[batch_index*batch_size:(batch_index+1)*batch_size, :], dataset_labels[batch_index*batch_size : (batch_index+1)*batch_size, :]

def train_neural_network(x):
    prediction = neural_net_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost) #default learning rate = 0.001

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # training
        for epoch in range(number_of_epochs):
            epoch_loss = 0
            for batch_index in range(int(dataset_train_features.shape[0] / batch_size)):
                epoch_x, epoch_y = get_next_batch(dataset_train_features, dataset_train_labels, batch_index, batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('epoch number:', epoch, 'epoch loss:', epoch_loss)

            # testing
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('accuracy:', accuracy.eval({x: dataset_test_features, y: dataset_test_labels}))

            # for batch_index in range(int(dataset_test_features.shape[0] / batch_size)):
            #     epoch_test_x, epoch_test_y = get_next_batch(dataset_test_features, dataset_test_labels, batch_index, batch_size)
            #     print('accuracy', accuracy.eval({x: epoch_test_x, y: epoch_test_y}))


dataset_train_features, dataset_train_labels = load_dataset('training.csv')
dataset_test_features, dataset_test_labels = load_dataset('test.csv')
print(dataset_train_features.shape)

pickle_dump(dataset_train_features, '/input/dataset_train_features.pickle')
pickle_dump(dataset_train_labels, '/input/dataset_train_labels.pickle')
pickle_dump(dataset_test_features, '/input/dataset_test_features.pickle')
pickle_dump(dataset_test_labels, '/input/dataset_test_labels.pickle')

dataset_train_features = pickle_retrieve('/input/dataset_train_features.pickle')
dataset_train_labels = pickle_retrieve('/input/dataset_train_labels.pickle')
dataset_test_features = pickle_retrieve('/input/dataset_test_features.pickle')
dataset_test_labels = pickle_retrieve('/input/dataset_test_labels.pickle')

dataset_train_features = dataset_train_features.astype('float32')
dataset_test_features = dataset_test_features.astype('float32')
dataset_train_features = dataset_train_features / 255.0
dataset_test_features = dataset_test_features / 255.0

print('dataset_train_features.shape:', dataset_train_features.shape)
print('dataset_train_labels.shape:', dataset_train_labels.shape)
print('dataset_test_features.shape:', dataset_test_features.shape)
print('dataset_test_labels.shape:', dataset_test_labels.shape)

train_neural_network(x)
