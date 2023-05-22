using MLDatasets

# Import the MNIST dataset
trainset = MNIST(:train)

# Split the dataset into training and test sets
x_train, y_train = MNIST(:train)
x_test, y_test = MNIST(:test)

(images, labels) = (reshape(train_x.features[:,:,1:1000], (28*28, 1000)), train_y.features[1:1000])