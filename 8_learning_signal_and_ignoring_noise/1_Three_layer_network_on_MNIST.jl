using MLDatasets

# training 28 x 28 x 60,000

images = MNIST(:train).features
targets = MNIST(:train).targets
single_image = MNIST(:train).features[:, :, 1]
single_target = MNIST(:train).targets[1]

# testing 28 x 28 x 10,000

images = MNIST(:test).features
targets = MNIST(:test).targets
single_image = MNIST(:test).features[:, :, 1]
single_target = MNIST(:test).targets[1]