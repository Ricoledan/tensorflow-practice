# Goal: Build and train a neural network that classifies images, then evaluate the accuracy of the model.
# Reference: https://www.tensorflow.org/tutorials/quickstart/beginner

import tensorflow as tf

# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the tf.keras.Sequential model by stacking layers. Choose an optimizer and loss function for training:
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
predictions = model(x_train[:1]).numpy()
predictions
print("predictions ", predictions)

# output
# [[-0.40302122  0.03669894  0.33556265 - 0.17986694  0.04705353  0.3180593 0.56460047  0.00185861  0.12526236  0.12785631]]

# The tf.nn.softmax function converts these logits to "probabilities" for each class
tf.nn.softmax(predictions).numpy()
print("softmax ", tf.nn.softmax(predictions).numpy())

# The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index and returns a scalar loss for each example.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()
print("loss ", loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# The Model.fit method adjusts the model parameters to minimize the loss
model.fit(x_train, y_train, epochs=5)
print("model fit", model.fit(x_train, y_train, epochs=5))
