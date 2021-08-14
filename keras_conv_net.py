import numpy as np
import tensorflow as tf
from tensorflow import keras

class conv_net():

    def __init__(self,
                layersizes, activations, learning_rate=0.01,
                optimizer='adam', losses='huber', huber_delta = 1.0,
                training_epochs = 1, steps_per_epoch = None):

        # Store Training Information
        self.training_epochs = training_epochs
        self.steps_per_epoch = steps_per_epoch

        # Establish a Model
        self.model = tf.keras.Sequential([keras.Input(shape=(layersizes[0],))])
        for i in np.arange(1, len(layersizes)):
            self.model.add(keras.layers.Dense(units=layersizes[i], activation=activations[i-1]))

        # Define the optimiser
        if optimizer=='adam': optim = keras.optimizers.Adam(lr=learning_rate)
        elif optimizer=='rmsprop': optim = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else: optim = keras.optimizers.Adam(lr=learning_rate)

        # Define loss function based on input
        if losses=='huber': loss_function = tf.losses.Huber(delta=huber_delta)
        elif losses=='crossentropy': loss_function = tf.losses.CategoricalCrossentropy(from_logits=True)
        elif losses=='mse': loss_function = tf.losses.MeanSquaredError()
        else: loss_function = tf.losses.Huber(delta=huber_delta)

        # Compile the model
        self.model.compile(optimizer=optim,
              loss=loss_function,
              metrics=['accuracy'])

    # Train the Network
    def train_network(self, X, Y):
        dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(len(Y)).batch(len(Y))
        self.model.fit(
            dataset,
            epochs=self.training_epochs,
            steps_per_epoch=self.steps_per_epoch,
            verbose=False
        )

    # Predict Network Outputs
    def predict_network(self, X, Y):
        dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(len(Y))
        return self.model.predict(dataset)
