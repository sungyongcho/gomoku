import tensorflow as tf


def create_CNN_model():
    print("Creating CNN model....")
    input_board = tf.keras.layers.Input(shape=(19, 19, 17), name="board_input")

    # Common
    x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(
        input_board
    )
    x = tf.keras.layers.BatchNormalization()(x)

    # Value Head
    v = tf.keras.layers.Conv2D(1, (1, 1), activation="relu", padding="same")(x)
    v = tf.keras.layers.BatchNormalization()(v)
    v = tf.keras.layers.Flatten()(v)
    v = tf.keras.layers.Dense(256, activation="relu")(v)
    v = tf.keras.layers.Dense(1, activation="tanh", name="value_output")(v)

    # Policy Head
    p = tf.keras.layers.Conv2D(2, (1, 1), activation="relu", padding="same")(x)
    p = tf.keras.layers.BatchNormalization()(p)
    p = tf.keras.layers.Flatten()(p)
    p = tf.keras.layers.Dense(362, activation="softmax", name="policy_output")(p)

    model = tf.keras.models.Model(inputs=[input_board], outputs=[p, v])
    model.compile(
        loss=["mean_squared_error", "categorical_crossentropy"], optimizer="adam"
    )

    # model.summary()
    return model


def create_mini_CNN_model():
    print("Creating mini CNN model....")
    input_board = tf.keras.layers.Input(shape=(9, 9, 17), name="board_input")

    # Common
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(
        input_board
    )
    x = tf.keras.layers.BatchNormalization()(x)

    # Value Head
    v = tf.keras.layers.Conv2D(1, (1, 1), activation="relu", padding="same")(x)
    v = tf.keras.layers.BatchNormalization()(v)
    v = tf.keras.layers.Flatten()(v)
    v = tf.keras.layers.Dense(128, activation="relu")(v)
    v = tf.keras.layers.Dense(1, activation="tanh", name="value_output")(v)

    # Policy Head
    p = tf.keras.layers.Conv2D(2, (1, 1), activation="relu", padding="same")(x)
    p = tf.keras.layers.BatchNormalization()(p)
    p = tf.keras.layers.Flatten()(p)
    p = tf.keras.layers.Dense(81, activation="softmax", name="policy_output")(p)

    model = tf.keras.models.Model(inputs=[input_board], outputs=[p, v])
    model.compile(
        loss=["mean_squared_error", "categorical_crossentropy"], optimizer="adam"
    )

    model.summary()
    return model


"""
def create_model():
    model = NeuralNet()
    model.create_network(
        [
            DenseLayer(INPUT_SHAPE, 200, activation="ReLU"),
            DenseLayer(200, 100, activation="ReLU", weights_initializer="zero"),
            DenseLayer(100, 50, activation="ReLU", weights_initializer="zero"),
            DenseLayer(
                50, OUTPUT_SHAPE, activation="softmax", weights_initializer="zero"
            ),
        ]
    )
    return model
"""
