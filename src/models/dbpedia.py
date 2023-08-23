
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,Embedding
from tensorflow.keras.layers import Conv1D,Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import numpy as np
max_features = 30522  # size of the vocabulary of tiny-bert
embedding_dim = 256

# SHARED MODEL DEFINITION
DBPEDIA_MATRIX_PATH = 'todo.npy'

def build_model_learn_embeddings(momentum=0.0, dropouts=False):
    """ Build the local model

    Args:
        momentum (float, optional): momentum value for SGD. Defaults to 0.0.
        dropouts (bool, optional): if True use dropouts. Defaults to False.

    Returns:
        object: model object
    """

    text_input = tf.keras.Input(shape=(embedding_dim, ), name='ids')

    x = tf.keras.layers.Embedding(max_features + 1, embedding_dim)(text_input)

    x = tf.keras.layers.Conv1D(
        128, 7, padding="valid", activation="relu", strides=3)(x)
    x = tf.keras.layers.Conv1D(
        128, 7, padding="valid", activation="relu", strides=3)(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)

    predictions = tf.keras.layers.Dense(
        14, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(text_input, predictions)

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model


def getModel():

    embedding_matrix = np.load(DBPEDIA_MATRIX_PATH)
    num_tokens = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]

    embedding_layer = tf.keras.layers.Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(
            embedding_matrix),
        trainable=False,
    )

    int_sequences_input = tf.keras.Input(shape=(None,), dtype="int64")
    embedded_sequences = embedding_layer(int_sequences_input)

    x = tf.keras.layers.Conv1D(
        128, 7, padding="valid", activation="relu", strides=3)(embedded_sequences)
    x = tf.keras.layers.Conv1D(
        128, 7, padding="valid", activation="relu", strides=3)(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)

    predictions = tf.keras.layers.Dense(
        14, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(int_sequences_input, predictions)

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model
