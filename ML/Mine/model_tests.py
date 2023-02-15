from models import Sequential
from layers import Dense
import tensorflow as tf


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    num_inputs = x_train[0]
    print(type(num_inputs))
