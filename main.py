import numpy as np
import pandas as pd

np.random.seed(1)


def normalize(x):
    x_normalized = np.zeros(np.shape(x))

    for i in range(np.shape(x)[1]):
        for j in range(np.shape(x[:, i])[0]):
            x_normalized[j][i] = (x[j][i] - max(x[:, i])) / (max(x[:, i]) - min(x[:, i]))

    return x_normalized


def normalize_output(output):
    y_normalized = np.zeros(np.shape(output)[0])

    for j in range(np.shape(output)[0]):
        y_normalized[j] = (output[j] - max(output)) / (max(output) - min(output))

    return y_normalized


def denormalize_output(output, y_normalized):
    return y_normalized * max(output) - y_normalized * min(output) + max(output)


def relu(x):
    relued_array = np.zeros(np.shape(x))
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            relued_array[i][j] = max(0.1 * x[i][j], x[i][j])

    return relued_array


def relu_derivative(x):
    relued_array = np.zeros(np.shape(x))
    for i in range(np.shape(x)[0]):
        for j in range(np.shape(x)[1]):
            if x[i][j] < 0:
                relued_array[i][j] = 0.1
            else:
                relued_array[i][j] = 1

    return relued_array


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_derivative(x):
    return 1 - x ** 2


data_file_path = './input/melb_data.csv'
data = pd.read_csv(data_file_path)

data = data.dropna(axis=0)

y = data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[features]

test_dataset = X.head()
test_output = y.head()

test_dataset = np.array(test_dataset)
test_output = np.array(test_output)

X = np.array(X)
y = np.array(y)

l1_weights = np.random.random([np.shape(X)[1], np.shape(X)[1] - 1]) - 0.5
l2_weights = np.random.random([np.shape(l1_weights)[1], 1]) - 0.5

Y = normalize_output(y)
X_normalized = normalize(X)

for k in range(100):
    l1 = relu(np.dot(X_normalized, l1_weights))
    l2 = tanh(np.dot(l1, l2_weights))

    l2_error = (Y - l2.T).T * tanh_derivative(l2)
    l1_error = (relu_derivative(l1)) * (np.dot(l2_error, l2_weights.T))
    delta_l1 = np.dot(X_normalized.T, l1_error) * 0.0001
    delta_l2 = np.dot(l1.T, l2_error) * 0.0001

    l1_weights = l1_weights + delta_l1
    l2_weights = l2_weights + delta_l2



