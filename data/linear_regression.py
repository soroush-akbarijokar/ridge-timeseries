import numpy as np
import time
import matplotlib.pyplot as plt



'''
Code for ridge regression closed-form solution
'''


def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(1), dtype=float)
    return data

def get_data_matrices_train(data, N):
    T = len(data)
    X = np.zeros((T-N, N))
    y = np.zeros(T-N)
    for i in range(N, T):
        X[i-N] = data[i-N:i]
        y[i-N] = data[i]
    return X, y

def get_data_matrices_test(data_train, data_test, N):
    # Create the initial window from the end of the training data
    window = list(data_train[-N:])

    X_test = []
    y_test = list(data_test)

    for point in data_test:
        X_test.append(window[-N:])
        window.append(point)

    return np.array(X_test[:-1]), np.array(y_test[:-1])

def ridge_regression(X, y, lam=100):
    num_features = X.shape[1]
    identity = np.eye(num_features)
    w = np.linalg.inv(X.T @ X + lam * identity) @ X.T @ y
    b = np.mean(y - X @ w)
    return w, b

def predict(X, w, b):
    return X @ w + b

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def time_series_cross_validation(X, y, lambdas, num_splits=5):
    T = len(y)
    split_length = T // num_splits
    errors = np.zeros((num_splits, len(lambdas)))

    for i in range(num_splits):
        train_end = split_length * (i + 1)
        X_train_cv = X[:train_end]
        y_train_cv = y[:train_end]
        X_val_cv = X[train_end:train_end + split_length]
        y_val_cv = y[train_end:train_end + split_length]
        print(f'proceed w split {i}')
        for j, lam in enumerate(lambdas):
            w, b = ridge_regression(X_train_cv, y_train_cv, lam)
            y_pred = predict(X_val_cv, w, b)
            errors[i, j] = mean_squared_error(y_val_cv, y_pred)

    mean_errors = errors.mean(axis=0)
    best_lambda_idx = np.argmin(mean_errors)
    return lambdas[best_lambda_idx]

# Load datasets
data_train = load_data('train_series.csv')
data_test = load_data('test_series.csv')
N = 35040

# Preparing the training dataset
X_train, y_train = get_data_matrices_train(data_train, N)
print('got the data')
# Optimize lam using time series cross-validation
# lambdas = [np.logspace(-4, 4, 100)]
lambdas = [0.01,0.1,1.0,10,100]
# print('wait for tuning')
# best_lam = time_series_cross_validation(X_train, y_train, lambdas)
# print('tuning done!')
print('waiting for the training')
# Train Ridge Regression using best lam
start_time = time.time()
w, b = ridge_regression(X_train, y_train)
end_time = time.time()
print('training done!')
# Make predictions on test data
# Prepare test data
X_test, y_test = get_data_matrices_test(data_train, data_test, N)

y_pred = predict(X_test, w, b)
print('prediction done!')
# Plotting the predicted temperature series vs ground truth series
plt.figure(figsize=(14, 7))
plt.plot(y_test, label="Ground Truth", color='blue')
plt.plot(y_pred, label="Predicted", color='red', linestyle='dashed')
plt.xlabel('Time step')
plt.ylabel('Temperature (°C)')
plt.title('Predicted Temperature vs Ground Truth')
plt.legend()
plt.show()

# Calculate and print MSE and training time
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Erroron Test Data: {mse}")
print(f"Training Time: {end_time - start_time} seconds")

'''
Code for ridge regression Cholesky Decomposition
'''

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(1), dtype=float)
    return data


def get_data_matrices_train(data, N):
    T = len(data)
    X = np.zeros((T - N, N))
    y = np.zeros(T - N)
    for i in range(N, T):
        X[i - N] = data[i - N:i]
        y[i - N] = data[i]
    return X, y


def get_data_matrices_test(data_train, data_test, N):
    # Create the initial window from the end of the training data
    window = list(data_train[-N:])

    X_test = []
    y_test = list(data_test)

    for point in data_test:
        X_test.append(window[-N:])
        window.append(point)

    return np.array(X_test[:-1]), np.array(y_test[:-1])


def ridge_regression_cholesky(X, y, lam=100):

    num_samples, num_features = X.shape
    identity = np.eye(num_features)

    # Compute the matrix for which we want the Cholesky decomposition
    A = X.T @ X + lam * identity

    # Cholesky decomposition
    L = np.linalg.cholesky(A)

    # Solve the two triangular systems
    z = np.linalg.solve(L, X.T @ y)
    w = np.linalg.solve(L.T, z)

    # Compute bias
    b = np.mean(y - X @ w)

    return w, b


def predict(X, w, b):
    return X @ w + b


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def time_series_cross_validation(X, y, lambdas, num_splits=5):
    T = len(y)
    split_length = T // num_splits
    errors = np.zeros((num_splits, len(lambdas)))

    for i in range(num_splits):
        train_end = split_length * (i + 1)
        X_train_cv = X[:train_end]
        y_train_cv = y[:train_end]
        X_val_cv = X[train_end:train_end + split_length]
        y_val_cv = y[train_end:train_end + split_length]
        print(f'proceed w split{i}')
        for j, lam in enumerate(lambdas):
            w, b = ridge_regression_cholesky(X_train_cv, y_train_cv, lam)
            y_pred = predict(X_val_cv, w, b)
            errors[i, j] = mean_squared_error(y_val_cv, y_pred)

    mean_errors = errors.mean(axis=0)
    best_lambda_idx = np.argmin(mean_errors)
    return lambdas[best_lambda_idx]


# Load datasets
data_train = load_data('train_series.csv')
data_test = load_data('test_series.csv')
N = 35040
print('got the data')
# Preparing the training dataset
X_train, y_train = get_data_matrices_train(data_train, N)

# Optimize lam using time series cross-validation
# lambdas = [np.logspace(-4, 4, 100)]
# lambdas = [0.01,0.1,1.0,10,100]

# best_lam = time_series_cross_validation(X_train, y_train, lambdas)
# print('Tuned parameters')
start_time = time.time()
# Train Ridge Regression using best lam
w, b = ridge_regression_cholesky(X_train, y_train)
end_time = time.time()
print('Trained the model')
# Prepare test data
X_test, y_test = get_data_matrices_test(data_train, data_test, N)
y_pred = predict(X_test, w, b)
print('waiting for plots')
# Plotting the predicted temperature series vs ground truth series
plt.figure(figsize=(14, 7))
plt.plot(y_test, label="Ground Truth", color='blue')
plt.plot(y_pred, label="Predicted", color='red', linestyle='dashed')
plt.xlabel('Time step')
plt.ylabel('Temperature (°C)')
plt.title('Predicted Temperature vs Ground Truth')
plt.legend()
plt.show()

# Calculate and print MSE and training time
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Data: {mse}")
print(f"Training Time: {end_time - start_time} seconds")


'''
Code for ridge regression conjugate gradient
'''


def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(1), dtype=float)
    return data


def get_data_matrices_train(data, N):
    T = len(data)
    X = np.zeros((T - N, N))
    y = np.zeros(T - N)
    for i in range(N, T):
        X[i - N] = data[i - N:i]
        y[i - N] = data[i]
    return X, y


def get_data_matrices_test(data_train, data_test, N):
    # Create the initial window from the end of the training data
    window = list(data_train[-N:])

    X_test = []
    y_test = list(data_test)

    for point in data_test:
        X_test.append(window[-N:])
        window.append(point)

    return np.array(X_test[:-1]), np.array(y_test[:-1])


def ridge_regression_cg(X, y, lam=1.0, max_iters=1000, tol=1e-6):
    num_samples, num_features = X.shape
    A = X.T @ X + lam * np.eye(num_features)
    b_vec = X.T @ y

    w = np.zeros(num_features)
    r = b_vec - A @ w
    p = r.copy()
    rsold = r.T @ r

    for _ in range(max_iters):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        w = w + alpha * p
        r = r - alpha * Ap
        rsnew = r.T @ r
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    bias = np.mean(y - X @ w)
    return w, bias


def predict(X, w, b):
    return X @ w + b


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Load datasets
data_train = load_data('train_series.csv')
data_test = load_data('test_series.csv')
N = 35040

# Preparing the training dataset
X_train, y_train = get_data_matrices_train(data_train, N)
print('got the data')

start_time = time.time()
# Train Ridge Regression using best lam
w, b = ridge_regression_cg(X_train, y_train)
end_time = time.time()
# Make predictions on test data
print('model trained')
# Prepare test data
X_test, y_test = get_data_matrices_test(data_train, data_test, N)
y_pred = predict(X_test, w, b)

# Plotting the predicted temperature series vs ground truth series
plt.figure(figsize=(14, 7))
plt.plot(y_test, label="Ground Truth", color='blue')
plt.plot(y_pred, label="Predicted", color='red', linestyle='dashed')
plt.xlabel('Time step')
plt.ylabel('Temperature (°C)')
plt.title('Predicted Temperature vs Ground Truth')
plt.legend()
plt.show()

# Calculate and print MSE and training time
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Data: {mse}")
print(f"Training Time: {end_time - start_time} seconds")


