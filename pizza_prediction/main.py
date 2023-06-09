import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea

def plot_setup():
    sea.set()
    plt.axis([0, 50, 0, 50])                            # scale axes (0, 50)
    plt.xticks(fontsize=15)                             # set x axis ticks
    plt.yticks(fontsize=15)                             # set y axis ticks
    plt.xlabel('Reservations', fontsize=14)             # set x axis label
    plt.xlabel('Pizzas', fontsize=14)                   # set y axis label


def plot(X, Y, w, b):
    plot_setup()
    plt.plot(X, Y, "bo")                   # use blue circle markers, not like red x markers, 'rx', etc.
    plt.plot(X, predict(X, w, b), label='Fitting line', color='red', linestyle='-')
    plt.show()


def predict(X, w, b):
    return X * w + b


# An error should always be positive
def loss(X, Y, w, b):
    # square or abs value to avoid negative error, but squaring has an additional benefits
    # np.average is for the input array X of possibly multiple elements, like numpy array
    # X and Y are both numpy arrays
    return np.average((predict(X, w, b) - Y) ** 2) 


def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)
    return (w_gradient, b_gradient)


def train_with_gradient(X, Y, iterations, lr):
    # This version of train with gradient is much faster than the previous one
    w = 0
    b = 0
    for i in range(iterations):
        if (i % 5000 == 0):
            print("Iteration %4d => Loss: %.10f" % (i, loss(X, Y, w, b)))
        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * lr
        b -= b_gradient * lr
    return w, b


def train(X, Y, iterations, lr): # lr -> learning rate
    w = 0                        # weight
    b = 0                        # bias
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        if i%300 == 0:
            print("Iteration %4d => Loss: %.6f" % (i, current_loss))
        if loss(X, Y, w + lr, b) < current_loss:            # Updating weight
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss:          # Updating weight
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss:          # Updating bias
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss:          # Updating bias
            b -= lr
        else:
            return w, b
        
    raise Exception("Couldn't converge within %d iterations" % iterations)

def main():
    X, Y = np.loadtxt('pizza.txt', skiprows=1, unpack=True)
    w, b = train(X, Y, iterations=10000, lr=0.01)
    print("\nw=%.3f, b=%.3f" % (w, b))
    print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))
    plot(X, Y, w, b)


def main1():
    # Using gradient vs using the errors and tuning in w and lr.
    X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
    w, b = train_with_gradient(X, Y, iterations=20000, lr=0.0001)
    print("\nw=%.10f" % w)
    plot(X, Y, w, b)

if __name__ == '__main__':
    # main()
    main1()