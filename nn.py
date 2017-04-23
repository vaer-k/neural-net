import os
import csv
import neural_funcs
import cPickle as pickle
import numpy as np
import pandas as pd

TEST = "./raw_data/test.csv"
TRAIN = "./raw_data/train.csv"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class NeuralNetwork:
    """
    :param layers: A list containing the number of units in each layer.
    The last integer element is considered the output layer
    :param activation_: The activation function to be used. Can be
    "logistic" or "tanh"
    """
    def __init__(self,
                 activation_="logistic",
                 cost="mse",
                 alpha=0.05,
                 lamda=0.5,
                 epochs=30,
                 layers=None,
                 batch_size=10,
                 evaluate=True,
                 weight_init="epsilon"):

        layers = layers or [25, 10]  # Default to one hidden layer with 25 units and one 10 unit output layer
        # layers = layers or [256, 64, 10]

        if len(layers) < 2:
            raise TypeError("The layers arg should be a list containing at least two integers")

        self.curr_cost = 0
        self.weights = [neural_funcs.Weight().get(weight_init)(x, y) for x, y in zip(layers[:-1], layers[1:])]
        self._params = {
            "evaluate": evaluate,
            "activation": activation_,
            "cost": cost,
            "alpha": alpha,
            "lamda": lamda,
            "epochs": epochs,
            "num_layers": len(layers) + 1,
            "layers": layers,
            "hidden_layers": layers[:-1],
            "output_layer": layers[-1],
            "batch_size": batch_size,
            "weight_init": weight_init
        }

    def rst_weights(self):
        layers = self.params["layers"]
        self.weights = [neural_funcs.Weight().get(self.params["weight_init"])(x, y) for x, y in zip(layers[:-1], layers[1:])]

    def fit(self, X=None, y=None, tuning=None):
        if not X:
            print("Fitting with default (MNIST) training data and labels...")
            X = pd.read_csv(TRAIN)
            y = np.array([X["label"].as_matrix()])
            X = X.drop("label", axis=1).as_matrix()

        self.params["layers"] = [X.shape[1]] + self.params["layers"]
        self._sgd(X, y, tuning=None)

    def predict(self, x):
        _, output = self._feedforward(x)
        return np.argmax(output[-1])

    def evaluate(self, test):
        test_results = [(self.predict(row[1:]), row[0]) for row in test]
        accuracy = neural_funcs.Evaluation.accuracy(test_results)
        f1 = neural_funcs.Evaluation.f1(test_results)
        return accuracy, f1

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, key_value):
        key, value = key_value
        self._params[key] = value
        return self._params

    def _feedforward(self, x):
        weighted_sums = []
        activations = [x]
        a = x  # init the first activation layer to the input X
        for theta in self.weights:
            a = np.insert(a, 0, 1)  # add bias term
            z = np.dot(theta, a)
            weighted_sums.append(z)
            a = neural_funcs.Activation().get(self.params["activation"])(z)

            activations.append(a)

        return weighted_sums, activations

    def _backprop(self, x, y):
        weighted_sums, activations = self._feedforward(x)
        nabla = [np.zeros(w.shape) for w in self.weights]

        # Use the output layer activations and weights to initialize error delta
        activ_deriv = neural_funcs.Activation().get(self.params["activation"])(weighted_sums[-1], derivative=True)
        delta = neural_funcs.Cost().get(self.params["cost"])(self._label(y), activations[-1], derivative=True, activation_deriv=activ_deriv)

        delta = np.array([delta]).T
        a = np.array([np.insert(activations[-2], 0, 1)]).T
        nabla[-1] = np.dot(delta, a.T)

        delta = np.insert(delta, 0, 1)
        delta = np.array([delta]).T

        # Backpropagate error
        for l in xrange(2, self.params["num_layers"]):
            delta = np.dot(self.weights[-l + 1].T, delta[1:]) \
                    * np.array([np.insert(neural_funcs.Activation().get(self.params["activation"])(weighted_sums[-l], derivative=True), 0, 1)]).T

            a = np.array([np.insert(activations[-l - 1], 0, 1)]).T
            nabla[-l] = np.dot(delta[1:], a.T)

        return nabla

    def _label(self, y):
        label = np.zeros(self.params["output_layer"])
        label[y] = 1
        return label

    def _sgd(self, X, y, tuning=None):
        tuning = tuning or ('lamda', lambda x: x / 10)
        data = np.concatenate((y.T, X), axis=1)
        training_length = int(len(data) * .9)
        train_val = data[:training_length]
        test = data[training_length:]

        cross_val_size = int(len(train_val) * .10)

        fold_num = 0
        fold_score = {}
        for cross_start in xrange(0, len(train_val), cross_val_size):
            self.rst_weights()
            if cross_start > 0:
                self.params[tuning[0]] = tuning[1](self.params[tuning[0]])

            print("\nStarting kfold #{0} with hyperparameter \"{1}\" at value {2}".format(fold_num, tuning[0], self.params[tuning[0]]))

            fold_num += 1
            filename = ROOT_DIR + "/models/score_{0}.csv".format(fold_num)

            try:
                os.remove(filename)
            except OSError:
                pass

            fold_score[fold_num] = {"accu": 0, "f1": 0, "weights": self.weights, "params": self.params}
            cross_end = cross_start + cross_val_size
            cross_val = train_val[cross_start:cross_end]
            train = np.concatenate((train_val[:cross_start], train_val[cross_end:]))

            n = len(train)
            alpha = self.params["alpha"]
            for i in xrange(self.params["epochs"]):
                train = np.random.permutation(train)
                batches = [train[m:m + self.params["batch_size"]]
                           for m in xrange(0, n, self.params["batch_size"])]

                print("Updating model with alpha {0}".format(round(alpha, 3)))
                for batch in batches:
                    y = batch[:, 0]
                    X = batch[:, 1:]
                    self._update_model(X, y)

                alpha *= 0.90 if alpha > 0.01 else .95

                self.curr_cost = self._compute_cost(train)

                if not i % 5:
                    print("\nCurrent cost: {0}\n".format(round(self.curr_cost, 3)))

                if self.params["evaluate"]:
                    train_accu = self.evaluate(train)[0] * 100
                    train_f1 = self.evaluate(train)[1]
                    cross_accu = self.evaluate(cross_val)[0] * 100
                    cross_f1 = self.evaluate(cross_val)[1]

                    with open(filename, "a+") as f:
                        writer = csv.writer(f)

                        if i == 0:
                            writer.writerow(("epoch", "cross_accu", "cross_f1", "cost"))

                        writer.writerow((i, cross_accu, cross_f1, self.curr_cost))

                    print("Epoch #{0} results:".format(i + 1))
                    print("\tTrain set:")
                    print("\t    accuracy: {0}%".format(train_accu))
                    print("\t    F1: {0}".format(train_f1))
                    print("\tCross validation set:")
                    print("\t    accuracy: {0}%".format(cross_accu))
                    print("\t    F1: {0}".format(cross_f1))

            fold_score[fold_num]["accu"] = cross_accu
            fold_score[fold_num]["f1"] = cross_f1
            fold_score[fold_num]["weights"] = list(self.weights)
            fold_score[fold_num]["params"] = dict(self.params)

        # Select the best fold and load params/weights back into model before testing
        best = 0
        for fold, scores in fold_score.iteritems():
            if best < scores["f1"]:
                best = scores["f1"]
                self._params = scores["params"]
                self.weights = scores["weights"]

        test_accu = self.evaluate(test)[0] * 100
        test_f1 = self.evaluate(test)[1]
        print("\nFinal results:")
        print("Best value for \"{0}\" param: {1}".format(tuning[0], self.params[tuning[0]]))
        print("\tTest set:")
        print("\t    accuracy: {0}%".format(test_accu))
        print("\t    F1: {0}\n".format(test_f1))

        with open(ROOT_DIR + "/models/weights.pkl".format(fold_num), 'w') as f:
            pickle.dump(self.weights, f)

        with open(ROOT_DIR + "/models/params.pkl".format(fold_num), 'w') as f:
            pickle.dump(self.params, f)

    def _update_model(self, X, y):
        nabla = [np.zeros(w.shape) for w in self.weights]
        m = len(X)
        for i in xrange(m):
            delta_nabla = self._backprop(X[i], y[i])
            nabla = [n + dn for n, dn in zip(nabla, delta_nabla)]

        # Update model weights
        alpha = self.params["alpha"]
        lamda = self.params["lamda"]
        self.weights = [w - (alpha / m) * n for w, n in zip(self.weights, nabla)]

        for theta in self.weights:
            theta[:, 1:] = [w - ((alpha * lamda) / m) * w for w in theta[:, 1:]]

    def _compute_cost(self, data):
        cost = 0
        m = len(data)
        for row in data:
            _, activations = self._feedforward(row[1:])
            cost += neural_funcs.Cost().get(self.params["cost"])(self._label(row[0]), activations[-1]) / m

        cost += np.sum([(self.params["lamda"] / (2 * m)) * np.sum(np.square(theta[:, 1:])) for theta in self.weights])
        return cost
