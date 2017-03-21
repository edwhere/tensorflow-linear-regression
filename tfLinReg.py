"""
Program to learn multivariate linear regression coefficients from data observations. The program uses a
Gradient Descent algorithm available in TensorFlow. The multivariate regression model can be defined by"
      y = w0 + w1 x1 + w2 x2 + ... + wn xn
The set {x1, x2, ..., xn} is known as the feature set (independent variables) and the term y is known as
the target (dependent variable). Given m observations of features and target values, the algorithm
computes the optimal set of coefficients {w0, w1, w2, ..., wn}. The term w0 is a bias term.
"""
import sys
import tensorflow as tf
import numpy as np
import argparse


def parse_command_line():
    """
    Parses command-line arguments
    :return: A dictionary object with the following arguments: rootname (string), learning (float), epochs (int)
    """
    parser = argparse.ArgumentParser()

    # Required arguments
    required_args = parser.add_argument_group("required arguments")
    help_msg = "The root name for all files (data, target, errors). The program uses two files as input: "
    help_msg += "<rootname>.data.txt and <rootname>.target.txt. The program generates one file as output: "
    help_msg += "<rootname>.mse.txt (the mean squared error per epoch)."
    required_args.add_argument("-r", "--rootname", help=help_msg)

    # Optional arguments
    parser.add_argument("-l", "--learning",
                        help="learning rate parameter (default: 0.001)", type=float, default=0.001)

    parser.add_argument("-e", "--epochs", help="number of training epochs (default: 100)", type=int, default=100)

    # Parsing and results
    args = parser.parse_args()

    options = {
        "rootname": args.rootname,
        "learning": args.learning,
        "epochs": args.epochs
    }

    return options


def prepare_data(xmat, yvec):
    """
    Takes a raw data matrix and converts it into an extended matrix that includes a column for a bias term. Also takes
    a raw target (output) vector and converts it into a column vector. The raw data matrix has a size of nrows x ncols.
    The number of collected data points is nrows. The number of features per data point is ncols. The extended matrix
    has a size of nrows x (ncols + 1). The first column of the extended matrix is set to ones.
    :param xmat: The raw data matrix (a numpy array)
    :param yvec: The raw target (output) vector (a numpy vector)
    :return: Extended matrix Xdat and target column vector Ydata; both numpy arrays
    """
    nrows = xmat.shape[0]

    if nrows != len(yvec):
        print "[Error] number of data points must be the same as the number of target values"
        sys.exit(1)

    # prepare a column of ones and append it as the first column (indicates that the first coeff is the bias term)
    col_ones = np.ones((nrows, 1), dtype=float)
    Xdat = np.concatenate((col_ones, xmat), axis=1)

    # reshape y vector into a single column vector
    Ydat = np.reshape(yvec, (nrows, 1))

    return Xdat, Ydat


def load_data_from_file(rootname):
    """
    Loads data from two files that share a rootname: <rootname>.data.txt and <rootname>.target.txt
    :param rootname: The root name of the data and target files
    :return: A matrix with the data values and a vector with the target values
    """
    datafile = rootname + ".data.txt"
    targetfile = rootname + ".target.txt"

    data = np.loadtxt(datafile)
    target = np.loadtxt(targetfile)

    return data, target


def save_errors_to_file(rootname, error_vector):
    """
    Saves the error_vector values to a file defined by rootname: <rootname>.sse.txt
    :param rootname: The root name that identifies all files related to this run (string)
    :param error_vector: A list of mean squared errors (MSE) returned by TensorFlow per epoch
    """
    filename = rootname + ".mse.txt"
    with open(filename, "w") as fh:
        for error in error_vector:
            fh.write(str(error) + "\n")


def tf_reg_operators(num_coeffs, learning_param):
    """
    Defines TensorFlow operators for searching the coefficients in a linear
    regression problem
    :param num_coeffs: The number of coefficients (int)
    :param learning_param: The learning rate parameter (float). Typically less than 0.1
    :return: A dictionary object containing the declared operators in a TensorFlow graph
    """
    ops = {}

    # Define placeholders for 2 tensors: the data matrix and the output (target) column vector
    ops['Xvalue'] = X = tf.placeholder(tf.float32, [None, num_coeffs])
    ops['Yvalue'] = Y = tf.placeholder(tf.float32, [None, 1])

    # Define a fixed-size tensor for the model coefficients
    ops['Wvalue'] = W = tf.Variable(tf.ones([num_coeffs, 1]))

    # Define an initialization function (initializes previously defined variables)
    ops['init'] = tf.global_variables_initializer()

    # Define a prediction operation, which in this case is a multiplication of data matrix with a coeff. column vector
    ops['predict'] = Yp = tf.matmul(X, W)

    # Define an operator that computes the mean squared error
    ops['cost'] = J = tf.reduce_mean(tf.square(tf.subtract(Y, Yp)))

    # Define an operator that finds model coefficients using gradient descent. This operation updates coefficients
    # for each point in the data set (i.e. the operation goes through one epoch).
    ops['train'] = tf.train.GradientDescentOptimizer(learning_param).minimize(J)

    return ops


def tf_reg_session(ops, Xdata, Ydata, num_epochs):
    """
    Given a collection of operators (TensorFlow graph) this function runs a sequence of operators that solve the
    linear regression problem using gradient descent.
    :param ops: The collection of TensorFlow operators
    :param Xdata: The extended data matrix
    :param Ydata: The column vector that contains the output (or target) values
    :param num_epochs: The number of epochs that the algorithm should iterate
    :return: A column vector with the final coefficients and a list of mean squared errors (MSE). There is one MSE
    value per epoch.
    """
    with tf.Session() as session:
        session.run(ops['init'])

        error_values = []

        for epoch in range(num_epochs):
            # current_coeffs =  session.run(ops['Wvalue'])

            # Find predicted values with current coefficients
            session.run(ops['predict'], feed_dict={ops['Xvalue']: Xdata, ops['Yvalue']: Ydata})

            # Find the mean squared error for the current epoch and append to a list
            Jval = session.run(ops['cost'], feed_dict={ops['Xvalue']:Xdata, ops['Yvalue']:Ydata})
            error_values.append(Jval)

            # Apply a gradient descent training algorithm to a batch of data/target values. Also
            # update the coefficients
            session.run(ops['train'], feed_dict={ops['Xvalue']: Xdata, ops['Yvalue']: Ydata})

        # Find final set of coefficients
        final_coeffs = session.run(ops['Wvalue'])

        return final_coeffs, error_values


def display_coefficients(coeff_column):
    """
    Displays the resulting coefficients
    :param coeff_column: A column vector with the resulting coefficients
    """
    cvec = np.reshape(coeff_column, (coeff_column.shape[0]))

    print "\n--------------------------------------------------------"
    print "Model Coefficients: "
    print "--------------------------------------------------------"
    for k in range(len(cvec)):
        print "w{}: {}".format(k, cvec[k])
    print "--------------------------------------------------------\n"

if __name__ == "__main__":
    # Extract parameters from command line
    opts = parse_command_line()

    # Read data from files using the rootname
    x_raw, t_raw = load_data_from_file(opts["rootname"])

    # Prepare data (X) and actual output (Y) matrices
    newX, newY = prepare_data(x_raw, t_raw)

    # Define the tensorflow operators for linear regression
    operators = tf_reg_operators(num_coeffs=newX.shape[1], learning_param=opts["learning"])

    # Evaluate the tensorflow operators and return results
    coeffs, errvals = tf_reg_session(ops=operators, Xdata=newX, Ydata=newY, num_epochs=opts["epochs"])

    # Display the final coefficients
    display_coefficients(coeff_column=coeffs)

    # Save cost function values (sum of squared errors) to a file <rootname>.sse.txt
    save_errors_to_file(opts["rootname"], errvals)
    print "\nSaved error values to error_values.txt"
