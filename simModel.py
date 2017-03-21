"""
Simulates a multivariate regression model of the form:  y = w0 + w1 x1 + w2 x2 + . . . + wn xn + v
The set {x1, x2, ..., xn} constitutes a feature vector. The term y constitutes the observed dependent
or target variable. The term v represents Gaussian random noise that obscures the observation.
"""
import argparse
import numpy as np


def command_line_arguments():
    """
    Parse command line arguments and return values using a dictionary object
    :return: A dictionary object with root, size, coeffs, mean, variance, max, min
    """
    parser = argparse.ArgumentParser()

    help_msg = "Root name of files that store the data and target results of this program."
    help_msg += "The results are stored in <rootname>.data.txt and <rootname>.target.txt"

    # Required argument list
    required_args = parser.add_argument_group("required arguments")

    required_args.add_argument("-r", "--rootname", help=help_msg, required=True)

    required_args.add_argument("-c", "--coeffs", nargs='+', help="Model coefficients: w0 w1 w2 ...",
                               type=float, required=True)

    # Optional argument list
    parser.add_argument("-s", "--size",
                        help="Number of data samples in the simulation (default: 100)", type=int, default=100)

    parser.add_argument("-m", "--mean", help="Noise mean value (default: 0.0)", type=float, default=0.0)

    parser.add_argument("-v", "--variance",
                        help="Noise variance (default: 1.0)", type=float, default=1.0)

    parser.add_argument("-mx", "--max",
                        help="Maximum or highest value for data range (default: 10)", type=float, default=10.0)

    parser.add_argument("-mn", "--min",
                        help="Minimum or lowest value for data range (default: 0)", type=float, default=0.0)

    # Parsing and results
    args = parser.parse_args()

    arguments = {
        'root': args.rootname,
        'size': int(args.size),
        'coeffs': [float(elem) for elem in args.coeffs],
        'mean': float(args.mean),
        'variance': float(args.variance),
        'max': float(args.max),
        'min': float(args.min)
    }

    return arguments


def generate_data(options):
    """
    Generates data according the model y = w0 + w1 x1 + w2 x2 + ... wn xn + v  where {x1, x2, ... xn} are
    dependent variables (features). The term v is Gaussian noise. Saves a data matrix that contains the
    observations: rows of data points where each data point is defined by its n features. Also saves a
    target column vector with all the values for y (the target or dependent variable).
    :param options: A dictionary object with user options entered using a command line
    """
    num_coeffs = len(options["coeffs"])   # Number of coefficients entered by the user
    num_vars = num_coeffs - 1    # The w0 coeff is a bias term and does not have an associated data variable
    num_data = options["size"]   # Number of data points requested by the user

    # Generate a data matrix with num_data rows and num_vars columns
    xdata = np.random.uniform(low=options["min"], high=options["max"], size=(num_data, num_vars))

    # Generate a column of ones
    column = np.ones((num_data, 1), dtype=float)

    # Generate a extended data matrix whose first column is all ones. This matrix can be used to generate
    # an output using Y = Xext W
    xext = np.concatenate((column, xdata), axis=1)

    # Generate the vector of coefficients for the equation y = w0 + w1 x1 + w2 x2 + . . .
    # The first coefficient is always the bias term
    colcoeffs = np.reshape(options["coeffs"], newshape=(num_coeffs, 1))

    # Generate the output using Y = Xext W
    y_out = np.matmul(xext, colcoeffs)

    # Generate a noise column vector using the Normal distribution. The vector has the same size as the output Y data
    mean = options["mean"]
    stddev = np.sqrt(options["variance"])

    print "noise standard deviation: ", stddev

    noise = np.random.normal(loc=mean, scale=stddev, size=(num_data, 1))

    print "max noise value: ", np.max(noise)
    print "min noise value: ", np.min(noise)

    # Add noise to the output Y data. The result is the observed output variable Y (also called the target)
    ydata = np.add(y_out, noise)

    print "max y value: ", np.max(ydata)
    print "min y value: ", np.min(ydata)

    # Create the data filenames and save the data
    data_file = options["root"] + ".data.txt"
    target_file = options["root"] + ".target.txt"

    np.savetxt(fname=data_file, X=xdata, fmt='%3.4f', delimiter='\t')
    np.savetxt(fname=target_file, X=ydata, fmt='%3.4f')

    print "data file and target file have been saved..."


if __name__ == "__main__":
    cl_options = command_line_arguments()
    generate_data(cl_options)
