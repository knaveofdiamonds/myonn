import numpy
import scipy.special

class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inputnodes = inputnodes
        self.outputnodes = outputnodes
        self.hiddennodes = hiddennodes
        self.learningrate = learningrate

        self.weights_input_hidden = numpy.random.normal(
            0.0,
            pow(self.hiddennodes, -0.5),
            (self.hiddennodes, self.inputnodes)
        )

        self.weights_hidden_output = numpy.random.normal(
            0.0,
            pow(self.outputnodes, -0.5),
            (self.outputnodes, self.hiddennodes)
        )

        self.activation_function = scipy.special.expit

    def train(self, input_list, target_list):
        targets = numpy.array(target_list, ndmin=2).T
        inputs = numpy.array(input_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.weights_hidden_output.T, output_errors)

        # Update the weights
        self.weights_hidden_output = (
            self.learningrate *
            numpy.dot(
                (output_errors * final_outputs * (1.0 - final_outputs)),
                numpy.transpose(hidden_outputs)
            )
        )

        self.weights_input_hidden = (
            self.learningrate *
            numpy.dot(
                (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                numpy.transpose(inputs)
            )
        )

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


n = NeuralNetwork(3,3,3,0.3)

print( n.query([1.0, 0.5, -1.5]) )
