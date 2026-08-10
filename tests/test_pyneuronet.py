import sys
import unittest
import os

# Add build directory to path to find pyneuronet
build_dir = os.environ.get(
    'PYNEURONET_BUILD_DIR',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build')),
)
sys.path.append(build_dir)

import pyneuronet

class TestNeuroNetBindings(unittest.TestCase):
    def test_xor_training(self):
        nn = pyneuronet.NeuroNet(2)
        nn.set_input_size(2)
        nn.resize_layer(0, 3)
        nn.resize_layer(1, 1)

        nn.set_activation_function(0, pyneuronet.ActivationFunctionType.ReLU)
        nn.set_activation_function(1, getattr(pyneuronet.ActivationFunctionType, "None"))

        inputs = [
            pyneuronet.Matrix([[0.0, 0.0]]),
            pyneuronet.Matrix([[0.0, 1.0]]),
            pyneuronet.Matrix([[1.0, 0.0]]),
            pyneuronet.Matrix([[1.0, 1.0]]),
        ]

        targets = [
            pyneuronet.Matrix([[0.0]]),
            pyneuronet.Matrix([[1.0]]),
            pyneuronet.Matrix([[1.0]]),
            pyneuronet.Matrix([[0.0]]),
        ]

        nn.train(inputs, targets, 0.1, 100)

        nn.set_input(inputs[1])
        out = nn.get_output()
        self.assertEqual(out.rows(), 1)
        self.assertEqual(out.cols(), 1)
        self.assertTrue(out.get(0, 0) > 0.0)

if __name__ == '__main__':
    unittest.main()
