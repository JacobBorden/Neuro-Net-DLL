# Python dense binding reference

The `pyneuronet` extension exposes dense networks and float matrices. This reference
covers the binding at master `a40b27d`; C++ features outside the table are not
Python APIs. See the [binding implementation](../src/python_bindings/pyneuronet.cpp)
and [dense declarations](../src/neural_network/neuronet.h) for argument types.

## Matrix

| Call | Behavior |
| --- | --- |
| `Matrix(rows, cols)` | Construct a float matrix. Use nonnegative dimensions. |
| `Matrix(nested_rows)` | Copy a rectangular nested sequence into float storage. |
| `rows()`, `cols()` | Return dimensions. |
| `get(row, col)`, `set(row, col, value)` | Read or write one zero-based element. |
| `to_list()` | Return a copied nested list. Editing it does not edit the matrix. |

Validate rectangular rows and index bounds in Python before calling the extension.
The nested-sequence constructor takes its width from the first row and indexes
subsequent rows without checking their lengths. The element access wrappers also
index directly; Python-style IndexError protection is not supplied by the binding.
Use finite, representable float values for numerical workflows.

```python
import pyneuronet as nn

matrix = nn.Matrix([[1.0, 2.0], [3.0, 4.0]])
assert (matrix.rows(), matrix.cols()) == (2, 2)
assert matrix.get(1, 0) == 3.0
matrix.set(0, 1, -5.0)
copy = matrix.to_list()
copy[0][0] = 99.0
assert matrix.get(0, 0) == 1.0
assert matrix.to_list() == [[1.0, -5.0], [3.0, 4.0]]
linear = getattr(nn.ActivationFunctionType, "None")
network = nn.NeuroNet(1)
network.set_input_size(2)
assert network.resize_layer(0, 1)
network.set_activation_function(0, linear)
assert network.get_layer_count() == 1
assert network.get_input_size() == 2
```

`None` is a Python keyword: use `getattr` for that enum member. The other exposed
members are ReLU, LeakyReLU, ELU, Softmax, Sigmoid, Tanh and Swish. Exposure alone
does not qualify every activation/loss gradient combination.

## Dense network

Arguments are positional: these bindings do not declare named `py::arg` parameters.
The names below describe argument order, not supported keyword spellings.

| Call | Corresponding C++ operation |
| --- | --- |
| `NeuroNet()`, `NeuroNet(layer_count)` | Default or layer-count constructor. |
| `resize_neuro_net(layer_count)` | `ResizeNeuroNet` |
| `set_input_size(feature_count)` | `SetInputSize` |
| `resize_layer(layer_index, neuron_count)` | `ResizeLayer`; inspect its boolean result. |
| `get_layer_count()`, `get_input_size()` | `getLayerCount`, `GetInputSize` |
| `set_activation_function(layer_index, activation)` | Select the layer activation. |
| `set_input(matrix)` | `SetInput`; inspect its boolean result. |
| `get_output()` | `GetOutput` |
| `train(inputs, targets, learning_rate, epochs)` | `Train`; inputs/targets are sequences of Matrix samples. |
| `save_model(filename)` | Save the dense model. |
| `NeuroNet.load_model(filename)` | Static loader returning a network. |

Configure layer count, input size, layer widths and activations before training.
The dense training declaration specifies 1×N input samples and corresponding
1×M targets, positive learning rate and epochs. Follow the actual
[training implementation](../src/neural_network/neuronet.cpp) for validation and
update semantics. C++ invalid_argument/runtime_error exceptions ordinarily map
through pybind11 to ValueError/RuntimeError; boolean rejection results still need
checking. Model loading has relative-path restrictions; use disposable relative
filenames for examples.

## Handle rejected calls

A rejected input returns `False`; check it before requesting a prediction.
The following continues the two-feature network above and uses rectangular,
in-bounds matrices. Invalid training arguments raise `ValueError` before any
training updates. Check the exception type rather than depending on message text.

```python
assert not network.set_input(nn.Matrix([[1.0]]))  # Wrong feature count.
assert not network.set_input(nn.Matrix([[1.0, 2.0], [3.0, 4.0]]))  # Two rows.
assert not nn.NeuroNet().set_input(nn.Matrix([[1.0, 2.0]]))  # No layers.

training_inputs = [nn.Matrix([[0.25, -0.75]])]
training_targets = [nn.Matrix([[0.5]])]
for inputs, targets, rate, epochs in [
    ([], training_targets, 0.1, 1),
    (training_inputs, [], 0.1, 1),
    (training_inputs, training_targets * 2, 0.1, 1),
    (training_inputs, training_targets, 0.0, 1),
    (training_inputs, training_targets, 0.1, 0),
]:
    try:
        network.train(inputs, targets, rate, epochs)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid training arguments were accepted")

# Supply a valid sample again before inference after a rejected call.
assert network.set_input(training_inputs[0])
```

## Save and reload predictions

This continues the configured one-layer example above without training. Model JSON
stores parameters with finite precision, so compare predictions with a tolerance.
The temporary working directory permits a disposable relative model filename and
restores the caller's directory even if a check fails.

```python
import math
import os
import tempfile
from pathlib import Path

samples = [nn.Matrix([[0.25, -0.75]]), nn.Matrix([[-1.0, 0.5]])]
predictions = []
for sample in samples:
    assert network.set_input(sample)
    predictions.append(network.get_output().get(0, 0))

previous_directory = Path.cwd()
with tempfile.TemporaryDirectory() as directory:
    try:
        os.chdir(directory)
        assert network.save_model("dense.json")
        restored = nn.NeuroNet.load_model("dense.json")
        assert restored.get_layer_count() == network.get_layer_count()
        assert restored.get_input_size() == network.get_input_size()
        for sample, expected in zip(samples, predictions):
            assert restored.set_input(sample)
            output = restored.get_output()
            assert (output.rows(), output.cols()) == (1, 1)
            assert math.isclose(output.get(0, 0), expected,
                                rel_tol=1e-5, abs_tol=1e-5)
    finally:
        os.chdir(previous_directory)
```

This checks same-build serialization and inference on two inputs. It does not
establish training quality or model compatibility across library versions.

## Limits

This binding exposes no seed setter, direct weight/bias access, loss selector,
mini-batch/validation configuration, optimizer selection, or train/eval mode.
Reproducibility cannot be inferred from calling a seed API in another library.
CNN, RNN, LSTM, transformer, genetic algorithm, vocabulary and shared Matrix
package interfaces are not exported here. There is no promise of pickle support,
a stable extension ABI, or compatibility across Python interpreter versions.

Build the extension for the interpreter that will import it. A successful import
from a build tree does not establish staged-install or release-package support.
This reference describes the current dense API; it does not add bindings or close
the broader M2/M4 training, packaging and platform acceptance gates.

The `pyneuronet_reference` CTest executes the Python examples above against the
matching build-tree extension, with assertions enabled even under Python `-O`.
