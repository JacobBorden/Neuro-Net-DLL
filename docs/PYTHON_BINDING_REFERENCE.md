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
in-bounds matrices. The argument checks below raise `ValueError` before any training updates. Check the exception type rather than depending on message text.

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

### A failed training call can leave updates applied

Training validates each sample's shape as it reaches that sample. A shape error
raises `RuntimeError`, and updates from earlier samples remain applied. If you need
to avoid partial updates from shape errors, validate all input and target shapes
against the configured network before calling `train`.

This separate linear network uses a zero input to isolate the bias. The first
target is one unit above its starting prediction, so a learning rate of 0.25
increases that prediction by 0.25 before the second input is rejected.

```python
import math

partial = nn.NeuroNet(1)
partial.set_input_size(2)
assert partial.resize_layer(0, 1)
partial.set_activation_function(0, linear)

zero = nn.Matrix([[0.0, 0.0]])
assert partial.set_input(zero)
initial = partial.get_output().get(0, 0)
assert math.isfinite(initial)
target_matrix = nn.Matrix([[initial + 1.0]])
wrong_width = nn.Matrix([[0.0]])

try:
    partial.train([zero, wrong_width], [target_matrix, target_matrix], 0.25, 1)
except RuntimeError:
    pass
else:
    raise AssertionError("Training accepted an input with the wrong width")

assert partial.set_input(zero)
updated = partial.get_output().get(0, 0)
assert math.isclose(updated, initial + 0.25, rel_tol=0.0, abs_tol=1e-5), (initial, updated)
```

### Check all sample shapes before training

Use the configured input width and final layer width to check every sample before
calling `train`. Here the same network takes two features and produces one output.
The check validates the shape of existing Matrix samples. Check that nested lists
are rectangular before constructing those matrices.

```python
def check_sample_shapes(samples, width, name):
    for index, sample in enumerate(samples):
        if (sample.rows(), sample.cols()) != (1, width):
            raise ValueError(f"{name} sample {index} must have shape 1 × {width}")


def train_checked(inputs, targets):
    check_sample_shapes(inputs, partial.get_input_size(), "Input")
    check_sample_shapes(targets, 1, "Target")
    partial.train(inputs, targets, 0.25, 1)


valid_target = nn.Matrix([[updated + 1.0]])
for inputs, targets in [
    ([zero, wrong_width], [valid_target, valid_target]),
    ([zero, nn.Matrix([[0.0, 0.0], [0.0, 0.0]])], [valid_target, valid_target]),
    ([zero, zero], [valid_target, nn.Matrix([[0.0, 0.0]])]),
    ([zero, zero], [valid_target, nn.Matrix([[0.0], [0.0]])]),
]:
    try:
        train_checked(inputs, targets)
    except ValueError:
        pass
    else:
        raise AssertionError("Shape check accepted a malformed sample")

    input_accepted = partial.set_input(zero)
    assert input_accepted
    assert partial.get_output().get(0, 0) == updated

train_checked([zero], [valid_target])
input_accepted = partial.set_input(zero)
assert input_accepted
assert math.isclose(partial.get_output().get(0, 0), updated + 0.25,
                    rel_tol=0.0, abs_tol=1e-5)
```

## Train a single sample

This tiny linear example checks the training call and error reduction. A nonzero input
exercises both weight updates and the bias update. The target is one unit above the initial prediction, so
the check does not depend on a particular random initialization or seed API.
For a linear prediction `w·x + b` and half squared error, one update changes
the residual by the factor `1 - learning_rate * (1 + x·x)`: the weight
contribution is `x·x` and the bias contribution is one. With `x = [0.5, -1]`
and rate 0.1, five updates leave `0.775 ** 5` of the initial residual. Check that recurrence with float tolerance as well as error reduction.
The measured objective here is half squared error for one scalar output; this does
not qualify other activation/loss combinations or generalization quality.

```python
import math

sample = nn.Matrix([[0.5, -1.0]])
assert network.set_input(sample)
before = network.get_output().get(0, 0)
assert math.isfinite(before)
target = before + 1.0
initial_error = 0.5 * (before - target) ** 2
network.train([sample], [nn.Matrix([[target]])], 0.1, 5)
assert network.set_input(sample)
after = network.get_output().get(0, 0)
assert math.isfinite(after)
final_error = 0.5 * (after - target) ** 2
assert final_error < initial_error, (initial_error, final_error)
expected_after = target + (before - target) * (1.0 - 0.1 * (1.0 + 0.5 ** 2 + (-1.0) ** 2)) ** 5
assert math.isclose(after, expected_after, rel_tol=1e-5, abs_tol=1e-5), (after, expected_after)
```

The next example saves this trained network and checks its predictions after load.

## Save and reload predictions

This continues the trained one-layer example above. Model JSON
stores parameters with finite precision, so compare predictions with a tolerance.
The temporary working directory permits a disposable relative model filename and
restores the caller's directory even if a check fails.

```python
import math
import os
import tempfile
from pathlib import Path

# Three affine-independent inputs distinguish both weights and the bias.
# The zero input catches coupled parameter changes hidden by the other two.
samples = [nn.Matrix([[0.25, -0.75]]), nn.Matrix([[-1.0, 0.5]]),
           nn.Matrix([[0.0, 0.0]])]
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
        # Resume training on the loaded parameters using the same linear contract.
        zero_sample = nn.Matrix([[0.0, 0.0]])
        assert restored.set_input(zero_sample)
        resumed_before = restored.get_output().get(0, 0)
        resumed_target = resumed_before - 1.0
        restored.train([zero_sample], [nn.Matrix([[resumed_target]])], 0.2, 3)
        assert restored.set_input(zero_sample)
        resumed_after = restored.get_output().get(0, 0)
        expected_resumed = resumed_target + (resumed_before - resumed_target) * 0.8 ** 3
        assert math.isfinite(resumed_after)
        assert math.isclose(resumed_after, expected_resumed,
                            rel_tol=1e-5, abs_tol=1e-5), (resumed_after, expected_resumed)
    finally:
        os.chdir(previous_directory)
```

This checks same-build serialization and inference of the trained model on three affine-independent inputs,
then verifies three additional bias updates after loading. This continuation check
covers this linear single-sample case only. It does not
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
