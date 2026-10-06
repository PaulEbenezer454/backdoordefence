"""Smoke tests for the centralized MNIST baseline components."""
import torch
from models.cnn import MNISTCNN, extract_parameters, layer_parameters, parameter_differences, restore_parameters


def test_model_output_shape_and_parameter_helpers():
    model = MNISTCNN()
    output = model(torch.zeros(2, 1, 28, 28))
    assert output.shape == (2, 10)
    params = extract_parameters(model)
    assert {"conv1", "conv2", "fc1", "output"}.issubset(layer_parameters(model))
    assert all(torch.equal(value, parameter.detach()) for (name, parameter), value in zip(model.named_parameters(), params.values()))
    changed = {name: value + 1 for name, value in params.items()}
    deltas = parameter_differences(changed, params)
    assert all(torch.allclose(delta, torch.ones_like(delta)) for delta in deltas.values())
    restore_parameters(model, params)
    assert all(torch.equal(value, extract_parameters(model)[name]) for name, value in params.items())

"""Smoke test both architecture input/output shapes."""
import torch
from models.cnn import create_model


def test_cifar_model_and_named_layers():
    model = create_model("cifar10_cnn")
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    assert {"conv1", "conv2", "conv3", "fc1", "output"}.issubset(dict(model.named_modules()))
