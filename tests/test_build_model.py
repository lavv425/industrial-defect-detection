import torch

from predict import build_model


def test_build_model_outputs_two_classes():
    model = build_model()
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)

    # binary good vs defective
    assert output.shape == (1, 2)


def test_classifier_head_is_replaced():
    model = build_model()
    # stock resnet50 fc is a single Linear; build_model swaps in a seq ending in a 2-unit Linear
    assert isinstance(model.fc, torch.nn.Sequential)
    assert model.fc[-1].out_features == 2
