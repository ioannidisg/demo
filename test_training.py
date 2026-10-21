from config import batch_size
from train import epochs, learning_rate


def test_training_parameters():
    assert epochs > 0
    assert learning_rate > 0
    assert batch_size > 0