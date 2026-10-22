import numpy as np
import pytest
import torch

from sctrange.hlut import HLUT


@pytest.fixture
def hlut() -> HLUT:
    return HLUT.pyradplan_default()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    torch.manual_seed(0)
