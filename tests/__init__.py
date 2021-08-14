import pytest
import torch

require_cuda = pytest.mark.skipif(lambda: torch.has_cuda(), reason="GPU required")
