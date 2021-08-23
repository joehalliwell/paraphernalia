import os

import pytest
import torch

skipif_no_cuda = pytest.mark.skipif(lambda: not torch.has_cuda(), reason="GPU required")

skipif_github_action = pytest.mark.skipif(
    lambda: not os.environ.get("GITHUB_ACTIONS", False),
    reason="Not running as a Github action",
)
