import pytest
import torch

from paraphernalia import running_in_github_action

skipif_no_cuda = pytest.mark.skipif(lambda: not torch.has_cuda(), reason="GPU required")

skipif_github_action = pytest.mark.skipif(
    lambda: not running_in_github_action(),
    reason="Not running as a Github action",
)
