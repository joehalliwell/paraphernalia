import pytest
import torch

from paraphernalia import running_in_github_action

skipif_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required"
)

skipif_github_action = pytest.mark.skipif(
    not running_in_github_action(),
    reason="Not running as a Github action",
)
