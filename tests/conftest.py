import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def cpu_only():
    """Reuse the CPU smoke tests safely on a machine with running GPU jobs."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            patch.setattr(torch.backends.mps, "is_available", lambda: False)
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            yield
        finally:
            torch.set_num_threads(threads)
