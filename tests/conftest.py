import pytest

from phost import Impl, ImplKind, Registry
from phost.stages import newton_linear as nl
from phost.testing import runtimes


@pytest.fixture
def registry():
    reg = Registry()
    nl.register(reg)
    reg.add_impl(Impl(id="test/reference", stage=nl.NAME, stage_version=1, version="1", title="reference",
                      kind=ImplKind.BUILTIN, factory=lambda host, params: runtimes.ReferenceRuntime(host, params)))
    return reg
