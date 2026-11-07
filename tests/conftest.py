import pytest

from cctvgpt import release_gate
from cctvgpt.config import Settings
from cctvgpt.context import TERMS_VERSION, Platform

# Small, fast clips for tests.
FAST = {"width": 320, "height": 240, "fps": 6, "duration_s": 2.0}


def make_platform(tmp_path, **overrides) -> Platform:
    s = Settings(data_dir=tmp_path / "data")
    for k, v in overrides.items():
        setattr(s, k, v)
    return Platform.open(s)


@pytest.fixture(scope="session")
def selftest_result(tmp_path_factory):
    p = make_platform(tmp_path_factory.mktemp("selftest"))
    return release_gate.selftest(p.keys)


@pytest.fixture
def platform(tmp_path, selftest_result):
    p = make_platform(tmp_path)
    release_gate.decide(p.store, *release_gate.renderer_component(), "approved", "owner", "test", selftest_result)
    return p


@pytest.fixture
def customer(platform):
    c = platform.store.create_customer("Acme Vision", "video_analytics", "ops@acme.example")
    platform.store.set_kyc(c["id"], "verified", "GSTIN-TEST", "admin")
    platform.store.accept_terms(c["id"], TERMS_VERSION, "user")
    return platform.store.customer(c["id"])
