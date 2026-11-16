import pytest

from synthgen.gateway import GatewayError, LLMGateway


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434"])
def test_accepts_local(url):
    assert LLMGateway(url).base_url.startswith("http")


@pytest.mark.parametrize(
    "url",
    ["https://api.example.com", "http://10.0.0.5:11434", "http://localhost.evil.com:11434"],
)
def test_refuses_non_local(url):
    with pytest.raises(GatewayError):
        LLMGateway(url)
