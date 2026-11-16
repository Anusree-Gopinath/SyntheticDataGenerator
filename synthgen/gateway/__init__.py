"""The LLM gateway: the only module allowed to call a language model."""

from synthgen.gateway.gateway import GatewayError, LLMGateway, LLMRequest

__all__ = ["GatewayError", "LLMGateway", "LLMRequest"]
