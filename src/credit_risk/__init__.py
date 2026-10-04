"""Credit risk lab; imports never load data or serialized models."""

from importlib.metadata import version

__version__ = version("credit-risk-lab")

if not isinstance(__version__, str) or not __version__:
    raise RuntimeError("Installed package metadata is incomplete; reinstall credit-risk-lab")
