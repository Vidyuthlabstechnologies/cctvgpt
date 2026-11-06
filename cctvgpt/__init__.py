"""CCTVGPT: prompt-based synthetic CCTV-style video generation with built-in safeguards.

All outputs are synthetic, visibly labelled, invisibly watermarked and carry
signed provenance. See README.md and docs/ for the mapping to the PRD.
"""

from .config import PACKAGE_VERSION as __version__

__all__ = ["__version__"]
