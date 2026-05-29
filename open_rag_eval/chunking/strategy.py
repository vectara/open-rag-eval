"""Chunking strategy descriptors for the chunking-comparison layer.

A :class:`ChunkingStrategy` is a small, library-agnostic descriptor of how
documents should be split into chunks before indexing. Local-document
connectors (LangChain, LlamaIndex) translate a strategy into their own native
splitter, so this module deliberately avoids importing any specific chunking
library.
"""

from dataclasses import dataclass
from typing import Any, List


@dataclass
class ChunkingStrategy:
    """A named chunking configuration.

    Attributes:
        name: Human-readable identifier used in output filenames and plot labels.
        chunk_size: Target chunk size (characters for LangChain's recursive
            splitter, tokens for LlamaIndex's sentence splitter).
        chunk_overlap: Overlap between consecutive chunks.
        splitter: Extension hook selecting the splitter family. Currently only
            "recursive" is implemented; connectors map it to their native splitter.
    """

    name: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    splitter: str = "recursive"


def parse_chunking_strategies(chunking_cfg: Any) -> List[ChunkingStrategy]:
    """Parse the ``chunking`` config block into a list of strategies.

    Args:
        chunking_cfg: The ``chunking`` mapping from the eval config. May be an
            OmegaConf node or a plain dict. Must contain a non-empty
            ``strategies`` list.

    Returns:
        A list of validated :class:`ChunkingStrategy` instances.

    Raises:
        ValueError: If no strategies are provided, a strategy is missing a name,
            strategy names are not unique, or ``chunk_overlap >= chunk_size``.
    """
    if chunking_cfg is None:
        raise ValueError("Chunking config is empty.")

    # Support both dict-like configs (OmegaConf DictConfig, dict) uniformly.
    strategies_cfg = (
        chunking_cfg.get("strategies")
        if hasattr(chunking_cfg, "get")
        else getattr(chunking_cfg, "strategies", None)
    )
    if not strategies_cfg:
        raise ValueError(
            "Chunking config must contain a non-empty 'strategies' list."
        )

    strategies: List[ChunkingStrategy] = []
    seen_names = set()
    for idx, item in enumerate(strategies_cfg):
        getter = item.get if hasattr(item, "get") else (lambda k, d=None: getattr(item, k, d))

        name = getter("name", None)
        if not name:
            raise ValueError(f"Chunking strategy at index {idx} is missing a 'name'.")
        name = str(name)
        if name in seen_names:
            raise ValueError(f"Duplicate chunking strategy name: '{name}'.")
        seen_names.add(name)

        chunk_size = int(getter("chunk_size", 1000))
        chunk_overlap = int(getter("chunk_overlap", 200))
        splitter = str(getter("splitter", "recursive"))

        if chunk_size <= 0:
            raise ValueError(
                f"Chunking strategy '{name}' has non-positive chunk_size: {chunk_size}."
            )
        if chunk_overlap < 0:
            raise ValueError(
                f"Chunking strategy '{name}' has negative chunk_overlap: {chunk_overlap}."
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"Chunking strategy '{name}' has chunk_overlap ({chunk_overlap}) "
                f">= chunk_size ({chunk_size})."
            )

        strategies.append(
            ChunkingStrategy(
                name=name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                splitter=splitter,
            )
        )

    return strategies
