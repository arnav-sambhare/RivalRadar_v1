"""Signal sources.

Each source is an isolated module exposing ``collect(config, rivals, context)
-> SignalBundle``, where ``context`` holds what earlier sources collected this
run. A source only fills the fields it owns (publications, patents, or
affiliations). Sources must fail soft: any error is caught by
``safe_collect`` and degrades to an empty contribution rather than breaking the
run.
"""

from .base import SourceError, safe_collect

__all__ = ["SourceError", "safe_collect"]
