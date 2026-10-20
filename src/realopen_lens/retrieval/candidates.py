"""Bounded, explicit candidate budgets independent of the displayed result count."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CandidateDepths:
    visual: int = 50
    semantic: int = 50
    lexical: int = 50

    def __post_init__(self) -> None:
        if any(type(value) is not int or not 1 <= value <= 1000 for value in asdict(self).values()):
            raise ValueError("Each candidate depth must be an integer from 1 to 1000")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)
