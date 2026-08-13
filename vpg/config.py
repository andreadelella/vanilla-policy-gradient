"""Shared configuration parsing helpers."""


def parse_hidden_sizes(value: str) -> list[int]:
    """Parse a comma-separated hidden-layer specification such as ``32,32``."""

    try:
        sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise ValueError(
            "hidden_sizes must contain positive integers, e.g. '32,32'"
        ) from error

    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(
            "hidden_sizes must contain positive integers, e.g. '32,32'"
        )
    return sizes
