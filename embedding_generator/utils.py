"""Utility functions for the embedding generator."""

from decimal import Decimal
from typing import Any


def convert_decimal(content: dict[str, Any]) -> dict[str, Any]:
    """Converts all Decimal types within the embeddings and average_embedding fields of the content dictionary to floats.

    Args:
        content: A dictionary containing the data, possibly including Decimal types.

    Returns:
        The updated dictionary with Decimal values converted to floats.
    """
    if "embeddings" in content:
        content["embeddings"] = [
            [float(x) if isinstance(x, Decimal) else x for x in embedding]
            if embedding
            else embedding
            for embedding in content["embeddings"]
        ]
    if content.get("average_embedding"):
        content["average_embedding"] = [
            float(x) if isinstance(x, Decimal) else x for x in content["average_embedding"]
        ]
    return content

