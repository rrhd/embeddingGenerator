"""CLI to generate embeddings for texts provided in an input file using a SentenceTransformer model."""

import csv
import json
from pathlib import Path

import click
from sentence_transformers import SentenceTransformer

from embedding_generator.generator import EmbeddingGenerator


def load_texts_from_file(file_path: Path) -> list[str]:
    """Load texts from a file based on its extension."""
    ext = file_path.suffix.lower()
    if ext == ".txt":
        with file_path.open("r", encoding="utf-8") as f:
            texts = [line.strip() for line in f.readlines()]
    elif ext == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            texts = json.load(f)
            if isinstance(texts, dict):
                texts = list(texts.values())  # Assumes the JSON is a dict of texts
    elif ext == ".csv":
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            texts = [
                row[0] for row in reader
            ]  # Assumes each row contains one text in the first column
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return texts


@click.command()
@click.option(
    "--input-file",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input file containing texts.",
    default='test.txt'
)
@click.option("--model-path", "-m", required=True, help="Path to the SentenceTransformer model.", default="Alibaba-NLP/gte-large-en-v1.5")
@click.option(
    "--output-path", "-o", default="data/embeddings", help="Path to save the embeddings."
)
@click.option(
    "--device", "-d", default="cpu", help="Device to use for embedding ('cpu' or 'cuda')."
)
def run_embedding(input_file: str, model_path: str, output_path: str, device: str) -> None:
    """CLI to generate embeddings for texts provided in an input file using a SentenceTransformer model."""
    file_path = Path(input_file)
    texts = load_texts_from_file(file_path)

    model = SentenceTransformer(model_path, trust_remote_code=True)
    model_settings = {"convert_to_tensor": True, "device": device, "show_progress_bar": False}
    EmbeddingGenerator(model, model_settings, save_path=output_path)(iter(texts))
    click.echo(f"Embeddings saved to {output_path}")


if __name__ == "__main__":
    run_embedding()
