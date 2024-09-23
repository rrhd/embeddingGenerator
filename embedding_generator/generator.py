"""A module for generating embeddings for a batch of texts using a SentenceTransformer model.

The EmbeddingGenerator processes texts in batches to manage memory usage efficiently.
It chunks texts into manageable sizes based on token limits and processes them in parallel.
The embeddings are saved to disk along with the average embeddings for each text.
"""

import hashlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from orjson import orjson
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .limit_estimator import TokenLimitEstimator
from .token_counter import TokenCounter
from .utils import convert_decimal


class EmbeddingGenerator:
    """A class for generating embeddings for a batch of texts using a SentenceTransformer model.

    The EmbeddingGenerator processes texts in batches to manage memory usage efficiently.
    It chunks texts into manageable sizes based on token limits and processes them in parallel.
    The embeddings are saved to disk along with the average embeddings for each text.

    Attributes:
        model: The SentenceTransformer model used for embedding generation.
        model_settings: Configuration settings for the SentenceTransformer model.
        save_path: Path where the embeddings will be saved.
        max_memory_usage: Maximum allowable memory usage for self.texts.
        limit_estimator: Instance of TokenLimitEstimator for managing token limits.
        token_counter: Instance of TokenCounter for estimating token counts.
        text_splitter: Instance for splitting texts into manageable chunks.
        texts: Dictionary to hold current texts and their processing data.
        text_generator: Iterator for input texts.
        current_chunks: List of current chunks being processed.
        progress_bar: TQDM progress bar for tracking progress.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        model_settings: dict,
        save_path: str = "data",
        max_memory_usage: int | None = None,
    ) -> None:
        """Initializes the EmbeddingGenerator for managing and processing text embeddings with specified model settings.

        Args:
            model: The SentenceTransformer model used for generating embeddings.
            model_settings: A dictionary of settings to configure the model during the embedding process.
            save_path: The directory path where embeddings will be saved. Defaults to 'data'.
            max_memory_usage: Optional maximum memory usage in bytes for self.texts. If None, estimated dynamically.

        Attributes:
            model: The SentenceTransformer model used for embedding generation.
            model_settings: Configuration settings for the SentenceTransformer model.
            save_path: Path where the embeddings will be saved.
            max_memory_usage: Maximum allowable memory usage for self.texts.
            limit_estimator: Instance of TokenLimitEstimator for managing token limits.
            token_counter: Instance of TokenCounter for estimating token counts.
            text_splitter: Instance for splitting texts into manageable chunks.
            texts: Dictionary to hold current texts and their processing data.
            text_generator: Iterator for input texts.
            current_chunks: List of current chunks being processed.
            progress_bar: TQDM progress bar for tracking progress.
        """
        self.model = model
        self.model_settings = model_settings
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.save_path / "embeddings_index.json"
        self.data_path = self.save_path / "embeddings_data"
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.limit_estimator = TokenLimitEstimator(initial_limit=model.max_seq_length * 10)
        self.token_counter = TokenCounter(self.model)
        print(f"Original model max_seq_length: {self.model.max_seq_length}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.model.max_seq_length,
            chunk_overlap=20,
            length_function=self.token_counter,
        )
        self.texts = {}
        self.text_generator: Iterator[str] = iter([])
        self.current_chunks: list[dict] = []
        self.progress_bar = None
        self.max_memory_usage = max_memory_usage or self.estimate_max_memory_usage()

    def __call__(self, texts: Iterator[str]) -> dict[str, list[float] | None]:
        """Processes embeddings in batches to manage memory usage.

        Args:
            texts: An iterator or generator of input texts.

        Returns:
            A dictionary where keys are texts, and values are their average embeddings.
        """
        self.text_generator = texts
        self.load_index()
        self.progress_bar = tqdm(desc="Generating Embeddings")

        while True:
            self.fill_texts()
            if not self.texts:
                break  # No more texts to process
            self.process_current_texts()

        # After processing all texts
        return self.load_average_embeddings_with_fallback()

    def fill_texts(self) -> None:
        """Dynamically loads texts to fill embedding batches efficiently without exceeding memory limits."""
        while self.needs_more_texts():
            try:
                text = next(self.text_generator)
                text = text.strip()
                if len(text) == 0:
                    continue  # Skip empty texts
                text_id = self.generate_text_id(text)
                if (
                        text_id in self.index_data
                        and self.index_data[text_id]["status"] == "completed"
                ):
                    continue  # Skip already processed texts
                self.estimate_tokens_in_text(text)
                self.texts[text_id] = {
                    "id": text_id,
                    "text": text,
                    "token_counts": [],
                    "embeddings": [],
                    "chunks": [],
                    "finished": [],
                    "average_embedding": None,
                }
                print(f"Added text: {text}, id: {text_id}")  # Debug statement
                if self.current_memory_usage() > self.max_memory_usage:
                    print("Max memory usage reached.")  # Debug statement
                    break  # Prevent exceeding memory usage limit
            except StopIteration:
                break  # No more texts to fetch

    def process_current_texts(self) -> None:
        """Processes the current batch of texts in self.texts."""
        total_texts = len(self.texts)
        self.progress_bar.total = total_texts
        self.load_data()

        while self.text_remains():
            if self.current_chunks:
                self.embed()
                self.save_data()
                self.remove_completed_texts()
            else:
                self.set_chunks()

        self.save_data()
        self.remove_completed_texts()

    def estimate_max_memory_usage(self) -> int:
        """Estimates the maximum memory usage for self.texts based on available system memory.

        Returns:
            Estimated maximum memory usage in bytes.
        """
        import psutil

        total_memory = psutil.virtual_memory().total
        max_usage = total_memory * 0.8  # Use up to 80% of total memory
        return int(max_usage)

    def needs_more_texts(self) -> bool:
        """Determines if more texts are needed to fill batches.

        Returns:
            True if more texts are needed, False otherwise.
        """
        total_tokens_in_incomplete_chunks = self.estimate_total_tokens_in_incomplete_chunks()
        tokens_needed = self.limit_estimator() - total_tokens_in_incomplete_chunks
        return tokens_needed > 0

    def current_memory_usage(self) -> int:
        """Estimates current memory usage of self.texts.

        Returns:
            Estimated memory usage in bytes.
        """
        # Estimate based on number of texts and average size
        average_text_size = self.estimate_average_text_size()
        return len(self.texts) * average_text_size

    def estimate_average_text_size(self) -> int:
        """Estimates the average size in bytes of a text in self.texts.

        Returns:
            Average text size in bytes.
        """
        # Use a sample of texts to estimate
        sample_texts = list(self.texts.keys())[:10]
        if sample_texts:
            average_size = sum(len(text.encode("utf-8")) for text in sample_texts) / len(
                sample_texts
            )
        else:
            average_size = 1000  # Default to 1KB if no data
        return int(average_size)

    def estimate_tokens_in_text(self, text: str) -> int:
        """Estimates the number of tokens in a text using the TokenCounter.

        Args:
            text: The input text.

        Returns:
            Estimated number of tokens.
        """
        return self.token_counter(text)

    def estimate_total_tokens_in_incomplete_chunks(self) -> int:
        """Estimates the total number of tokens in incomplete chunks.

        Returns:
            Total tokens in incomplete chunks.
        """
        total_tokens = 0
        for data in self.texts.values():
            if "token_counts" in data and "finished" in data:
                for count, finished in zip(data["token_counts"], data["finished"], strict=False):
                    if not finished:
                        total_tokens += count
            else:
                # If the text hasn't been chunked yet, estimate its token count
                total_tokens += self.estimate_tokens_in_text("".join(data.get("chunks", [])) or "")
        return total_tokens

    def remove_completed_texts(self) -> None:
        """Removes fully processed texts from self.texts to free up memory."""
        keys_to_remove = []
        for text_id, data in self.texts.items():
            if data.get("average_embedding") is not None:
                keys_to_remove.append(text_id)
                self.progress_bar.update(1)
                print(f"Removed completed text_id: {text_id}")  # Debug statement
        for text_id in keys_to_remove:
            del self.texts[text_id]

    def text_remains(self) -> bool:
        """Checks if there are any unfinished texts that still need to be processed and sets up chunks for them.

        Returns:
            True if there are texts remaining that require further processing, False otherwise.
        """
        unfinished_texts = [
            text
            for text, values in self.texts.items()
            if (not values["finished"] or not all(values["finished"]))
        ]
        return bool(unfinished_texts)

    def set_chunks(self) -> None:
        """Determines the best chunks for processing based on the current token limits."""
        self.find_best_chunks()

    def embed(self) -> None:
        """Processes and embeds the current chunks of text. If an out-of-memory (OOM) error occurs, the token limit is adjusted accordingly."""
        try:
            self.embed_chunks()
            tokens = sum(chunk["length"] for chunk in self.current_chunks)
            self.succeed(tokens)
            self.current_chunks = []
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                tokens = sum(chunk["length"] for chunk in self.current_chunks)
                self.fail(tokens)
                # Clear current chunks to prevent retrying the same failing batch
                self.current_chunks = []
                torch.cuda.empty_cache()

    def embed_chunks(self) -> None:
        """Embeds the current chunks of text using the model and updates the chunks with their respective embeddings."""
        with torch.no_grad():
            tensor = self.model.encode(
                [chunk["chunk"] for chunk in self.current_chunks], **self.model_settings
            )
            embeddings = tensor.detach().cpu().tolist()
            del tensor
            torch.cuda.empty_cache()

        for embedding, chunk in zip(embeddings, self.current_chunks, strict=False):
            chunk["embedding"] = embedding

        self.remove_finished_chunks()

    def succeed(self, tokens: int) -> None:
        """Handles a successful embedding attempt by updating the token limit model.

        Args:
            tokens: The number of tokens used in the successful attempt.
        """
        self.limit_estimator.add_success(tokens)

    def fail(self, tokens: int) -> None:
        """Handles a failed embedding attempt by updating the token limit model.

        Args:
            tokens: The number of tokens used in the failed attempt.
        """
        self.limit_estimator.add_fail(tokens)

    def remove_finished_chunks(self) -> None:
        """Updates the embeddings and marks chunks as finished for all completed chunks in the current processing list."""
        for info in self.current_chunks:
            chunk = info["chunk"]
            text_id = info["group_name"]
            embedding = info["embedding"]
            chunk_index = info["chunk_index"]

            print(f"Processing chunk {chunk_index} for text_id: {text_id}")  # Debug statement
            if text_id in self.texts:
                data = self.texts[text_id]
                if 0 <= chunk_index < len(data["chunks"]):
                    data["embeddings"][chunk_index] = embedding
                    data["finished"][chunk_index] = True
                    print(f"Marked chunk {chunk_index} as finished for text_id: {text_id}")  # Debug statement
                else:
                    print(f"Invalid chunk index {chunk_index} for text_id: {text_id}")  # Debug statement
            else:
                print(f"text_id {text_id} not found in self.texts")  # Debug statement

        self.calculate_all_average_embeddings()

    def find_best_chunks(self) -> None:
        """Selects the best chunks to process using an efficient greedy approach with numpy."""
        # Gather incomplete chunks
        incomplete_chunks = []
        for text_id, data in self.texts.items():
            for idx, (length, chunk, finished) in enumerate(zip(
                    data["token_counts"], data["chunks"], data["finished"], strict=False
            )):
                if not finished:
                    incomplete_chunks.append({
                        "length": length,
                        "chunk": chunk,
                        "text_id": text_id,
                        "chunk_index": idx,
                    })
        print(f"Incomplete chunks: {incomplete_chunks}")  # Debug statement

        # If no incomplete chunks, chunk all texts without chunks
        if not incomplete_chunks:
            for text_id, data in self.texts.items():
                if not data["chunks"]:
                    self.chunk_text(text_id)
            # Update incomplete_chunks after chunking all necessary texts
            incomplete_chunks = []
            for text_id, data in self.texts.items():
                for idx, (length, chunk, finished) in enumerate(zip(
                        data["token_counts"], data["chunks"], data["finished"], strict=False
                )):
                    if not finished:
                        incomplete_chunks.append({
                            "length": length,
                            "chunk": chunk,
                            "text_id": text_id,
                            "chunk_index": idx,
                        })
            print(f"Incomplete chunks after chunking: {incomplete_chunks}")  # Debug statement

        # Proceed with processing incomplete_chunks as before
        if not incomplete_chunks:
            print("No chunks to process.")  # Debug statement
            return  # No chunks to process

        # Extract arrays for sorting and selection
        lengths = np.array([item['length'] for item in incomplete_chunks], dtype=np.int32)
        chunks = np.array([item['chunk'] for item in incomplete_chunks])
        group_names = np.array([item['text_id'] for item in incomplete_chunks])
        chunk_indices = np.array([item['chunk_index'] for item in incomplete_chunks], dtype=np.int32)

        # Sort indices by lengths in descending order
        sorted_indices = np.argsort(-lengths)
        sorted_lengths = lengths[sorted_indices]
        sorted_chunks = chunks[sorted_indices]
        sorted_group_names = group_names[sorted_indices]
        sorted_chunk_indices = chunk_indices[sorted_indices]

        # Compute cumulative sum of token counts
        cumulative_lengths = np.cumsum(sorted_lengths)
        max_tokens_per_batch = self.limit_estimator()
        # Find the indices where cumulative length is within the limit
        within_limit = cumulative_lengths <= max_tokens_per_batch

        if not np.any(within_limit):
            # No chunks can fit within the limit; process the largest available chunk
            selected_indices = np.array([sorted_indices[0]])
        else:
            selected_indices = sorted_indices[within_limit]

        # Prepare the current chunks
        self.current_chunks = [
            {
                "length": int(sorted_lengths[i]),
                "chunk": sorted_chunks[i],
                "group_name": sorted_group_names[i],
                "chunk_index": int(sorted_chunk_indices[i]),
            }
            for i in selected_indices
        ]
        print(f"Selected current_chunks: {self.current_chunks}")  # Debug statement

    def chunk_text(self, text_id: str) -> None:
        """Splits the given text into manageable chunks based on the model's token limits.

        Args:
            text_id: The unique identifier for the text to be chunked and processed for embeddings.
        """
        print(f"Chunking text_id: {text_id}")  # Debug statement
        data = self.texts[text_id]
        text = data["text"]
        chunks = self.text_splitter.split_text(text)
        print(f"Chunks for text_id {text_id}: {chunks}")  # Debug statement
        token_counts = [self.token_counter(chunk) for chunk in chunks]
        print(f"Token counts for text_id {text_id}: {token_counts}")  # Debug statement
        data["chunks"] = chunks
        data["token_counts"] = token_counts
        data["embeddings"] = [None] * len(chunks)
        data["finished"] = [False] * len(chunks)

    def calculate_all_average_embeddings(self) -> None:
        """Calculates and assigns the weighted average embedding for all groups of text chunks that have completed the embedding process.

        The average embedding is calculated based on the weighted sum of embeddings and token counts.
        """
        for data in self.texts.values():
            if data["average_embedding"] is None and all(data["finished"]):
                embeddings = np.array(data["embeddings"], dtype=np.float32)
                token_counts = np.array(data["token_counts"], dtype=np.float32)
                # Calculate weighted average
                weighted_embeddings = embeddings * token_counts[:, np.newaxis]
                total_tokens = token_counts.sum()
                average_embedding = weighted_embeddings.sum(axis=0) / total_tokens
                data["average_embedding"] = average_embedding.tolist()

    def load_data(self) -> None:
        """Loads data for texts currently in self.texts."""
        for text_id, data in self.texts.items():
            data_file = self.data_path / f"{text_id}.json"
            if data_file.exists():
                with data_file.open("rb") as f:
                    content = orjson.loads(f.read())
                    content = convert_decimal(content)
                    data.update(content)

    def save_data(self) -> None:
        """Saves embeddings data per text to individual files and updates the index."""
        for text_id, data in self.texts.items():
            data_to_save = {
                "token_counts": data["token_counts"],
                "embeddings": data["embeddings"],
                "chunks": data["chunks"],
                "finished": data["finished"],
                "average_embedding": data["average_embedding"],
            }
            data_file = self.data_path / f"{text_id}.json"
            # Save data to individual file
            with data_file.open("wb") as f:
                json_bytes = orjson.dumps(data_to_save)
                f.write(json_bytes)
            # Update index
            self.index_data[text_id] = {
                "status": "completed" if data.get("average_embedding") else "in_progress",
                "text": data["text"],
            }
        # Save the updated index
        with self.index_path.open("wb") as f:
            json_bytes = orjson.dumps(self.index_data)
            f.write(json_bytes)

    def load_index(self) -> None:
        """Loads the embeddings index from disk."""
        self.index_data = {}
        if self.index_path.exists():
            with self.index_path.open("rb") as f:
                self.index_data = orjson.loads(f.read())

    def generate_text_id(self, text: str) -> str:
        """Generates a unique identifier for a text.

        Args:
            text: The input text.

        Returns:
            A unique identifier string for the text.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def load_average_embeddings_with_fallback(self) -> dict[str, list[float] | None]:
        """Loads only the average embeddings for each text key from the index.

        Returns:
            A dictionary where the key is the text, and the value is the average embedding or None.
        """
        average_embeddings = {}
        self.load_index()
        for text_id, entry in self.index_data.items():
            if entry["status"] == "completed":
                data_file = self.data_path / f"{text_id}.json"
                if data_file.exists():
                    with data_file.open("rb") as f:
                        data = orjson.loads(f.read())
                        average_embeddings[entry["text"]] = data.get("average_embedding")
                else:
                    average_embeddings[entry["text"]] = None
            else:
                average_embeddings[entry["text"]] = None
        return average_embeddings
