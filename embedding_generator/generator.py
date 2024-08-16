import shutil
from pathlib import Path

import ijson
import numpy as np
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from orjson import orjson
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from embedding_generator.limit_estimator import TokenLimitEstimator
from embedding_generator.token_counter import TokenCounter
from embedding_generator.utils import convert_decimal


class EmbeddingGenerator:

    def __init__(self, model: SentenceTransformer, model_settings: dict, save_path: str = 'data') -> None:
        """
        Initializes the EmbeddingGenerator for managing and processing text embeddings with specified model settings.

        Args:
            model: The SentenceTransformer model used for generating embeddings.
            model_settings: A dictionary of settings to configure the model during the embedding process.
            save_path: The directory path where embeddings will be saved as a JSON file. Defaults to 'data'.

        Attributes:
            changed_keys (list[str]): List of keys that have been modified and need to be saved.
            chunk_iterations (int): Counter for the number of chunk iterations.
            knapsack_iterations (int): Counter for the number of knapsack iterations.
            chunk_solution_attempts (int): Counter for the number of attempts to find a chunk solution.
            computational_iterations (int): Counter for the number of computational iterations.
            chunk_proposals (Optional[List[Dict]]): Proposals for chunking text to fit within token limits.
            token_tolerance (int): Tolerance level for the difference between the token count and max tokens per batch.
            model_settings (dict): Configuration settings for the SentenceTransformer model.
            current_chunks (list[dict]): The current list of text chunks being processed for embedding.
            model (SentenceTransformer): The SentenceTransformer model used for embedding generation.
            max_tokens_per_batch (int): Maximum number of tokens allowed per batch.
            current_total_tokens (int): The current total number of tokens being processed.
            texts (Optional[dict]): A dictionary to hold the text and corresponding embeddings.
            token_counter (TokenCounter): Instance of TokenCounter for estimating token counts.
            text_splitter (RecursiveCharacterTextSplitter): Instance for splitting texts into manageable chunks.
            save_path (Path): Path where the embeddings JSON file will be saved.
            limit_model (TokenLimitEstimator): Instance of TokenLimitEstimator for managing token limits.
        """
        self.changed_keys: list[str] = []
        self.chunk_iterations = 0
        self.knapsack_iterations = 0
        self.chunk_solution_attempts = 0
        self.computational_iterations = 0
        self.chunk_proposals = None
        self.token_tolerance = 3000
        self.model_settings = model_settings
        self.current_chunks: list[dict] = []
        self.model = model
        self.max_tokens_per_batch = self.model.max_seq_length * 10
        self.current_total_tokens = 0
        self.texts: dict = {}
        self.token_counter = TokenCounter(self.model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.model.max_seq_length,
            chunk_overlap=20,
            length_function=self.token_counter
        )
        self.save_path = Path(save_path) / "embeddings.json"
        self.limit_model = TokenLimitEstimator(self.max_tokens_per_batch)


    def __call__(self, texts: list[str]) -> dict[str, list[float] | None]:
        """
        Generates embeddings for a list of input texts, managing chunking, token limits, and saving progress.

        Args:
            texts: A list of strings, each representing a text to be embedded.

        Returns:
            A dictionary where each key is a text, and the value is another dictionary containing the finished average embedding for the chunks of that text
        """
        self.texts = {
            text: {
                'token_counts': [],
                'embeddings': [],
                'chunks': [],
                'finished': [],
                'average_embedding': None
            } for text in texts if len(text) > 0}
        total_texts = len(self.texts)
        self.load_data()
        self.progress_bar = tqdm(total=total_texts, desc="Generating Embeddings", initial=total_texts - len(self.texts))
        while self.text_remains():
            if self.current_chunks:
                self.embed()
                if self.computational_iterations % 100 == 0:
                    self.save_data()
                completed_texts = self.get_completed_groups()
                # If the progress bar is already initialized, update it
                self.progress_bar.n = total_texts + len(completed_texts) - len(self.texts)
                self.progress_bar.refresh()
        self.save_data()
        return self.load_average_embeddings_with_fallback(texts)

    def embed(self) -> None:
        """
        Processes and embeds the current chunks of text. If an out-of-memory (OOM) error occurs,
        the token limit is adjusted accordingly.

        Raises:
            RuntimeError: If an OOM error occurs, it adjusts the token limit based on the number of tokens
                          in the current chunks and logs the failure.
        """
        try:
            self.embed_chunks()
            self.current_chunks = []
            self.computational_iterations += 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                tokens = sum(chunk["length"] for chunk in self.current_chunks)
                self.fail(tokens)

    def text_remains(self) -> bool:
        """
        Checks if there are any unfinished texts that still need to be processed and sets up chunks for them.

        Returns:
            A boolean indicating whether there are any texts remaining that require further processing.
        """
        unfinished_texts = [text for text, values in self.texts.items()
                            if len(text) > 0 and (not values['finished'] or not all(values['finished']))]
        if unfinished_texts:
            self.set_chunks()
            return True
        return False

    def set_chunks(self) -> None:
        """
        Determines the best chunks for processing based on the current token limits. If the chunk iterations
        exceed the limit, assigns a single unfinished or unchunked text for processing.
        """
        self.chunk_iterations = 0
        while self.find_best_chunks() and self.chunk_iterations < 5:
            self.update_chunks()
            self.chunk_iterations += 1

        # If we exceed the chunk iterations limit, assign a single unfinished or unchunked text
        if self.chunk_iterations >= 5 and not self.current_chunks:
            text = self.get_next_incomplete_or_unchunked_text()
            if text:
                # If the text hasn't been chunked yet, chunk it
                if not self.texts[text]['chunks']:
                    self.chunk_text(text)

                # Assign the first available chunk to current_chunks
                self.current_chunks = [{
                    'length': self.texts[text]['token_counts'][0],
                    'chunk': self.texts[text]['chunks'][0],
                    'group_name': text
                }]

    def embed_chunks(self) -> None:
        """
        Embeds the current chunks of text using the model and updates the chunks with their respective embeddings.
        Also handles cleanup by clearing GPU cache and removes finished chunks from the current processing list.
        """
        with torch.no_grad():
            tensor = self.model.encode([chunk["chunk"] for chunk in self.current_chunks], **self.model_settings)
            embeddings = tensor.detach().tolist()
            del tensor
            torch.cuda.empty_cache()

        for embedding, chunk in zip(embeddings, self.current_chunks):
            chunk['embedding'] = embedding

        self.remove_finished_chunks()
        tokens = sum(chunk["length"] for chunk in self.current_chunks)
        self.succeed(tokens)

    def fail(self, tokens: int) -> None:
        """
        Handles a failed embedding attempt by updating the token limit model based on the number of tokens
        in the failed attempt and adjusts the maximum tokens per batch accordingly.

        Args:
            tokens: The number of tokens used in the failed embedding attempt.
        """
        self.limit_model.add_fail(tokens)
        self.max_tokens_per_batch = self.limit_model()

    def succeed(self, tokens: int) -> None:
        """
        Handles a successful embedding attempt by updating the token limit model based on the number of tokens
        used in the successful attempt and adjusts the maximum tokens per batch accordingly.

        Args:
            tokens: The number of tokens used in the successful embedding attempt.
        """
        self.limit_model.add_success(tokens)
        self.max_tokens_per_batch = self.limit_model()


    def chunk_text(self, text: str) -> None:
        """
        Splits the given text into manageable chunks based on the model's token limits and initializes
        the corresponding data structures for storing token counts, embeddings, and completion status.

        Args:
            text: The input text to be chunked and processed for embeddings.
        """
        chunks = self.chunker(text)
        self.texts[text]['chunks'] = chunks
        self.texts[text]['token_counts'] = self.get_token_counts(chunks)
        self.texts[text]['embeddings'] = [None] * len(chunks)
        self.texts[text]['finished'] = [False] * len(chunks)

    def chunker(self, text: str) -> list[str]:
        """
        Splits the input text into smaller chunks that fit within the model's maximum sequence length.
        If the text is shorter than the maximum length, it is returned as a single chunk.

        Args:
            text: The input text to be split into smaller chunks.

        Returns:
            A list of strings where each string is a chunk of the original text.
        """
        self.computational_iterations += 1
        if self.count_tokens(text) < self.model.max_seq_length:
            return [text]
        return self.text_splitter.split_text(text)

    def get_token_counts(self, chunks: list[str]) -> list[int]:
        """
        Calculates the token count for each chunk of text.

        Args:
            chunks: A list of text chunks for which token counts need to be calculated.

        Returns:
            A list of integers where each integer represents the token count for a corresponding chunk of text.
        """
        return [self.count_tokens(chunk) for chunk in chunks]

    def count_tokens(self, chunk: str) -> int:
        """
        Counts the number of tokens in a given chunk of text using the TokenCounter.

        Args:
            chunk: A string representing a chunk of text for which the token count is needed.

        Returns:
            The number of tokens in the given chunk of text.
        """
        return self.token_counter(chunk)

    def remove_finished_chunks(self) -> None:
        """
        Updates the embeddings and marks chunks as finished for all completed chunks in the current processing list.
        It also updates the list of changed keys and recalculates the average embeddings for all groups.
        """
        self.changed_keys = []
        for info in self.current_chunks:
            chunk = info['chunk']
            group_name = info['group_name']
            embedding = info['embedding']

            if group_name in self.texts:
                self.changed_keys.append(group_name)
                group = self.texts[group_name]

                # Find the index of the chunk in the group
                if chunk in group['chunks']:
                    index = group['chunks'].index(chunk)
                    group['embeddings'][index] = embedding
                    group['finished'][index] = True

        self.calculate_all_average_embeddings()

    def get_next_incomplete_or_unchunked_text(self) -> str | None:
        """
        Retrieves the next text that either has unfinished chunks or has not been chunked yet.

        Returns:
            The text to be processed next, or None if all texts are already processed.
        """
        for text, data in self.texts.items():
            # Check if the text is already chunked and has unfinished chunks
            if data['chunks'] and not all(data['finished']):
                return text
            # Check if the text hasn't been chunked yet
            if not data['chunks']:
                return text
        return None  # Return None if all texts are processed


    def get_incomplete_chunks(self) -> dict[str, dict[str, list]]:
        """
        Retrieves all incomplete chunks across all text groups.

        Returns:
            A dictionary where the key is the group name, and the value is another dictionary containing
            the unfinished chunks, their token counts, embeddings, and their finished status.
        """
        incomplete_chunks = {}
        for group_name, group_data in self.texts.items():
            if 'chunks' in group_data and 'finished' in group_data:
                unfinished_indices = [i for i, finished in enumerate(group_data['finished']) if not finished]
                if unfinished_indices:
                    incomplete_chunks[group_name] = {
                        'chunks': [group_data['chunks'][i] for i in unfinished_indices],
                        'token_counts': [group_data['token_counts'][i] for i in unfinished_indices] if 'token_counts' in group_data else [],
                        'embeddings': [group_data['embeddings'][i] for i in unfinished_indices],
                        'finished': [False] * len(unfinished_indices),
                    }
        return incomplete_chunks

    def get_chunked_text(self) -> dict[str, dict[str, list]]:
        """
        Retrieves texts that have been chunked and still have incomplete chunks.

        Returns:
            A dictionary where the key is the text, and the value is another dictionary containing
            the chunks, token counts, embeddings, and their finished status for each chunk.
        """
        return {text: values for text, values in self.get_incomplete_chunks().items() if len(values.get('chunks', [])) > 0}

    def preprocess_items(self, items: list[tuple[int, str, str]], threshold: float = 0.01) -> list[tuple[int, str, str]]:
        """
        Filters out items that are too small to significantly impact the solution based on a threshold.

        Args:
            items: A list of tuples, where each tuple contains the token count, the chunk, and the group name.
            threshold: A float representing the minimum proportion of the max tokens per batch for an item to be retained. Defaults to 0.01.

        Returns:
            A list of tuples where each item has a token count greater than the calculated threshold.
        """
        # Calculate the threshold value based on the max tokens per batch
        threshold_value = self.max_tokens_per_batch * threshold
        return [item for item in items if item[0] > threshold_value]

    def find_best_chunks(self, max_iterations: int = 1000) -> bool:
        """
        Finds the best chunks of text for embedding by maximizing token usage while staying within the token limit.

        Args:
            max_iterations: The maximum number of iterations for refining the chunk selection. Defaults to 1000.

        Returns:
            A boolean indicating whether a suitable set of chunks was found.
        """
        # Gather and preprocess text chunks
        items = self.preprocess_items([
            (length, chunk, group_name)
            for group_name, group_data in self.get_chunked_text().items()
            for length, chunk in zip(group_data['token_counts'], group_data['chunks'])
        ])
        if not items:
            items = [
                (length, chunk, group_name)
                for group_name, group_data in self.get_chunked_text().items()
                for length, chunk in zip(group_data['token_counts'], group_data['chunks'])
            ]

        if not items:
            self.update_chunks()
            return False  # No items to process

        # Sort items by length in descending order for potentially better packing
        items.sort(key=lambda x: x[0], reverse=True)
        weights = np.array([item[0] for item in items], dtype=np.int32)
        dp = np.zeros(self.max_tokens_per_batch + 1)

        iteration_count = 0
        for weight in weights:
            if weight <= self.max_tokens_per_batch:
                # Store the old dp array to detect changes
                old_dp = dp.copy()
                dp[weight:] = np.maximum(dp[weight:], dp[:-weight] + weight)
                # Increment iteration_count and break if reached max_iterations
                iteration_count += 1
                if iteration_count >= max_iterations:
                    break  # Stop updating after reaching max_iterations
                # Early exit if there have been no changes
                if np.array_equal(old_dp, dp):
                    break  # Stop updating if no change is detected

        # Check if the solution is within the acceptable tolerance
        max_weight = np.max(dp)
        if self.max_tokens_per_batch - max_weight < self.token_tolerance:
            return self._process_solution(dp, items)

        return self._process_solution(dp, items)

    def greedy_solution(self, items: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
        """
        Implements a greedy algorithm to select a set of chunks that maximizes token usage while staying
        within the token limit for the current batch.

        Args:
            items: A list of tuples where each tuple contains the token count, the chunk, and the group name.

        Returns:
            A list of tuples representing the selected chunks that fit within the token limit.
        """
        current_total = 0
        solution = []
        for length, chunk, group_name in items:
            if current_total + length <= self.max_tokens_per_batch:
                solution.append((length, chunk, group_name))
                current_total += length
            if self.max_tokens_per_batch - current_total < self.token_tolerance:
                break
        return solution

    def _process_solution(self, dp, items):
        # Reconstruct the solution from the DP array
        solution = []
        j = np.argmax(dp)  # Start from the maximum value in the dp array
        for weight, chunk, group_name in reversed(items):
            if j >= weight and dp[j] == dp[j - weight] + weight:
                solution.append((weight, chunk, group_name))
                j -= weight

        # Prepare the list of current chunks to be processed
        self.current_chunks = [{
            'length': length,
            'chunk': chunk,
            'group_name': group_name
        } for length, chunk, group_name in solution]

        if self.max_tokens_per_batch - sum(item[0] for item in solution) < self.token_tolerance:
            self.chunk_solution_attempts = 0
            return False  # Early stop if close to the tolerance limit
        self.chunk_solution_attempts += 1
        return True

    def update_chunks(self) -> None:
        """
        Updates the list of chunks by checking if any text has not yet been chunked. If a text with no chunks
        is found, it is chunked and added to the processing list.
        """
        for text, values in self.texts.items():
            if len(values.get('chunks', [])) == 0:
                self.chunk_text(text)
                break

    def get_completed_groups(self) -> dict[str, dict[str, list]]:
        """
        Retrieves all groups of text that have completed the embedding process, where all chunks are finished.

        Returns:
            A dictionary where the key is the group name, and the value is another dictionary containing
            the group's data, including chunks, embeddings, and their finished status.
        """
        completed_groups = {}
        for group_name, group_data in self.texts.items():
            if len(group_data['chunks']) > 0:
                if all(group_data['finished']):
                    completed_groups[group_name] = group_data
        return completed_groups

    def calculate_average_embedding(self, group_name: str) -> None:
        """
        Calculates and assigns the weighted average embedding for a group of text chunks if all chunks
        in the group are finished.

        Args:
            group_name: The name of the group for which the average embedding should be calculated.
        """
        group = self.texts[group_name]
        if not all(group['finished']):
            return  # Skip if not all embeddings are finished

        embeddings = np.array(group['embeddings'])
        token_counts = np.array(group['token_counts'])

        # Calculate weighted average using NumPy
        weighted_sum = np.dot(embeddings.T, token_counts)
        total_tokens = token_counts.sum()
        average_embedding = weighted_sum / total_tokens

        # Assign the average embedding
        group['average_embedding'] = average_embedding.tolist()

    def calculate_all_average_embeddings(self) -> None:
        """
        Calculates and assigns the weighted average embedding for all groups of text chunks
        that have completed the embedding process.
        """
        completed_groups = self.get_completed_groups()
        for group_name in completed_groups:
            self.calculate_average_embedding(group_name)


    def save_data(self) -> None:
        """
        Incrementally saves the embedding data to a JSON file, ensuring that large files are handled efficiently.
        The method writes to a temporary file and then renames it to avoid data corruption.

        Updates:
            - Converts Decimal types to floats.
            - For complete entries, only saves the average embedding.
            - Writes the key-value pairs to a temporary file, handling updates for changed keys.
            - Renames the temporary file to the original file after writing.
            - Removes completed entries from self.texts after saving.
        """
        file_path = self.save_path
        temp_path = file_path.with_suffix('.tmp')
        keys_to_remove = set()

        with file_path.open('rb') as f, open(temp_path, 'wb') as temp_file:
            # Write opening bracket for JSON object
            temp_file.write(b'{\n')

            # Read the existing JSON file incrementally
            parser = ijson.kvitems(f, '')
            first_entry = True  # To handle commas between JSON objects

            for text, content in parser:
                # Convert Decimal types to floats
                content = convert_decimal(content)
                if content['average_embedding']:
                    content = {
                        'average_embedding': content['average_embedding']
                    }
                if text in self.changed_keys:
                    new_content = self.texts[text]

                    # Check if all 'finished' values are True
                    if 'finished' in new_content and all(new_content['finished']) and len(content['finished']) > 0:
                        keys_to_remove.add(text)

                    content = new_content
                    self.changed_keys.remove(text)

                # Write the key-value pair to the temp file
                if not first_entry:
                    temp_file.write(b',\n')
                json_bytes = orjson.dumps({text: content})
                temp_file.write(json_bytes[1:-1])  # Strip the outer braces
                first_entry = False

            # Write closing bracket for JSON object
            temp_file.write(b'\n}')

        # Rename the temporary file to the original file
        shutil.move(temp_path, self.save_path)

        # Remove completed entries from self.texts
        for key in keys_to_remove:
            del self.texts[key]

    def load_data(self) -> None:
        """
        Loads existing embedding data from a JSON file, updating the internal state with the loaded data.
        Completed texts that do not require further processing are skipped.

        Updates:
            - Ensures embeddings are lists of floats by converting Decimal types.
            - Merges loaded data with the existing internal state, updating token counts, embeddings, chunks,
              finished status, and average embeddings.
        """
        file_path = self.save_path

        if file_path.exists():
            try:
                with file_path.open('rb') as f:
                    parser = ijson.kvitems(f, '')

                    for text, content in parser:
                        # Check if all 'finished' values are True
                        if ('finished' in content and all(content['finished']) and len(content['finished']) > 0) or (list(content.keys()) == ['average_embedding']):
                            # Skip loading this text as it is already completed
                            del self.texts[text]
                            continue

                        # Ensure embeddings are lists of floats
                        content = convert_decimal(content)

                        # Update or add entries to self.texts
                        if text in self.texts:
                            self.texts[text].update({
                                'token_counts': content.get('token_counts', []),
                                'embeddings': content.get('embeddings', []),
                                'chunks': content.get('chunks', []),
                                'finished': content.get('finished', []),
                                'average_embedding': content.get('average_embedding', None)
                            })
                        else:
                            self.texts[text] = {
                                'token_counts': content.get('token_counts', []),
                                'embeddings': content.get('embeddings', []),
                                'chunks': content.get('chunks', []),
                                'finished': content.get('finished', []),
                                'average_embedding': content.get('average_embedding', None)
                            }
            except Exception as e:
                print(f'Error processing JSON file: {e}')

    def load_average_embeddings_with_fallback(self, texts: list[str]) -> dict[str, list[float] | None]:
        """
        Loads only the average embeddings for each text key from the saved file. If an embedding is missing,
        it sets the value to None.

        Args:
            texts: A list of text keys for which the average embeddings are needed.

        Returns:
            A dictionary where the key is the text identifier, and the value is the average embedding (a list of floats)
            or None if the embedding is missing.
        """
        average_embeddings = {text: None for text in texts}
        file_path = self.save_path

        if file_path.exists():
            try:
                with file_path.open('rb') as f:
                    parser = ijson.kvitems(f, '')

                    for text, content in parser:
                        if text in average_embeddings and 'average_embedding' in content and content[
                            'average_embedding']:
                            average_embeddings[text] = content['average_embedding']

            except Exception as e:
                print(f'Error processing JSON file: {e}')

        return average_embeddings