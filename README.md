# `EmbeddingGenerator` Documentation

## Overview

The `EmbeddingGenerator` class is designed to efficiently generate embeddings for a list of input texts using a model such as `SentenceTransformer`. It manages the process of splitting texts into manageable chunks, embedding them while considering token limits, and saving the results incrementally to avoid memory issues. The class is optimized to handle large datasets by processing texts in batches and managing resources effectively.

---

## Usage Example

Here's how you might use the `EmbeddingGenerator`:

```python
from sentence_transformers import SentenceTransformer

# Initialize your model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your model

# Define model settings
model_settings = {
    'convert_to_tensor': True,
    'device': 'cuda',  # or 'cpu'
    'show_progress_bar': False
}

# Create an instance of EmbeddingGenerator
embedding_generator = EmbeddingGenerator(model, model_settings, save_path='data')

# Prepare your texts as an iterator
texts = iter([
    "This is the first text.",
    "Here's the second text.",
    # ... add more texts
])

# Generate embeddings
embeddings = embedding_generator(texts)

# Output embeddings
print(embeddings)
```

---

## What Happens During Execution

1. **Initialization**:
   - The `EmbeddingGenerator` is initialized with a model, model settings, and a save path.
   - It sets up internal structures for managing texts, embeddings, and progress tracking.

2. **Text Loading and Memory Management**:
   - Texts are loaded from the provided iterator using the `fill_texts` method.
   - The class dynamically loads texts while monitoring memory usage to prevent exceeding `max_memory_usage`.

3. **Text Chunking**:
   - Texts are split into chunks using `RecursiveCharacterTextSplitter` based on the model's `max_seq_length`.
   - The splitter ensures chunks are appropriately sized for the model to process efficiently.

4. **Token Counting**:
   - The `TokenCounter` estimates the number of tokens in each chunk.
   - This information is used to manage batch sizes and ensure they fit within token limits.

5. **Batch Selection**:
   - The `find_best_chunks` method selects chunks to process in the next batch, maximizing batch sizes without exceeding limits.
   - Chunks are sorted and selected based on their token counts.

6. **Embedding Generation**:
   - The `embed` method processes the selected chunks using the model.
   - Embeddings are generated and associated with their respective chunks.

7. **Error Handling and Token Limit Adjustment**:
   - If a `RuntimeError` occurs (e.g., out-of-memory error), the `fail` method adjusts the token limit to prevent future errors.
   - Successful batches inform the `succeed` method to update the token limit estimator positively.

8. **Saving Progress**:
   - Embeddings and metadata are saved incrementally using the `save_data` method.
   - Data is saved per text to individual files to avoid loading large JSON files entirely.

9. **Resource Cleanup**:
   - Completed texts are removed from memory using the `remove_completed_texts` method.
   - This ensures efficient memory usage throughout the process.

10. **Final Output Generation**:
    - Upon completion, `load_average_embeddings_with_fallback` is called to compile the average embeddings for each text.
    - The output is a dictionary mapping each text to its average embedding or `None` if unavailable.

---

## Output

The output of the `EmbeddingGenerator` is a dictionary where each key is an input text, and the value is one of the following:

- **List of Floats**: The average embedding for the text, represented as a list of floats.
- **`None`**: Indicates that the embedding for the text could not be generated or is missing.

### Example Output

```python
{
    "This is the first text.": [0.234, -0.987, 0.123, ...],  # Embedding vector
    "Here's the second text.": [0.456, -0.654, 0.789, ...]   # Embedding vector
}
```

---

## Notes for Users

- **File Structure**:
  - The `EmbeddingGenerator` saves data in a structured directory:
    ```
    data/
    ├── embeddings_index.json
    └── embeddings_data/
        ├── <text_id1>.json
        └── <text_id2>.json
    ```
  - Each text's data is saved in a separate JSON file, preventing the need to load large files into memory.

- **Memory Efficiency**:
  - Designed to handle large datasets by managing memory usage and saving progress incrementally.
  - Texts are removed from memory once processed to conserve resources.

- **Resumable Processing**:
  - If the process is interrupted, it can be resumed, and the class will continue from where it left off, avoiding recomputation.

- **GPU Utilization**:
  - Attempts to maximize GPU utilization by processing large batches without exceeding memory limits.
  - Adjusts batch sizes dynamically based on successful and failed attempts.

- **Error Handling**:
  - Handles out-of-memory errors gracefully by adjusting token limits and retrying with smaller batches.

- **Missing Data**:
  - If any embeddings are missing, the output dictionary will contain `None` for those texts.

---

## Advanced Usage and Customization

### Adjusting Chunk Size

- By default, the chunk size is set based on the model's `max_seq_length`.
- You can customize the chunk size if needed:
  ```python
  embedding_generator.text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1024,  # Desired chunk size
      chunk_overlap=20,
      length_function=embedding_generator.token_counter,
  )
  ```

### Handling Large Batches

- Increase the initial token limit to allow larger batches:
  ```python
  embedding_generator.limit_estimator = TokenLimitEstimator(initial_limit=2048)
  ```

- Adjust the model settings to change the batch size:
  ```python
  model_settings = {
      'convert_to_tensor': True,
      'device': 'cuda',
      'show_progress_bar': False,
      'batch_size': 64  # Adjust as per your GPU capacity
  }
  ```

### Monitoring Progress

- The class uses `tqdm` to display a progress bar during processing.
- You can access or customize it via `embedding_generator.progress_bar`.

## Running Embedding Generation from an Input File

The `run_embedding.py` script accepts an input file containing texts in various formats.

### Supported Input File Formats

- **Plain Text (`.txt`)**: Each line is treated as a separate text.
- **JSON (`.json`)**: The file can contain a list or dictionary of texts.
- **CSV (`.csv`)**: Each row's first column is treated as a text.

### Example Usage

```bash
python scripts/run_embedding.py --input-file "path/to/texts.json" --model-path "path/to/model" --save-path "data" --device "cuda"
```

### Command-Line Options

- **`--input-file` or `-i`**: Path to the input file.
- **`--model-path` or `-m`**: Path to the SentenceTransformer model.
- **`--save-path` or `-o`**: Directory where embeddings will be saved. Defaults to `data`.
- **`--device` or `-d`**: Device to use (`'cpu'` or `'cuda'`). Defaults to `'cpu'`.

### Notes

- Ensure the input file is properly formatted according to its extension.
- The embeddings are saved incrementally in the specified save path.
- The script handles large datasets efficiently, but ensure sufficient disk space is available.

---

## Conclusion

The `EmbeddingGenerator` is a robust tool for generating embeddings for large datasets, designed with efficiency and scalability in mind. By managing resources effectively, handling errors gracefully, and providing mechanisms for customization, it ensures that embedding generation tasks can be performed reliably, even with extensive datasets.
