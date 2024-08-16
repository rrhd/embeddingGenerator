# EmbeddingGenerator Documentation

## Overview

The `EmbeddingGenerator` class is designed to efficiently generate embeddings for a list of input texts using a model such as `SentenceTransformer`. This class manages the process of splitting texts into manageable chunks, embedding them while considering token limits, and saving the results incrementally to avoid memory issues.

## Usage Example

Here is an example of how you might use the `EmbeddingGenerator`:

```python
model_settings = dict(convert_to_tensor=True, device=self.device, show_progress_bar=False) #  HuggingFace model settings
embedding_generator = EmbeddingGenerator(self.model, model_settings)
embeddings = embedding_generator(texts)
```

### What Happens During Execution

1. **Text Chunking**: The input texts are split into smaller chunks based on the model's token limit, ensuring that the text segments are manageable for the model to process.

2. **Embedding Generation**: Each chunk of text is passed through the model to generate embeddings. The embeddings for each chunk are saved incrementally to a JSON file, allowing the process to handle large datasets without exceeding memory limits.

3. **Data Saving**: As embeddings are generated, they are saved to a specified file (default `embeddings.json`). This file contains the token counts, embeddings, and other metadata for each text.

4. **Handling Large Files**: If the file containing the embeddings becomes too large, only the necessary data (e.g., average embeddings) is loaded to avoid memory overload. Fully processed texts are removed from memory to conserve resources.

5. **Loading and Returning Embeddings**: Upon completion or during reloading, the class attempts to load only the average embeddings for each text. If an embedding is missing, the method returns `None` for that text.

### Output

The output of the `EmbeddingGenerator` is a dictionary where each key is an input text, and the value is one of the following:

- **List of Floats**: The average embedding for the text, which is a list of floats representing the embedding vector.
- **`None`**: Indicates that the embedding for the text could not be generated or is missing.

### Example Output

For an input list of texts:

```python
texts = ["This is the first text.", "Here's the second text."]
```

The output might look like:

```python
{
    "This is the first text.": [0.234, -0.987, 0.123, ...],  # Embedding vector
    "Here's the second text.": None  # Embedding not available
}
```

### Notes for Users

- **File Creation**: The `EmbeddingGenerator` will produce a file named `embeddings.json` (or another specified name) in the specified directory. This file contains all the embedding data generated during the process.
  
- **Memory Efficiency**: The class is designed to handle large datasets by saving progress incrementally and reloading only necessary data.

- **Missing Data**: If any embeddings are missing or could not be generated, the output dictionary will reflect this with `None` values for those texts.

- **Restarting**: If the process is interrupted, it can be restarted, and the class will continue processing from where it left off, avoiding recomputation of already processed texts.

- **Intensive GPU Utilization**: The `EmbeddingGenerator` will attempt to utilize all available GPU memory during the embedding process, making it computationally intensive. Be prepared for high GPU usage, especially with large datasets.

- **Large Dataset Handling**: If the resulting dataset is too large to fit in memory, users might need to manually load and manage the data from the `embeddings.json` file. The resulting file can also be very large, so ensure sufficient disk space is available.

## Running Embedding Generation from an Input File

The `run_embedding.py` script can now accept an input file containing texts in various formats. The script supports plain text files, JSON files, and CSV files.

### Supported Input File Formats

- **Plain Text (`.txt`)**: Each line in the file is treated as a separate text for embedding.
- **JSON (`.json`)**: The file can be a list or a dictionary of texts. If it's a dictionary, the values are treated as the texts to be embedded.
- **CSV (`.csv`)**: The script assumes that each row contains one text in the first column.

### Example Usage

To run the embedding generator with an input file:

```bash
python scripts/run_embedding.py --input-file "path/to/texts.json" --model-path "path/to/model" --output-path "data/embeddings.json" --device "cuda"
```

### Command-Line Options

- **`--input-file` or `-i`**: Path to the input file containing texts. Supported formats: `.txt`, `.json`, `.csv`.
- **`--model-path` or `-m`**: Path to the SentenceTransformer model.
- **`--output-path` or `-o`**: Path where the embeddings will be saved. Defaults to `data/embeddings.json`.
- **`--device` or `-d`**: Device to use for embedding. Options are `'cpu'` or `'cuda'`. Defaults to `'cpu'`.

### Notes

- The input file format is detected based on its extension, and the texts are processed accordingly.
- The CLI will generate embeddings for the provided texts and save them to the specified output file.
- Ensure that the input file is properly formatted according to the chosen extension.
- The resulting data file might be very large, so ensure that there is sufficient disk space available.
- If the dataset is too large to fit in memory, you might need to load and handle the data manually from the `embeddings.json` file.
- The embedding process is computationally intensive and will attempt to utilize all available GPU space.


## Conclusion

The `EmbeddingGenerator` provides a robust and efficient way to generate embeddings for large datasets, handling memory constraints and ensuring that the output reflects all input texts, even if some embeddings are missing. This makes it a versatile tool for embedding tasks where scalability and resource management are critical.
