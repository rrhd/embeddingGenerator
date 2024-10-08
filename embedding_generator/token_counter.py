"""A module to count tokens in text using a SentenceTransformer model."""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression


class TokenCounter:
    """A class to count tokens in text using a SentenceTransformer model.

    The TokenCounter uses the SentenceTransformer model's tokenizer to count the number of tokens in text.
    If the model's R-squared value exceeds a given confidence threshold, it switches to using a linear regression model
    to approximate token counts based on text lengths.

    Attributes:
        model: The SentenceTransformer model that contains the tokenizer to be used for counting tokens.
        confidence_threshold: The R-squared value at which the model should switch to using the regression model for approximation.
        text_lengths: A list of lists containing the lengths of texts for training the regression model.
        token_counts: A list of token counts for training the regression model.
        regression_model: A linear regression model for approximating token counts based on text lengths.
        use_approximation: A boolean indicating whether to use the regression model for token count approximation.
    """

    def __init__(self, model: SentenceTransformer, confidence_threshold: float = 0.95) -> None:
        """Initializes the TokenCounter with a given SentenceTransformer model and confidence threshold.

        Args:
            model: The SentenceTransformer model that contains the tokenizer to be used for counting tokens.
            confidence_threshold: The R-squared value at which the model should switch to using the regression
                                  model for approximation. Defaults to 0.95.
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.text_lengths: list[list[int]] = []
        self.token_counts: list[int] = []
        self.regression_model: LinearRegression | None = None
        self.use_approximation = False

    def __call__(self, text: str) -> int:
        """Calculates the token count for a given text using the model's tokenizer.

        If the regression model has been trained and exceeds the confidence threshold,
        it uses the approximate token count instead.

        Args:
            text: The input text for which the token count is to be calculated.

        Returns:
            The number of tokens in the input text.
        """
        if not self.use_approximation:
            token_count = len(self.model.tokenizer.encode(text))
            self.text_lengths.append([len(text)])
            self.token_counts.append(token_count)

            if len(self.text_lengths) % 100 == 0:
                self._train_regression_model()

            return token_count
        else:
            return self._approximate_token_count(text)

    def _train_regression_model(self) -> None:
        """Trains a linear regression model to predict token counts based on text lengths.

        If the model's R-squared value exceeds the confidence threshold, switches to using
        the approximation for future token counts.
        """
        x = np.array(self.text_lengths)
        y = np.array(self.token_counts)
        self.regression_model = LinearRegression().fit(x, y)
        r_squared = self.regression_model.score(x, y)

        if r_squared > self.confidence_threshold:
            self.use_approximation = True

    def _approximate_token_count(self, text: str) -> int:
        """Approximates the token count for a given text using the trained regression model.

        Args:
            text: The input text for which the token count is to be approximated.

        Returns:
            The approximated number of tokens in the input text.
        """
        return int(self.regression_model.predict([[len(text)]])[0])
