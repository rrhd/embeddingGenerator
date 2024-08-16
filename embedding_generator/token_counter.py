import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression


class TokenCounter:
    def __init__(self, model: SentenceTransformer, confidence_threshold: float = 0.95) -> None:
        """
        Initializes the TokenCounter with a given SentenceTransformer model and confidence threshold.

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
        """
        Calculates the token count for a given text using the model's tokenizer.
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
        """
        Trains a linear regression model to predict token counts based on text lengths.
        If the model's R-squared value exceeds the confidence threshold, switches to using
        the approximation for future token counts.
        """
        X = np.array(self.text_lengths)
        y = np.array(self.token_counts)
        self.regression_model = LinearRegression().fit(X, y)
        r_squared = self.regression_model.score(X, y)

        if r_squared > self.confidence_threshold:
            self.use_approximation = True

    def _approximate_token_count(self, text: str) -> int:
        """
        Approximates the token count for a given text using the trained regression model.

        Args:
            text: The input text for which the token count is to be approximated.

        Returns:
            The approximated number of tokens in the input text.
        """
        return int(self.regression_model.predict([[len(text)]])[0])