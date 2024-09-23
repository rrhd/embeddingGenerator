"""A class to estimate the token limit for embedding batches based on past successes and failures."""

import torch


class TokenLimitEstimator:
    """A class to estimate the token limit for embedding batches based on past successes and failures.

    The TokenLimitEstimator adjusts the maximum token limit for embedding batches based on the outcomes
    of past batches. It uses an Exponential Moving Average (EMA) to smooth the adjustments and adapt
    to changing conditions over time.

    Attributes:
        max_gpu_memory: The maximum available GPU memory in MB.
        current_limit: The current token limit for embedding batches.
        min_limit: The minimum allowable token limit.
        ema_alpha: The alpha value for the EMA calculation.
        ema_limit: The EMA of the token limit.
        last_outcome: The outcome of the last batch (True for success, False for OOM).
        streak: The number of consecutive successes or failures.
    """

    def __init__(self, initial_limit: int | None = None, min_limit: int = 1000) -> None:
        """Initializes the TokenLimitEstimator to adaptively adjust the maximum token limit for embedding batches based on past successes and failures.

        Args:
            initial_limit: Optional initial token limit. If None, it starts with an estimate
                           based on GPU memory or an initial test batch.
            min_limit: Minimum allowable token limit. Defaults to 1000.
        """
        self.max_gpu_memory = self.get_max_gpu_memory()  # In MB
        self.current_limit = initial_limit or self.estimate_initial_limit()
        self.min_limit = min_limit
        self.ema_alpha = 0.9  # Higher value for smoother EMA
        self.ema_limit = self.current_limit
        self.last_outcome = None
        self.streak = 0

    def __call__(self) -> int:
        """Returns the current token limit for embedding batches, based on the EMA.

        Returns:
            The current token limit as an integer.
        """
        return int(max(self.ema_limit, self.min_limit))

    def update(self, success: bool, tokens: int) -> None:
        """Updates the token limit estimator based on the outcome of the last batch.

        Args:
            success: True if the last batch was successful, False if it caused an OOM.
            tokens: The number of tokens in the last batch.
        """
        if self.last_outcome == success:
            self.streak += 1
        else:
            self.streak = 1
        self.last_outcome = success

        if success:
            # On success, try increasing the limit slightly
            increment = tokens * 0.05
            new_limit = tokens + increment
            self.ema_limit = self.ema_alpha * self.ema_limit + (1 - self.ema_alpha) * new_limit
        else:
            # On failure, reduce the limit significantly
            reduction = tokens * 0.5
            new_limit = max(tokens - reduction, self.min_limit)
            self.ema_limit = self.ema_alpha * self.ema_limit + (1 - self.ema_alpha) * new_limit
            # Clear CUDA cache to free up memory
            torch.cuda.empty_cache()

    def estimate_initial_limit(self) -> int:
        """Estimates the initial token limit based on the available GPU memory.

        Returns:
            Estimated initial token limit.
        """
        # Estimate based on GPU memory (assuming 1 token ~ 2KB in GPU memory)
        # This may need to be adjusted based on the actual model and hardware
        estimated_tokens = (
            self.max_gpu_memory * 1024
        ) / 2  # Convert MB to KB, divide by 2KB per token
        return int(max(min(estimated_tokens, 1_000_000), self.min_limit))

    def get_max_gpu_memory(self) -> int:
        """Retrieves the maximum available GPU memory in MB.

        Returns:
            Total GPU memory in MB.
        """
        if torch.cuda.is_available():
            gpu_properties = torch.cuda.get_device_properties(0)
            total_memory = gpu_properties.total_memory / (1024 * 1024)  # Convert bytes to MB
            return total_memory
        else:
            return 4000  # Default to 4GB if no GPU is available

    def add_success(self, tokens: int) -> None:
        """Updates the token limit after a successful batch.

        Args:
            tokens: The number of tokens in the successful batch.
        """
        self.update(True, tokens)

    def add_fail(self, tokens: int) -> None:
        """Updates the token limit after a failed batch due to OOM.

        Args:
            tokens: The number of tokens in the failed batch.
        """
        self.update(False, tokens)
