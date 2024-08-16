

class TokenLimitEstimator:
    def __init__(self, initial_limit: int = 1000, learning_rate: float = 0.1, max_limit: int = 250000,
                 min_limit: int = 1000) -> None:
        """
        Initializes the TokenLimitEstimator to dynamically adjust the maximum token limit for embedding
        batches based on success and failure rates.

        Args:
            initial_limit: The starting token limit for embedding batches. Defaults to 1000.
            learning_rate: The rate at which the token limit is adjusted after successes or failures.
                           Defaults to 0.1.
            max_limit: The maximum allowable token limit for embedding batches. Defaults to 250000.
            min_limit: The minimum allowable token limit for embedding batches. Defaults to 1000.

        """
        self.current_limit = initial_limit
        self.learning_rate = learning_rate
        self.max_limit = max_limit
        self.min_limit = min_limit
        self.success_streak = 0
        self.failure_streak = 0
        self.convergence_threshold = 100  # Number of consecutive successes to consider converged


    def __call__(self) -> int:
        """
        Returns the current token limit for embedding batches.

        Returns:
            The current token limit as an integer.
        """
        return self.current_limit

    def update(self, success: bool, tokens: int) -> None:
        """
        Updates the current token limit based on the success or failure of the last embedding attempt.

        Args:
            success: A boolean indicating whether the last embedding attempt was successful (True) or failed due to OOM (False).
            tokens: The number of tokens used in the last embedding attempt.

        Updates:
            Adjusts the `current_limit` based on the outcome. If successful, the limit may increase gradually
            until the convergence threshold is reached. If failed, the limit decreases more aggressively
            depending on the number of consecutive failures.
        """
        if success:
            self.success_streak += 1
            self.failure_streak = 0

            if self.success_streak < self.convergence_threshold:
                new_limit = max(self.current_limit + self.learning_rate * (self.max_limit - tokens), self.current_limit)
                if new_limit > self.current_limit:
                    self.current_limit = new_limit
        else:
            self.success_streak = 0
            self.failure_streak += 1

            decrease_factor = 0.9 ** self.failure_streak
            self.current_limit *= decrease_factor

        self.current_limit = int(max(min(self.current_limit, self.max_limit), self.min_limit))

    def add_success(self, tokens: int) -> None:
        """
        Updates the token limit after a successful embedding attempt.

        Args:
            tokens: The number of tokens used in the successful embedding attempt.
        """
        self.update(True, tokens)

    def add_fail(self, tokens: int) -> None:
        """
        Updates the token limit after a failed embedding attempt due to an out-of-memory (OOM) error.

        Args:
            tokens: The number of tokens used in the failed embedding attempt.
        """
        self.update(False, tokens)