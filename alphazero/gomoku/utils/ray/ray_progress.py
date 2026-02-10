import ray
from tqdm import tqdm


@ray.remote(num_cpus=0)
class ProgressBarActor:
    """Ray Actor to manage and display a tqdm progress bar from multiple workers."""

    def __init__(self, total: int, description: str):
        self.tqdm = tqdm(total=total, desc=description)
        self.count = 0

    def update(self, n: int = 1):
        """Update the progress bar by n."""
        self.tqdm.update(n)
        self.count += n

    def get_counter(self) -> int:
        """Return the current count."""
        return self.count

    def close(self):
        """Close the progress bar."""
        self.tqdm.close()
