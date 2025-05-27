from dataclasses import dataclass
from typing import Any


@dataclass
class AverageMeter:
    """Computes and stores the average and current value.

    Attributes:
        name: Name of the meter for identification
        fmt: Format string for printing values
    """
    name: str
    fmt: str = ':f'

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update statistics with new value.

        Args:
            val: Value to update with
            n: Number of items this value represents (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self) -> str:
        """String representation showing current value and average."""
        fmtstr = '{name}: {val' + self.fmt + '} (avg: {avg' + self.fmt + '})'
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg)

    def state_dict(self) -> dict[str, Any]:
        """Get the current state as a dictionary."""
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from a dictionary.

        Args:
            state_dict: Dictionary containing meter state
        """
        self.val = state_dict['val']
        self.sum = state_dict['sum']
        self.count = state_dict['count']
        self.avg = state_dict['avg']
