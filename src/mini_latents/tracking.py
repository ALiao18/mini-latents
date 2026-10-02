from dataclasses import dataclass


@dataclass
class FitFlags:
    """Status flag from model fit.

    Params
    ------
    converged: bool, default False
        True if log likelihood converged before reaching max_iters
    decreasing_ll: bool, default False
        True if log likelihood decreased during training
    """

    converged: bool = False
    decreasing_ll: bool = False

    def display(self) -> None:
        """
        print the fit flags
        """
        print(f"converged: {self.converged}")
        print(f"decreasing ll: {self.decreasing_ll}")
