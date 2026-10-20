from dataclasses import dataclass


@dataclass
class FitFlags:
    """Status flag from model fit.

    Params
    ------
    converged: bool, default False
        True if log likelihood converged before reaching max_iter
    decreasing_ll: bool, default False
        True if log likelihood decreased during training
    heywood: bool, default False
        True if variance of any dimension (psi) is less than set noise floor
    """

    converged: bool = False
    decreasing_ll: bool = False
    heywood: bool = False

    def display(self) -> None:
        """
        print the fit flags
        """
        print(f"converged: {self.converged}")
        print(f"decreasing ll: {self.decreasing_ll}")
        print(f"heywood: {self.heywood}")
