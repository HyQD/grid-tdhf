class ConvergenceError(Exception):
    def __init__(self, message="Integrator did not converge."):
        super().__init__(message)
