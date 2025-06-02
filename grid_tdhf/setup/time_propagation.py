def setup_time_propagation(imaginary=False):
    if imaginary:
        from grid_tdhf.time_propagation.imag import run_imag_time_propagation
    else:
        from grid_tdhf.time_propagation.imag import run_time_propagation
