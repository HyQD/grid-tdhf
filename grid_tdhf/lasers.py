import numpy as np


class SineSquareLaser:
    supported_gauges = ("length", "velocity")
    required_params = {"E0", "omega", "ncycles", "gauge", "phase", "t0"}

    def __init__(self, *, E0, omega, ncycles, gauge, phase=0.0, t0=0.0):
        self.E0 = E0
        self.A0 = E0 / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

        self._ncycles_is_one = True if ncycles == 1 else False
        self._A1_t0 = self._A1(0)

        if gauge == "length":
            self.__call__ = self.electric_field
        elif gauge == "velocity":
            self.__call__ = self.vector_potential
        else:
            raise ValueError(f"Gauge '{gauge}' is not supported.")

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def electric_field(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.sin(self.omega * dt + self._phase(dt))
            * self.field_strength
        )
        return pulse

    def vector_potential(self, t):
        dt = t - self.t0

        pulse = (
            self.field_strength
            * (self._A1(dt) - self._A1_t0)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
        )
        return pulse

    def _A1(self, t):
        if not self._ncycles_is_one:
            f0 = (
                self.tprime
                * np.cos(t * (self.omega - 2 * np.pi / self.tprime) + self._phase(t))
                / (self.omega * self.tprime - 2 * np.pi)
            )
        else:
            f0 = 0
        f1 = (
            self.tprime
            * np.cos(t * (self.omega + 2 * np.pi / self.tprime) + self._phase(t))
            / (self.omega * self.tprime + 2 * np.pi)
        )
        f2 = 2 * np.cos(self._phase(t)) * np.cos(self.omega * t) / self.omega
        f3 = 2 * np.sin(self._phase(t)) * np.sin(self.omega * t) / self.omega
        return (1 / 4.0) * (-f0 - f1 + f2 - f3)


class TrapezoidalLaser:
    supported_gauges = ("length", "velocity")
    required_params = {"E0", "omega", "ncycles", "ncycles_ramp", "gauge", "phase", "t0"}

    def __init__(self, *, E0, omega, ncycles, ncycles_ramp, gauge, phase=0.0, t0=0.0):
        self.E0 = E0
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0
        self.T1 = 2 * ncycles_ramp * np.pi / omega
        self.T2 = 2 * (ncycles - ncycles_ramp) * np.pi / omega

        if gauge == "length":
            self.__call__ = self.electric_field
        elif gauge == "velocity":
            self.__call__ = self.vector_potential
        else:
            raise ValueError(f"Gauge '{gauge}' is not supported.")

    def electric_field(self, t):
        dt = t - self.t0
        ret = (
            self.E0
            * np.sin(self.omega * t + self.phase)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
        )
        if dt <= self.T1:
            ret *= dt / self.T1
        elif dt >= self.T2:
            ret *= (self.tprime - dt) / self.T1

        return ret

    def vector_potential(self, t):
        dt = t - self.t0
        if dt <= self.T1:
            ret = self._A1(dt)
        elif (dt > self.T1) and (dt < self.T2):
            ret = self._A1(self.T1) + self._A2(t)
        else:
            ret = self._A1(self.T1) + self._A2(self.T2) + self._A3(t)

        return ret * np.heaviside(dt, 1.0) * np.heaviside(self.tprime - dt, 1.0)

    def _A1(self, t):
        c = self.E0 / (self.T1 * self.omega**2)
        f1 = np.sin(self.omega * t)
        f2 = self.omega * t * np.cos(self.omega * t)
        return c * (f1 - f2)

    def _A2(self, t):
        c = self.E0 / self.omega
        f1 = np.cos(self.omega * self.T1)
        f2 = np.cos(self.omega * t)
        return c * (f1 - f2)

    def _A3(self, t):
        c = self.E0 / (self.T1 * self.omega**2)
        f1 = self.omega * (t - self.tprime) * np.cos(self.omega * t)
        f2 = np.sin(self.omega * t)
        f3 = self.omega * (self.T2 - self.tprime) * np.cos(self.omega * self.T2)
        f4 = np.sin(self.omega * self.T2)
        return c * (f1 - f2 - f3 + f4)


class DiscreteDeltaPulse:
    supported_gauges = "length"
    required_params = {"E0", "dt", "gauge", "phase", "t0"}

    def __init__(self, *, E0, dt, gauge, t0=0.0):
        self.E0 = E0
        self.dt = dt
        self.t0 = t0

        if gauge == "length":
            self.__call__ = self.electric_field
        else:
            raise ValueError(f"Gauge '{gauge}' is not supported.")

    def electric_field(self, t):
        dt = t - self.t0
        return (
            self.E0 * np.heaviside(self.dt - dt, 0.0) * np.heaviside(self.dt + dt, 0.0)
        )
