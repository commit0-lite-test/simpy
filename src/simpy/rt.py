"""Execution environment for events that synchronizes passing of time
with the real-time (aka *wall-clock time*).
"""
from time import monotonic, sleep

from simpy.core import Environment, Infinity, SimTime


class RealtimeEnvironment(Environment):
    """Execution environment for an event-based simulation which is
    synchronized with the real-time (also known as wall-clock time). A time
    step will take *factor* seconds of real time (one second by default).
    A step from ``0`` to ``3`` with a ``factor=0.5`` will, for example, take at
    least 1.5 seconds.

    The :meth:`step()` method will raise a :exc:`RuntimeError` if a time step
    took too long to compute. This behaviour can be disabled by setting
    *strict* to ``False``.
    """

    def __init__(
        self, initial_time: SimTime = 0, factor: float = 1.0, strict: bool = True
    ):
        Environment.__init__(self, initial_time)
        self.env_start = initial_time
        self.real_start = monotonic()
        self._factor = factor
        self._strict = strict

    @property
    def factor(self) -> float:
        """Scaling factor of the real-time."""
        return self._factor

    @property
    def strict(self) -> bool:
        """Running mode of the environment. :meth:`step()` will raise a
        :exc:`RuntimeError` if this is set to ``True`` and the processing of
        events takes too long."""
        return self._strict

    def sync(self) -> None:
        """Synchronize the internal time with the current wall-clock time.

        This can be useful to prevent :meth:`step()` from raising an error if
        a lot of time passes between creating the RealtimeEnvironment and
        calling :meth:`run()` or :meth:`step()`.
        """
        self.env_start = self._now
        self.real_start = monotonic()

    def step(self) -> None:
        """Process the next event after enough real-time has passed for the
        event to happen.

        The delay is scaled according to the real-time :attr:`factor`. With
        :attr:`strict` mode enabled, a :exc:`RuntimeError` will be raised, if
        the event is processed too slowly.
        """
        evt_time = self.peek()

        if evt_time < Infinity:
            real_time = self.real_start + (evt_time - self.env_start) / self._factor
            sleep_time = real_time - monotonic()

            if sleep_time > 0:
                sleep(sleep_time)

            if self._strict and monotonic() - real_time > 0:
                raise RuntimeError("Simulation too slow for real-time execution")

        return Environment.step(self)
