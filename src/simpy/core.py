"""
Core components for event-discrete simulation environments.

"""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import count
from types import MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simpy.events import (
        NORMAL,
        URGENT,
        AllOf,
        AnyOf,
        Event,
        EventPriority,
        Process,
        ProcessGenerator,
        Timeout,
    )

PENDING: object = object()
'Unique object to identify pending values of events.'

Infinity: float = float('inf')
T = TypeVar('T')


class BoundClass(Generic[T]):
    """Allows classes to behave like methods.

    The ``__get__()`` descriptor is basically identical to
    ``function.__get__()`` and binds the first argument of the ``cls`` to the
    descriptor instance.

    """

    def __init__(self, cls: Type[T]):
        self.cls = cls

    def __get__(
        self,
        instance: Optional[BoundClass[T]],
        owner: Optional[Type[BoundClass[T]]] = None,
    ) -> Union[Type[T], MethodType]:
        if instance is None:
            return self.cls
        return MethodType(self.cls, instance)

    @staticmethod
    def bind_early(instance: Any) -> None:
        """Bind all :class:`BoundClass` attributes of the *instance's* class
        to the instance itself to increase performance."""
        cls = type(instance)
        for name, obj in cls.__dict__.items():
            if isinstance(obj, BoundClass):
                setattr(instance, name, obj.__get__(instance, cls))  # type: ignore


class EmptySchedule(Exception):
    """Thrown by an :class:`Environment` if there are no further events to be
    processed."""


class StopSimulation(Exception):
    """Indicates that the simulation should stop now."""

    @classmethod
    def callback(cls, event: Event) -> None:
        """Used as callback in :meth:`Environment.run()` to stop the simulation
        when the *until* event occurred."""
        raise cls(event.value)


SimTime = Union[int, float]


class Environment:
    """Execution environment for an event-based simulation. The passing of time
    is simulated by stepping from event to event.

    You can provide an *initial_time* for the environment. By default, it
    starts at ``0``.

    This class also provides aliases for common event types, for example
    :attr:`process`, :attr:`timeout` and :attr:`event`.

    """

    def __init__(self, initial_time: SimTime = 0):
        self._now = initial_time
        self._queue: List[Tuple[SimTime, EventPriority, int, Event]] = []
        self._eid = count()
        self._active_proc = None
        BoundClass.bind_early(self)

    @property
    def now(self) -> SimTime:
        """The current simulation time."""
        return self._now

    @property
    def active_process(self):
        """The currently active process of the environment."""
        return self._active_proc

    if TYPE_CHECKING:
        def process(self, generator: ProcessGenerator):
            """Create a new :class:`~simpy.events.Process` instance for
            *generator*."""
            from simpy.events import Process
            return Process(self, generator)

        def timeout(self, delay: SimTime = 0, value: Optional[Any] = None):
            """Return a new :class:`~simpy.events.Timeout` event with a *delay*
            and, optionally, a *value*."""
            from simpy.events import Timeout
            return Timeout(self, delay, value)

        def event(self):
            """Return a new :class:`~simpy.events.Event` instance.

            Yielding this event suspends a process until another process
            triggers the event.
            """
            from simpy.events import Event
            return Event(self)

        def all_of(self, events: Iterable[Event]):
            """Return a :class:`~simpy.events.AllOf` condition for *events*."""
            from simpy.events import AllOf
            return AllOf(self, events)

        def any_of(self, events: Iterable[Event]):
            """Return a :class:`~simpy.events.AnyOf` condition for *events*."""
            from simpy.events import AnyOf
            return AnyOf(self, events)
    else:
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from simpy.events import Process, Timeout, Event, AllOf, AnyOf
        process = BoundClass(Process)
        timeout = BoundClass(Timeout)
        event = BoundClass(Event)
        all_of = BoundClass(AllOf)
        any_of = BoundClass(AnyOf)

    def schedule(
        self, event: Event, priority: EventPriority = NORMAL, delay: SimTime = 0
    ) -> None:
        """Schedule an *event* with a given *priority* and a *delay*."""
        heappush(self._queue, (self._now + delay, priority, next(self._eid), event))

    def peek(self) -> SimTime:
        """Get the time of the next scheduled event. Return
        :data:`~simpy.core.Infinity` if there is no further event."""
        try:
            return self._queue[0][0]
        except IndexError:
            return Infinity

    def step(self) -> None:
        """Process the next event.

        Raise an :exc:`EmptySchedule` if no further events are available.

        """
        try:
            self._now, _, _, event = heappop(self._queue)
        except IndexError:
            raise EmptySchedule from None

        # Process the event
        event._ok = True
        if event.callbacks:
            for callback in event.callbacks:
                callback(event)
        if not event.defused:
            event._value = event.callbacks
            event.callbacks = []

    def run(self, until: Optional[Union[SimTime, Event]] = None) -> Optional[Any]:
        """Executes :meth:`step()` until the given criterion *until* is met.

        - If it is ``None`` (which is the default), this method will return
          when there are no further events to be processed.

        - If it is an :class:`~simpy.events.Event`, the method will continue
          stepping until this event has been triggered and will return its
          value.  Raises a :exc:`RuntimeError` if there are no further events
          to be processed and the *until* event was not triggered.

        - If it is a number, the method will continue stepping
          until the environment's time reaches *until*.

        """
        if until is None:
            while self._queue:
                self.step()
        elif isinstance(until, Event):
            if until.triggered:
                return until.value
            while not until.triggered:
                if not self._queue:
                    raise RuntimeError('No scheduled events left but "until" event was not triggered')
                self.step()
            return until.value
        else:
            try:
                until = float(until)
            except ValueError:
                raise ValueError(f"Expected 'until' to be a number, got '{until}'")
            
            if until < self.now:
                raise ValueError(f'until(={until}) must be greater than the current simulation time.')
            
            while self._queue and self._queue[0][0] <= until:
                self.step()
                if self.now >= until:
                    break

        return None

    def step(self) -> None:
        """Process the next event.

        Raise an :exc:`EmptySchedule` if no further events are available.

        """
        try:
            self._now, _, _, event = heappop(self._queue)
        except IndexError:
            raise EmptySchedule()

        # Process the event
        event._ok = True
        event._defused = False

        if event._value is not PENDING:
            if event.callbacks:
                event._value = None
                for callback in event.callbacks:
                    callback(event)
                event.callbacks = None  # type: ignore
            if not event._defused:
                if isinstance(event._value, Exception):
                    raise event._value
        else:
            event._value = None
            for callback in event.callbacks:
                callback(event)
            event.callbacks = None  # type: ignore

    def timeout(self, delay: SimTime = 0, value: Optional[Any] = None) -> Timeout:
        """Schedule a :class:`~simpy.events.Timeout` event for time *now* +
        *delay*."""
        if delay < 0:
            raise ValueError(f'Negative delay {delay}')
        event = Timeout(self, delay, value)
        self.schedule(event)
        return event
