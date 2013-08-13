"""
The ``simpy`` module provides SimPy's end-user API. It therefore
aggregates Simpy's various classes and methods:


Core classes and functions
--------------------------

.. currentmodule:: simpy.core

- :class:`Environment`: SimPy's central class. It contains
  the simulation's state and lets the PEMs interact with it (i.e.,
  schedule events).
- :class:`Process`: This class represents a PEM while
  it is executed in an environment. An instance of it is returned by
  :meth:`Environment.start()`. It inherits :class:`Event`.
- :class:`Interrupt`: This exception is thrown into a process if it gets
  interrupted by another one.


Resources
---------

.. currentmodule:: simpy.resources.resource

- :class:`Resource`: Can be used by a limited number of processes at a
  time (e.g., a gas station with a limited number of fuel pumps).

- :class:`PriorityResource`: Like :class:`Resource`, but waiting
  processes are sorted by priority.

- :class:`PreemptiveResource`: Version of :class:`Resource` with
  preemption.

.. currentmodule:: simpy.resources.container

- :class:`Container`: Models the production and consumption of a
  homogeneous, undifferentiated bulk. It may either be continuous (like
  water) or discrete (like apples).

.. currentmodule:: simpy.resources.store

- :class:`Store`: Allows the production and consumption of discrete
  Python objects.

- :class:`FilterStore`: Like :class:`Store`, but items taken out of it
  can be filtered with a user-defined function.


Monitoring
----------

.. currentmodule:: simpy.monitoring

*[Not yet implemented]*


Other
-----

.. currentmodule:: simpy

.. autofunction:: test

.. - :func:`test`: Run the test suite on the installed copy of Simpy.

"""
from simpy.core import Environment, Interrupt, Process
from simpy.resources.resource import (
    Resource, PriorityResource, PreemptiveResource)
from simpy.resources.container import Container
from simpy.resources.store import Store, FilterStore


__all__ = [
    'test',
    'Environment', 'Interrupt', 'Process',
    'Resource', 'PriorityResource', 'PreemptiveResource',
    'Container',
    'Store', 'FilterStore',
]
__version__ = '3.0a1'


def test():
    """Runs SimPy's test suite via `py.test <http://pytest.org/latest/>`_."""
    import os.path
    try:
        import mock
        import pytest
    except ImportError:
        print('You need pytest and mock to run the tests. '
              'Try "pip install pytest mock".')
    else:
        pytest.main([os.path.dirname(__file__)])
