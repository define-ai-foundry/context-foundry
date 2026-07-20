"""Tests for context_foundry.fusion.sources.base."""

import pytest

from context_foundry.fusion.sources.base import SapientSource


def test_sapient_source_is_abstract():
    with pytest.raises(TypeError):
        SapientSource()


def test_sapient_source_subclass_must_implement_iter_events():
    class Incomplete(SapientSource):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_base_iter_events_raises_not_implemented_when_invoked_via_super():
    class Passthrough(SapientSource):
        def iter_events(self):
            return super().iter_events()

    with pytest.raises(NotImplementedError):
        Passthrough().iter_events()
