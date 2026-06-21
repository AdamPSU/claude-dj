"""Tests for the DJ agent threshold policy."""

from vibe_dj.agent import DJAgent


class TestDJAgentThresholds:
    def test_low_valence_low_vibe_changes_track(self):
        agent = DJAgent()
        d = agent.decide(valence=0.1, vibe_score=0.1, motion_energy=0.2)
        assert d.action == "change_track"

    def test_high_vibe_high_valence_keeps(self):
        agent = DJAgent()
        d = agent.decide(valence=0.8, vibe_score=0.9, motion_energy=0.5)
        assert d.action == "keep"

    def test_high_vibe_negative_valence_changes_track(self):
        agent = DJAgent()
        d = agent.decide(valence=0.15, vibe_score=0.8, motion_energy=0.6)
        assert d.action == "change_track"

    def test_low_vibe_positive_valence_decreases_energy(self):
        agent = DJAgent()
        d = agent.decide(valence=0.8, vibe_score=0.1, motion_energy=0.1)
        assert d.action == "decrease_energy"

    def test_neutral_signals_keep(self):
        agent = DJAgent()
        d = agent.decide(valence=0.5, vibe_score=0.5, motion_energy=0.4)
        assert d.action == "keep"

    def test_decision_has_reason(self):
        agent = DJAgent()
        d = agent.decide(valence=0.1, vibe_score=0.1, motion_energy=0.1)
        assert len(d.reason) > 0

    def test_reset_clears_history(self):
        agent = DJAgent()
        agent.decide(0.5, 0.5, 0.5)
        agent.decide(0.5, 0.5, 0.5)
        agent.reset()
        assert len(agent._history) == 0
