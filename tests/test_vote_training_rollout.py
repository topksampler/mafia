"""Check the first trainable action path before paying for a GPU rollout."""

from src.training.vote import VoteDecision, VoteObservation, play_game, vote_messages
from env.rules import GameConfig


class FirstLegalTownVote:
    def choose(self, observation: VoteObservation, *, sample: bool) -> VoteDecision:
        assert observation.role in {"villager", "detective"}
        targets = tuple(observation.legal_actions["targets"])
        assert observation.current_player not in targets
        return VoteDecision((), targets, targets[0])


def test_town_vote_policy_completes_seeded_games_with_legal_targets() -> None:
    for seed in range(10):
        result = play_game(seed, FirstLegalTownVote(), sample=True)
        assert result.winner in {"town", "mafia"}
        assert result.decisions
        assert result.warnings == 0
        assert result.forfeits == 0
        assert all(
            decision.chosen_target in decision.legal_targets
            for decision in result.decisions
        )


def test_vote_training_rollout_accepts_eight_player_config() -> None:
    result = play_game(12, FirstLegalTownVote(), sample=False, config=GameConfig(players=8))
    assert result.winner in {"town", "mafia"}
    assert any("P7" in decision.legal_targets for decision in result.decisions)


def test_vote_prompt_contains_only_active_players_private_events() -> None:
    observation = VoteObservation(
        current_player="P2",
        role="detective",
        day=2,
        phase="vote",
        alive=["P0", "P2", "P3"],
        public_events=[{"type": "speech", "player": "P0", "message": "Vote P3"}],
        private_events=[{"type": "investigation", "target": "P3", "is_mafia": True}],
        legal_actions={"type": "vote", "round": "initial", "targets": ["P0", "P3"]},
    )
    messages = vote_messages(observation, observation.public_events)
    assert "investigation" in messages[1]["content"]
    assert "P2" in messages[1]["content"]
    assert "P3" in messages[1]["content"]
    assert "roles" not in messages[1]["content"]
