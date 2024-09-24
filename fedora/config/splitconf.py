from dataclasses import dataclass, field
import typing as t



@dataclass
class NoiseConfig:
    mu: t.Any
    sigma: t.Any
    flip_percent: t.Any



@dataclass
class SplitConfig:
    split_type: str
    noise: NoiseConfig
    num_splits: int  # should be equal to num_clients
    num_noisy_clients: int
    num_class_per_client: int
    dirichlet_alpha: float
    # Train test split ratio within the client,
    # Now this is auto determined by the test set size
    test_fractions: list[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        # assert self.test_fraction == Range(0.0, 1.0), f'Invalid value {self.test_fraction} for test fraction'
        known_splits = {
            "one_noisy_client",
            "n_noisy_clients",
            "n_distinct_noisy_clients",
            "n_distinct_label_flipped_clients",
            "one_label_flipped_client",
            "n_label_flipped_clients",
            "iid",
            "imbalanced",
            "one_imbalanced_client",
            "patho",
            "dirichlet",
        }
        if self.split_type in {
            "one_noisy_client",
            "n_noisy_clients",
            "one_label_flipped_client",
            "n_label_flipped_clients",
        }:
            assert (
                self.noise
            ), "Noise config should be provided for noisy client or label flipped client"
        if self.split_type == "patho":
            assert (
                self.num_class_per_client
            ), "Number of pathological splits should be provided"
        if self.split_type == "dirichlet":
            assert (
                self.dirichlet_alpha
            ), "Dirichlet alpha should be provided for dirichlet split"

        assert self.split_type in known_splits, f"Invalid split type: {self.split_type}"
        assert (
            self.num_noisy_clients <= self.num_splits
        ), "Number of pathological splits should be less than or equal to number of splits"

