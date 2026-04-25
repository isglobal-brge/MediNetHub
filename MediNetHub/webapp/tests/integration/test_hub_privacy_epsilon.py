"""
Hub integration tests: privacy_epsilon / privacy_delta aggregation in fit_metrics_aggregation_fn.

Validates that:
- privacy_epsilon is saved to the TrainingJob after each round
- worst-case (max) epsilon is chosen across clients
- invalid / negative / non-finite epsilon values are ignored
- missing privacy_epsilon in client metrics is handled safely
- privacy_delta is always set to 1e-5 when epsilon is valid

Run with:
    cd MediNetHub/MediNetHub
    python manage.py test webapp.tests.integration.test_hub_privacy_epsilon --verbosity=2
"""
import math
import os
from unittest.mock import MagicMock, patch, PropertyMock

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "medinet.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import TestCase

from webapp.models import ModelConfig, TrainingJob
from webapp.server_fn.server import create_fit_metrics_aggregation_fn


DL_CONFIG_JSON = {
    "model": {
        "metadata": {"model_type": "dl"},
        "dataset": {"selected_datasets": [{"dataset_id": 1, "dataset_name": "T"}]},
        "training": {
            "optimizer": {
                "type": "Adam",
                "differential_privacy": {"noise_multiplier": 1.0, "max_grad_norm": 1.0},
            }
        },
    },
    "train": {"rounds": 3, "epochs": 1},
}


def _make_job(user, model_config):
    return TrainingJob.objects.create(
        user=user,
        model_config=model_config,
        name="DP Test Job",
        status="running",
        total_rounds=3,
        current_round=1,
    )


def _make_server_manager(job):
    """Return a minimal ServerManager-like mock backed by the real DB job.

    sm.job is the real Django model instance so save() calls hit the DB.
    save_metrics / update_client_status are auto-mocked (no-ops for these tests).
    """
    sm = MagicMock()
    sm.job = job  # real model; Django's own save() method is unmodified
    return sm


def _make_strategy(current_round=1):
    strategy = MagicMock()
    strategy.current_round = current_round
    strategy.round_start_time = None
    return strategy


class TestFitMetricsEpsilonAggregation(TestCase):
    """fit_metrics_aggregation_fn saves privacy_epsilon to job correctly."""

    def setUp(self):
        self.user = User.objects.create_user("eps_test_user", password="pass")
        self.model_config = ModelConfig.objects.create(
            user=self.user,
            name="Eps Config",
            model_type="dl",
            config_json=DL_CONFIG_JSON,
        )
        self.job = _make_job(self.user, self.model_config)

    def _run_aggregation(self, client_metrics_list, current_round=1):
        """Helper: build the aggregation fn and call it with the given metrics list."""
        sm = _make_server_manager(self.job)
        strategy = _make_strategy(current_round)
        fn = create_fit_metrics_aggregation_fn(sm, strategy)
        result = fn(client_metrics_list)
        # Reload job from DB to get persisted values
        self.job.refresh_from_db()
        return result

    # --- Nominal cases ---

    def test_single_client_epsilon_saved(self):
        """Single client with valid epsilon → saved to job."""
        metrics = [(100, {"accuracy": 0.9, "loss": 0.1, "privacy_epsilon": 2.5,
                          "precision": 0.9, "recall": 0.9, "f1": 0.9})]
        self._run_aggregation(metrics)
        self.assertAlmostEqual(self.job.privacy_epsilon, 2.5, places=6)
        self.assertAlmostEqual(self.job.privacy_delta, 1e-5, places=10)

    def test_multiple_clients_max_epsilon_saved(self):
        """Multiple clients → worst-case (maximum) epsilon is persisted."""
        metrics = [
            (100, {"accuracy": 0.85, "loss": 0.2, "privacy_epsilon": 1.8,
                   "precision": 0.8, "recall": 0.8, "f1": 0.8}),
            (80,  {"accuracy": 0.90, "loss": 0.15, "privacy_epsilon": 3.2,
                   "precision": 0.9, "recall": 0.9, "f1": 0.9}),
            (120, {"accuracy": 0.88, "loss": 0.18, "privacy_epsilon": 2.1,
                   "precision": 0.85, "recall": 0.85, "f1": 0.85}),
        ]
        self._run_aggregation(metrics)
        self.assertAlmostEqual(self.job.privacy_epsilon, 3.2, places=6)

    def test_epsilon_updated_across_rounds(self):
        """Each round overwrites privacy_epsilon with the latest cumulative value."""
        metrics_r1 = [(100, {"accuracy": 0.8, "loss": 0.3, "privacy_epsilon": 1.0,
                             "precision": 0.8, "recall": 0.8, "f1": 0.8})]
        metrics_r2 = [(100, {"accuracy": 0.85, "loss": 0.25, "privacy_epsilon": 2.0,
                             "precision": 0.85, "recall": 0.85, "f1": 0.85})]

        sm = _make_server_manager(self.job)
        strategy = _make_strategy(current_round=1)
        fn = create_fit_metrics_aggregation_fn(sm, strategy)

        strategy.current_round = 1
        fn(metrics_r1)
        self.job.refresh_from_db()
        self.assertAlmostEqual(self.job.privacy_epsilon, 1.0, places=6)

        strategy.current_round = 2
        fn(metrics_r2)
        self.job.refresh_from_db()
        self.assertAlmostEqual(self.job.privacy_epsilon, 2.0, places=6)

    def test_delta_always_1e5(self):
        """privacy_delta is always set to 1e-5 regardless of how many clients."""
        metrics = [(50, {"accuracy": 0.9, "loss": 0.1, "privacy_epsilon": 0.7,
                         "precision": 0.9, "recall": 0.9, "f1": 0.9})]
        self._run_aggregation(metrics)
        self.assertEqual(self.job.privacy_delta, 1e-5)

    # --- Missing / absent epsilon ---

    def test_no_epsilon_in_client_metrics_leaves_job_unchanged(self):
        """Clients not reporting epsilon → job.privacy_epsilon stays None."""
        metrics = [(100, {"accuracy": 0.9, "loss": 0.1,
                          "precision": 0.9, "recall": 0.9, "f1": 0.9})]
        self._run_aggregation(metrics)
        self.assertIsNone(self.job.privacy_epsilon)
        self.assertIsNone(self.job.privacy_delta)

    def test_mixed_clients_some_with_epsilon(self):
        """Only clients reporting valid epsilon contribute; others ignored."""
        metrics = [
            (100, {"accuracy": 0.9, "loss": 0.1, "privacy_epsilon": 2.0,
                   "precision": 0.9, "recall": 0.9, "f1": 0.9}),
            (80,  {"accuracy": 0.88, "loss": 0.15,  # no privacy_epsilon key
                   "precision": 0.88, "recall": 0.88, "f1": 0.88}),
        ]
        self._run_aggregation(metrics)
        self.assertAlmostEqual(self.job.privacy_epsilon, 2.0, places=6)

    # --- Invalid / adversarial epsilon values ---

    def test_negative_epsilon_ignored(self):
        """Node sends -1.0 when DP failed — must be ignored."""
        metrics = [(100, {"accuracy": 0.9, "loss": 0.1, "privacy_epsilon": -1.0,
                          "precision": 0.9, "recall": 0.9, "f1": 0.9})]
        self._run_aggregation(metrics)
        self.assertIsNone(self.job.privacy_epsilon)

    def test_zero_epsilon_ignored(self):
        """Zero epsilon is physically impossible — ignore."""
        metrics = [(100, {"accuracy": 0.9, "loss": 0.1, "privacy_epsilon": 0.0,
                          "precision": 0.9, "recall": 0.9, "f1": 0.9})]
        self._run_aggregation(metrics)
        self.assertIsNone(self.job.privacy_epsilon)

    def test_nan_epsilon_ignored(self):
        """NaN epsilon must be ignored (isfinite guard)."""
        metrics = [(100, {"accuracy": 0.9, "loss": 0.1, "privacy_epsilon": float("nan"),
                          "precision": 0.9, "recall": 0.9, "f1": 0.9})]
        self._run_aggregation(metrics)
        self.assertIsNone(self.job.privacy_epsilon)

    def test_inf_epsilon_ignored(self):
        """Infinite epsilon must be ignored (isfinite guard)."""
        metrics = [(100, {"accuracy": 0.9, "loss": 0.1, "privacy_epsilon": float("inf"),
                          "precision": 0.9, "recall": 0.9, "f1": 0.9})]
        self._run_aggregation(metrics)
        self.assertIsNone(self.job.privacy_epsilon)

    def test_string_epsilon_ignored(self):
        """Non-numeric privacy_epsilon from Hub-manipulated metrics → ignored safely."""
        metrics = [(100, {"accuracy": 0.9, "loss": 0.1, "privacy_epsilon": "bad",
                          "precision": 0.9, "recall": 0.9, "f1": 0.9})]
        self._run_aggregation(metrics)
        self.assertIsNone(self.job.privacy_epsilon)

    def test_none_epsilon_ignored(self):
        """Explicit None privacy_epsilon → ignored."""
        metrics = [(100, {"accuracy": 0.9, "loss": 0.1, "privacy_epsilon": None,
                          "precision": 0.9, "recall": 0.9, "f1": 0.9})]
        self._run_aggregation(metrics)
        self.assertIsNone(self.job.privacy_epsilon)

    def test_all_invalid_epsilon_leaves_job_unchanged(self):
        """All clients send invalid epsilon → job.privacy_epsilon stays None."""
        metrics = [
            (100, {"accuracy": 0.9, "loss": 0.1, "privacy_epsilon": -1.0,
                   "precision": 0.9, "recall": 0.9, "f1": 0.9}),
            (80,  {"accuracy": 0.85, "loss": 0.2, "privacy_epsilon": float("nan"),
                   "precision": 0.85, "recall": 0.85, "f1": 0.85}),
        ]
        self._run_aggregation(metrics)
        self.assertIsNone(self.job.privacy_epsilon)

    def test_valid_epsilon_wins_over_invalids(self):
        """Mix of valid and invalid epsilon → only valid values contribute."""
        metrics = [
            (100, {"accuracy": 0.9, "loss": 0.1, "privacy_epsilon": -1.0,
                   "precision": 0.9, "recall": 0.9, "f1": 0.9}),
            (80,  {"accuracy": 0.85, "loss": 0.2, "privacy_epsilon": 4.5,
                   "precision": 0.85, "recall": 0.85, "f1": 0.85}),
            (60,  {"accuracy": 0.88, "loss": 0.15, "privacy_epsilon": float("inf"),
                   "precision": 0.88, "recall": 0.88, "f1": 0.88}),
        ]
        self._run_aggregation(metrics)
        self.assertAlmostEqual(self.job.privacy_epsilon, 4.5, places=6)

    # --- Other metrics still work ---

    def test_other_metrics_aggregated_correctly(self):
        """Adding epsilon tracking must not break normal metric aggregation."""
        metrics = [
            (100, {"accuracy": 0.8, "loss": 0.3, "precision": 0.75, "recall": 0.70,
                   "f1": 0.72, "privacy_epsilon": 2.0}),
            (100, {"accuracy": 0.9, "loss": 0.1, "precision": 0.85, "recall": 0.88,
                   "f1": 0.86, "privacy_epsilon": 1.5}),
        ]
        result = self._run_aggregation(metrics)
        self.assertAlmostEqual(result["accuracy"], 0.85, places=6)
        self.assertAlmostEqual(result["loss"], 0.2, places=6)

    def test_empty_metrics_returns_empty_dict(self):
        """Empty metrics list → returns {} without crashing."""
        sm = _make_server_manager(self.job)
        strategy = _make_strategy()
        fn = create_fit_metrics_aggregation_fn(sm, strategy)
        result = fn([])
        self.assertEqual(result, {})


class TestTrainingJobPrivacyFields(TestCase):
    """TrainingJob model: privacy fields default to None, can be set."""

    def setUp(self):
        self.user = User.objects.create_user("field_test_user", password="pass")
        self.model_config = ModelConfig.objects.create(
            user=self.user,
            name="Field Config",
            model_type="dl",
            config_json=DL_CONFIG_JSON,
        )

    def test_privacy_fields_default_null(self):
        """New TrainingJob has privacy_epsilon and privacy_delta as None."""
        job = TrainingJob.objects.create(
            user=self.user,
            model_config=self.model_config,
            name="Null DP Job",
            status="pending",
            total_rounds=3,
        )
        self.assertIsNone(job.privacy_epsilon)
        self.assertIsNone(job.privacy_delta)

    def test_privacy_fields_can_be_set(self):
        """privacy_epsilon and privacy_delta can be written and retrieved."""
        job = TrainingJob.objects.create(
            user=self.user,
            model_config=self.model_config,
            name="Set DP Job",
            status="running",
            total_rounds=3,
            privacy_epsilon=3.14,
            privacy_delta=1e-5,
        )
        job.refresh_from_db()
        self.assertAlmostEqual(job.privacy_epsilon, 3.14, places=6)
        self.assertAlmostEqual(job.privacy_delta, 1e-5, places=10)

    def test_privacy_fields_persist_via_update_fields(self):
        """update_fields=['privacy_epsilon', 'privacy_delta'] persists correctly."""
        job = TrainingJob.objects.create(
            user=self.user,
            model_config=self.model_config,
            name="Update DP Job",
            status="running",
            total_rounds=3,
        )
        job.privacy_epsilon = 7.77
        job.privacy_delta = 1e-5
        job.save(update_fields=["privacy_epsilon", "privacy_delta"])
        job.refresh_from_db()
        self.assertAlmostEqual(job.privacy_epsilon, 7.77, places=6)
        self.assertAlmostEqual(job.privacy_delta, 1e-5, places=10)
