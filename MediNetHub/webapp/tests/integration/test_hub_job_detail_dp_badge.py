"""
Hub integration tests: Tarea 7 — Differential Privacy badge in job_detail template.

Validates that:
- A "Differential Privacy" card is always rendered in the job detail view
- When privacy_epsilon is None → "Inactive" badge + "No differential privacy" message
- When privacy_epsilon is set → "Active" badge + formatted epsilon value
- Delta is shown when privacy_delta is set alongside epsilon

Run with:
    cd MediNetHub/MediNetHub
    PYTHONIOENCODING=utf-8 python manage.py test webapp.tests.integration.test_hub_job_detail_dp_badge --verbosity=2
"""
import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "medinet.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from webapp.models import ModelConfig, TrainingJob


DL_CONFIG_JSON = {
    "model": {
        "metadata": {"model_type": "dl"},
        "dataset": {"selected_datasets": [{"dataset_id": 1, "dataset_name": "T"}]},
        "training": {
            "optimizer": {
                "type": "Adam",
                "learning_rate": 0.001,
            }
        },
    },
    "train": {"rounds": 3, "epochs": 1, "batch_size": 32},
    "federated": {
        "name": "FedAvg",
        "parameters": {
            "fraction_fit": 1.0,
            "min_fit_clients": 1,
            "min_available_clients": 1,
        },
    },
}


def _make_job(user, model_config, **kwargs):
    return TrainingJob.objects.create(
        user=user,
        model_config=model_config,
        name="DP Badge Test Job",
        status="completed",
        total_rounds=3,
        **kwargs,
    )


class TestJobDetailDPBadge(TestCase):
    """job_detail template renders the Differential Privacy card correctly."""

    def setUp(self):
        self.user = User.objects.create_user("dp_badge_user", password="pass")
        self.client.login(username="dp_badge_user", password="pass")
        self.model_config = ModelConfig.objects.create(
            user=self.user,
            name="DP Badge Config",
            model_type="dl",
            config_json=DL_CONFIG_JSON,
        )

    def _get_detail(self, job):
        url = reverse("job_detail", kwargs={"job_id": job.id})
        return self.client.get(url)

    # --- Card always present ---

    def test_dp_card_always_rendered(self):
        """Differential Privacy card is always present in the job detail page."""
        job = _make_job(self.user, self.model_config)
        resp = self._get_detail(job)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Differential Privacy")

    # --- No DP (privacy_epsilon is None) ---

    def test_inactive_badge_when_no_epsilon(self):
        """When privacy_epsilon is None, the card shows the Inactive badge."""
        job = _make_job(self.user, self.model_config)
        self.assertIsNone(job.privacy_epsilon)
        resp = self._get_detail(job)
        self.assertContains(resp, "Inactive")

    def test_no_dp_message_when_no_epsilon(self):
        """When privacy_epsilon is None, the card explains DP is not active."""
        job = _make_job(self.user, self.model_config)
        resp = self._get_detail(job)
        self.assertContains(resp, "No differential privacy applied")

    def test_active_badge_not_shown_when_no_epsilon(self):
        """Active badge must NOT appear when privacy_epsilon is None."""
        job = _make_job(self.user, self.model_config)
        resp = self._get_detail(job)
        self.assertNotContains(resp, 'badge bg-info text-dark">Active')

    # --- Active DP (privacy_epsilon set) ---

    def test_active_badge_when_epsilon_set(self):
        """When privacy_epsilon is set, the card shows the Active badge."""
        job = _make_job(
            self.user, self.model_config, privacy_epsilon=2.5, privacy_delta=1e-5
        )
        resp = self._get_detail(job)
        self.assertContains(resp, "Active")

    def test_epsilon_value_displayed(self):
        """Epsilon value is rendered in the card body."""
        job = _make_job(
            self.user, self.model_config, privacy_epsilon=1.2345, privacy_delta=1e-5
        )
        resp = self._get_detail(job)
        # Django floatformat:4 renders 1.2345 as "1.2345"
        self.assertContains(resp, "1.2345")

    def test_epsilon_label_displayed(self):
        """'Epsilon (ε)' label is rendered when epsilon is set."""
        job = _make_job(
            self.user, self.model_config, privacy_epsilon=0.8, privacy_delta=1e-5
        )
        resp = self._get_detail(job)
        self.assertContains(resp, "Epsilon")

    def test_delta_displayed_when_set(self):
        """Delta (δ) row is rendered when privacy_delta is set alongside epsilon."""
        job = _make_job(
            self.user, self.model_config, privacy_epsilon=3.0, privacy_delta=1e-5
        )
        resp = self._get_detail(job)
        self.assertContains(resp, "Delta")
        self.assertContains(resp, "1e-5")

    def test_no_dp_message_not_shown_when_epsilon_set(self):
        """'No differential privacy applied' message must NOT appear when epsilon is set."""
        job = _make_job(
            self.user, self.model_config, privacy_epsilon=0.5, privacy_delta=1e-5
        )
        resp = self._get_detail(job)
        self.assertNotContains(resp, "No differential privacy applied")

    def test_inactive_badge_not_shown_when_epsilon_set(self):
        """Inactive badge must NOT appear when privacy_epsilon is set."""
        job = _make_job(
            self.user, self.model_config, privacy_epsilon=0.5, privacy_delta=1e-5
        )
        resp = self._get_detail(job)
        self.assertNotContains(resp, 'badge bg-secondary">Inactive')

    # --- Edge cases ---

    def test_low_epsilon_small_value_displayed(self):
        """Very low epsilon (strong privacy) is displayed with full precision."""
        job = _make_job(
            self.user, self.model_config, privacy_epsilon=0.0001, privacy_delta=1e-5
        )
        resp = self._get_detail(job)
        self.assertContains(resp, "0.0001")

    def test_high_epsilon_displayed(self):
        """High epsilon (weak privacy) is also displayed correctly."""
        job = _make_job(
            self.user, self.model_config, privacy_epsilon=15.0, privacy_delta=1e-5
        )
        resp = self._get_detail(job)
        self.assertContains(resp, "15.0000")

    def test_epsilon_without_delta_renders_safely(self):
        """If only privacy_epsilon is set (delta None), the page still renders."""
        job = _make_job(self.user, self.model_config, privacy_epsilon=2.0)
        resp = self._get_detail(job)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Active")
        self.assertContains(resp, "2.0000")

    def test_login_required(self):
        """Unauthenticated requests are redirected to login."""
        self.client.logout()
        job = _make_job(self.user, self.model_config)
        resp = self._get_detail(job)
        self.assertRedirects(resp, f"/login/?next=/jobs/{job.id}/", fetch_redirect_response=False)
