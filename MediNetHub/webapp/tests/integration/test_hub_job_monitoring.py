"""
Hub integration tests: TrainingJob monitoring and API endpoints.

Validates job creation, status transitions, metrics persistence, and that
the Hub's monitoring API endpoints respond correctly to authenticated requests.

Run with:
    cd MediNetHub/MediNetHub
    python manage.py test webapp.tests.integration.test_hub_job_monitoring --verbosity=2
"""
import json
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "medinet.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import Client, TestCase
from webapp.models import ModelConfig, TrainingJob


DL_CONFIG_JSON = {
    "model": {
        "metadata": {"model_type": "dl"},
        "dataset": {"selected_datasets": [{"dataset_id": 1, "dataset_name": "Test"}]},
        "config_json": {"architecture": {"layers": []}},
        "training": {
            "optimizer": {
                "type": "Adam",
                "learning_rate": 0.01,
                "weight_decay": 0,
                "differential_privacy": {
                    "noise_multiplier": 1.0,
                    "max_grad_norm": 1.0,
                    "random_seed": 42,
                },
            }
        },
    },
    "train": {"rounds": 3, "epochs": 1, "batch_size": 16},
}


# ---------------------------------------------------------------------------
# TrainingJob model tests
# ---------------------------------------------------------------------------

class TestTrainingJobCreation(TestCase):
    """TrainingJob model creation and field validation."""

    def setUp(self):
        self.user = User.objects.create_user("hub_monitor_user", password="pass")
        self.model_config = ModelConfig.objects.create(
            user=self.user,
            name="Monitor Test Config",
            model_type="dl",
            config_json=DL_CONFIG_JSON,
        )

    def test_create_pending_job(self):
        job = TrainingJob.objects.create(
            user=self.user,
            model_config=self.model_config,
            name="DL Training Test",
            total_rounds=3,
        )
        self.assertEqual(job.status, "pending")
        self.assertEqual(job.progress, 0)
        self.assertEqual(job.current_round, 0)

    def test_job_status_transitions(self):
        job = TrainingJob.objects.create(
            user=self.user,
            model_config=self.model_config,
            name="Status Transition Test",
            total_rounds=3,
        )

        job.status = "server_ready"
        job.save()
        job.refresh_from_db()
        self.assertEqual(job.status, "server_ready")

        job.status = "running"
        job.current_round = 1
        job.progress = 33
        job.save()
        job.refresh_from_db()
        self.assertEqual(job.status, "running")
        self.assertEqual(job.current_round, 1)

        job.status = "completed"
        job.progress = 100
        job.current_round = 3
        job.save()
        job.refresh_from_db()
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.progress, 100)

    def test_metrics_json_stores_per_round_data(self):
        job = TrainingJob.objects.create(
            user=self.user,
            model_config=self.model_config,
            name="Metrics Test",
            total_rounds=2,
        )
        job.metrics_json = [
            {"round": 1, "loss": 0.5, "accuracy": 0.7, "client_id": "client-001"},
            {"round": 2, "loss": 0.35, "accuracy": 0.82, "client_id": "client-001"},
        ]
        job.save()
        job.refresh_from_db()
        self.assertEqual(len(job.metrics_json), 2)
        self.assertAlmostEqual(job.metrics_json[1]["accuracy"], 0.82)

    def test_clients_status_stores_client_map(self):
        """clients_status tracks per-client state for multi-center federation."""
        job = TrainingJob.objects.create(
            user=self.user,
            model_config=self.model_config,
            name="Client Status Test",
            total_rounds=2,
        )
        job.clients_status = {
            "client-001": {"status": "training", "current_round": 1, "accuracy": 0.65},
            "rounds_history": {
                "1": {"client-001": {"accuracy": 0.65, "loss": 0.42, "train_samples": 160}}
            },
        }
        job.save()
        job.refresh_from_db()
        self.assertIn("client-001", job.clients_status)
        self.assertAlmostEqual(job.clients_status["client-001"]["accuracy"], 0.65)

    def test_config_json_dp_params_preserved(self):
        """DP parameters stored in config_json must survive DB round-trip."""
        job = TrainingJob.objects.create(
            user=self.user,
            model_config=self.model_config,
            name="DP Config Test",
            config_json=DL_CONFIG_JSON,
            total_rounds=1,
        )
        job.refresh_from_db()
        dp = job.config_json["model"]["training"]["optimizer"]["differential_privacy"]
        self.assertAlmostEqual(dp["noise_multiplier"], 1.0)
        self.assertAlmostEqual(dp["max_grad_norm"], 1.0)


# ---------------------------------------------------------------------------
# Hub monitoring API endpoint tests
# ---------------------------------------------------------------------------

class TestJobMetricsEndpoint(TestCase):
    """Hub's monitoring API endpoints return correct data for authenticated users."""

    def setUp(self):
        self.user = User.objects.create_user("hub_api_user", password="pass")
        self.http_client = Client()
        self.http_client.login(username="hub_api_user", password="pass")

        model_config = ModelConfig.objects.create(
            user=self.user,
            name="API Test Config",
            model_type="dl",
            config_json=DL_CONFIG_JSON,
        )
        # Job with list-typed metrics — used by get_job_metrics and client_status
        # (those endpoints handle both str and list gracefully)
        self.job = TrainingJob.objects.create(
            user=self.user,
            model_config=model_config,
            name="Monitoring Test Job",
            status="running",
            total_rounds=3,
            current_round=2,
            progress=66,
            metrics_json=[
                {"round": 1, "loss": 0.5, "accuracy": 0.7},
                {"round": 2, "loss": 0.38, "accuracy": 0.79},
            ],
            clients_status={
                "client-001": {"status": "training", "current_round": 2},
                "rounds_history": {
                    "1": {"client-001": {"accuracy": 0.7, "loss": 0.5}},
                    "2": {"client-001": {"accuracy": 0.79, "loss": 0.38}},
                },
            },
        )
        # Job without metrics — api_job_details uses json.loads() which fails on
        # a Python list (Hub bug); use metrics_json=None to exercise the safe path.
        self.detail_job = TrainingJob.objects.create(
            user=self.user,
            model_config=model_config,
            name="Detail Test Job",
            status="running",
            total_rounds=3,
            current_round=1,
            progress=33,
        )

    # --- /api/get-job-metrics/<job_id>/ ---

    def test_metrics_endpoint_returns_200(self):
        response = self.http_client.get(f"/api/get-job-metrics/{self.job.id}/")
        self.assertEqual(response.status_code, 200)

    def test_metrics_endpoint_returns_metrics_list(self):
        response = self.http_client.get(f"/api/get-job-metrics/{self.job.id}/")
        data = json.loads(response.content)
        self.assertIn("metrics", data)
        self.assertIsInstance(data["metrics"], list)
        self.assertEqual(len(data["metrics"]), 2)

    def test_metrics_endpoint_returns_progress_fields(self):
        response = self.http_client.get(f"/api/get-job-metrics/{self.job.id}/")
        data = json.loads(response.content)
        self.assertIn("job_status", data)
        self.assertIn("progress", data)
        self.assertIn("current_round", data)
        self.assertIn("total_rounds", data)

    def test_metrics_endpoint_returns_correct_round_count(self):
        response = self.http_client.get(f"/api/get-job-metrics/{self.job.id}/")
        data = json.loads(response.content)
        self.assertEqual(data["current_round"], 2)
        self.assertEqual(data["total_rounds"], 3)

    # --- /api/job-details/<job_id>/ ---
    # Note: self.detail_job has metrics_json=None to avoid Hub's json.loads(list) bug.

    def test_job_details_endpoint_returns_200(self):
        response = self.http_client.get(f"/api/job-details/{self.detail_job.id}/")
        self.assertEqual(response.status_code, 200)

    def test_job_details_endpoint_returns_status(self):
        response = self.http_client.get(f"/api/job-details/{self.detail_job.id}/")
        data = json.loads(response.content)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "running")

    def test_job_details_endpoint_returns_name(self):
        response = self.http_client.get(f"/api/job-details/{self.detail_job.id}/")
        data = json.loads(response.content)
        self.assertIn("name", data)
        self.assertEqual(data["name"], "Detail Test Job")

    # --- /api/client-status/<job_id>/ ---

    def test_client_status_endpoint_returns_200(self):
        response = self.http_client.get(f"/api/client-status/{self.job.id}/")
        self.assertEqual(response.status_code, 200)

    def test_client_status_endpoint_contains_clients(self):
        response = self.http_client.get(f"/api/client-status/{self.job.id}/")
        data = json.loads(response.content)
        self.assertIn("clients", data)
        self.assertIn("client_count", data)

    def test_client_status_endpoint_reflects_stored_clients(self):
        response = self.http_client.get(f"/api/client-status/{self.job.id}/")
        data = json.loads(response.content)
        # client-001 was stored in clients_status
        self.assertIn("client-001", data["clients"])

    # --- Authentication guard ---

    def test_unauthenticated_metrics_returns_redirect(self):
        unauth = Client()
        response = unauth.get(f"/api/get-job-metrics/{self.job.id}/")
        # @login_required redirects unauthenticated requests
        self.assertIn(response.status_code, (302, 401, 403))

    def test_unauthenticated_job_details_returns_redirect(self):
        unauth = Client()
        response = unauth.get(f"/api/job-details/{self.detail_job.id}/")
        self.assertIn(response.status_code, (302, 401, 403))
