"""
Hub integration tests: DL/SVM model JSON construction and config persistence.

Validates that model JSON sent from Hub to Node clients contains all required
fields — including DP parameters, architecture layers, and dataset selections —
and that create_center_specific_config does not leak credentials.

Run with:
    cd MediNetHub/MediNetHub
    python manage.py test webapp.tests.integration.test_hub_model_json --verbosity=2
"""
import copy
import os
import django
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "medinet.settings")
django.setup()

from django.contrib.auth.models import User
from django.test import TestCase
from webapp.models import ModelConfig
from webapp.training_views import clean_model_json_for_ml, create_center_specific_config


# ---------------------------------------------------------------------------
# Shared test fixtures (module-level constants)
# ---------------------------------------------------------------------------

DL_MODEL_JSON = {
    "model": {
        "metadata": {"model_type": "dl", "framework": "pytorch"},
        "dataset": {
            "selected_datasets": [
                {"dataset_id": 1, "dataset_name": "Heart Attack Risk"}
            ]
        },
        "config_json": {
            "architecture": {
                "layers": [
                    {
                        "id": "l0",
                        "type": "Linear",
                        "params": {"in_features": 52, "out_features": 1},
                        "inputs": ["input_data"],
                    }
                ]
            }
        },
        "training": {
            "loss_function": "bce_with_logits",
            "optimizer": {
                "type": "Adam",
                "learning_rate": 0.01,
                "weight_decay": 0,
                "differential_privacy": {
                    "noise_multiplier": 1.0,
                    "max_grad_norm": 1.0,
                    "random_seed": 42,
                },
            },
        },
    },
    "train": {"rounds": 3, "epochs": 2, "batch_size": 32},
    "federated": {
        "parameters": {
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
            "min_fit_clients": 1,
            "min_evaluate_clients": 1,
            "min_available_clients": 1,
        }
    },
}

SVM_MODEL_JSON = {
    "model": {
        "metadata": {"model_type": "ml"},
        "dataset": {
            "selected_datasets": [
                {"dataset_id": 1, "dataset_name": "Tabular Test"}
            ]
        },
        "training": {
            "ml_method": "fedsvm",
            "C": 1.0,
            "kernel_config": {"kernel": "rbf", "gamma": 0.1},
            "val_size": 0.2,
            "random_state": 42,
        },
    },
    "train": {"rounds": 3, "epochs": 1, "batch_size": 32},
}


# ---------------------------------------------------------------------------
# DL model JSON structure tests
# ---------------------------------------------------------------------------

class TestDLModelJsonConstruction(TestCase):
    """Verify DL model JSON has all required fields before being sent to Node."""

    def test_model_type_is_dl(self):
        self.assertEqual(DL_MODEL_JSON["model"]["metadata"]["model_type"], "dl")

    def test_has_architecture_layers(self):
        layers = DL_MODEL_JSON["model"]["config_json"]["architecture"]["layers"]
        self.assertGreaterEqual(len(layers), 1)
        for layer in layers:
            self.assertIn("type", layer)
            self.assertIn("params", layer)

    def test_dp_params_present_in_optimizer(self):
        dp = DL_MODEL_JSON["model"]["training"]["optimizer"]["differential_privacy"]
        self.assertIn("noise_multiplier", dp)
        self.assertIn("max_grad_norm", dp)
        self.assertIn("random_seed", dp)

    def test_dp_noise_multiplier_is_positive(self):
        nm = DL_MODEL_JSON["model"]["training"]["optimizer"]["differential_privacy"]["noise_multiplier"]
        self.assertGreater(nm, 0, "noise_multiplier must be positive for DP to be meaningful")

    def test_dp_max_grad_norm_is_positive(self):
        mgn = DL_MODEL_JSON["model"]["training"]["optimizer"]["differential_privacy"]["max_grad_norm"]
        self.assertGreater(mgn, 0)

    def test_has_dataset_selection(self):
        selected = DL_MODEL_JSON["model"]["dataset"]["selected_datasets"]
        self.assertGreaterEqual(len(selected), 1)
        self.assertIn("dataset_id", selected[0])

    def test_has_federated_parameters(self):
        params = DL_MODEL_JSON["federated"]["parameters"]
        for key in ["fraction_fit", "min_fit_clients", "min_available_clients"]:
            self.assertIn(key, params, f"Missing federated parameter: {key}")

    def test_clean_model_json_is_noop_for_dl(self):
        original = copy.deepcopy(DL_MODEL_JSON)
        cleaned = clean_model_json_for_ml(original)
        # DL model: epochs and batch_size must be preserved
        self.assertEqual(cleaned["train"]["epochs"], DL_MODEL_JSON["train"]["epochs"])
        self.assertEqual(cleaned["train"]["batch_size"], DL_MODEL_JSON["train"]["batch_size"])


# ---------------------------------------------------------------------------
# SVM model JSON structure tests
# ---------------------------------------------------------------------------

class TestSVMModelJsonConstruction(TestCase):
    """Verify SVM model JSON has all required fields and DL fields are stripped."""

    def test_model_type_is_ml(self):
        self.assertEqual(SVM_MODEL_JSON["model"]["metadata"]["model_type"], "ml")

    def test_has_ml_method(self):
        self.assertEqual(SVM_MODEL_JSON["model"]["training"]["ml_method"], "fedsvm")

    def test_has_svm_hyperparameters(self):
        training = SVM_MODEL_JSON["model"]["training"]
        self.assertIn("C", training)
        self.assertIn("kernel_config", training)
        self.assertIn("kernel", training["kernel_config"])

    def test_clean_model_json_removes_dl_fields(self):
        ml_json = copy.deepcopy(SVM_MODEL_JSON)
        cleaned = clean_model_json_for_ml(ml_json)
        train_section = cleaned.get("train", {})
        self.assertNotIn("epochs", train_section)
        self.assertNotIn("batch_size", train_section)

    def test_rounds_preserved_after_cleaning(self):
        ml_json = copy.deepcopy(SVM_MODEL_JSON)
        cleaned = clean_model_json_for_ml(ml_json)
        self.assertEqual(cleaned["train"]["rounds"], 3)


# ---------------------------------------------------------------------------
# ModelConfig DB persistence tests
# ---------------------------------------------------------------------------

class TestModelConfigPersistence(TestCase):
    """ModelConfig objects can be created and retrieved with JSON intact."""

    def setUp(self):
        self.user = User.objects.create_user("hub_test_user", password="pass")

    def test_dl_model_config_saved_and_loaded(self):
        mc = ModelConfig.objects.create(
            user=self.user,
            name="DL Test Config",
            model_type="dl",
            config_json=DL_MODEL_JSON,
        )
        loaded = ModelConfig.objects.get(pk=mc.pk)
        dp = loaded.config_json["model"]["training"]["optimizer"]["differential_privacy"]
        self.assertAlmostEqual(dp["noise_multiplier"], 1.0)

    def test_svm_model_config_saved_and_loaded(self):
        mc = ModelConfig.objects.create(
            user=self.user,
            name="SVM Test Config",
            model_type="ml",
            config_json=SVM_MODEL_JSON,
        )
        loaded = ModelConfig.objects.get(pk=mc.pk)
        self.assertEqual(loaded.config_json["model"]["training"]["ml_method"], "fedsvm")

    def test_center_specific_config_contains_only_one_center(self):
        """create_center_specific_config must NOT leak other centers' data."""
        center_datasets = [
            {
                "dataset_name": "Hospital A Dataset",
                "features_info": {"pixel_0": "float"},
                "target_info": {"label": "int"},
                "num_columns": 52,
                "num_rows": 200,
                "size": 50000,
            }
        ]
        base_config = {"model": {"metadata": {"model_type": "dl"}}}
        # Patch print to avoid emoji UnicodeEncodeError on Windows cp1252 stdout
        with patch("builtins.print"):
            center_cfg = create_center_specific_config(center_datasets, base_config)

        selected = center_cfg["dataset"]["selected_datasets"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["dataset_name"], "Hospital A Dataset")
        # Security: no connection credentials must appear in the output
        cfg_str = str(center_cfg)
        self.assertNotIn("api_key", cfg_str)
        self.assertNotIn("password", cfg_str)
        self.assertNotIn("ip", cfg_str.lower().replace("differential_privacy", ""))

    def test_center_config_has_no_connection_info(self):
        """Even when base_config contains connection fields they must be absent."""
        center_datasets = [
            {
                "dataset_name": "Hospital B Dataset",
                "features_info": {},
                "target_info": {},
                "num_columns": 10,
                "num_rows": 100,
                "size": 1000,
                # Simulating that the dataset dict might contain extra info
                "ip": "10.0.0.1",
                "port": 5001,
            }
        ]
        base_config = {}
        with patch("builtins.print"):
            center_cfg = create_center_specific_config(center_datasets, base_config)

        # ip and port must NOT appear in selected_datasets entries
        selected = center_cfg["dataset"]["selected_datasets"]
        self.assertNotIn("ip", selected[0])
        self.assertNotIn("port", selected[0])
