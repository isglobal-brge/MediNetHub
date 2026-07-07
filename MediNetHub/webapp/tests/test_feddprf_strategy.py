"""
Tests for FedDP Random Forest Strategy

Tests cover:
- Serialization/deserialization of trees
- Tree format validation
- Security validation
- Feature bounds extraction
- Strategy initialization and execution
"""

import os
import sys
import django

# Configure Django settings before importing Django models
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medinet.settings')
django.setup()

import unittest
import numpy as np
import pickle
import base64
from unittest.mock import Mock, MagicMock, patch
from django.test import TestCase
from webapp.server_fn.strategies import (
    serialize_trees,
    deserialize_trees,
    validate_tree_format,
    validate_security,
    extract_feature_bounds_from_metadata,
    compute_global_bounds_from_clients,
    FedDPRandomForestStrategy
)


class TestTreeSerialization(TestCase):
    """Tests for tree serialization and deserialization"""

    def setUp(self):
        """Create sample tree state for testing"""
        self.sample_tree = {
            'tree_structure': {
                'type': 'split',
                'feature': 0,
                'threshold': 0.5,
                'left': {
                    'type': 'leaf',
                    'label': 0
                },
                'right': {
                    'type': 'leaf',
                    'label': 1
                }
            },
            'n_classes': 2,
            'max_depth': 10,
            'feature_bounds': {
                'min': [0.0, 0.0, 0.0, 0.0],
                'max': [1.0, 1.0, 1.0, 1.0]
            },
            'epsilon': 0.1
        }

    def test_serialize_empty_list(self):
        """Test serializing empty tree list"""
        result = serialize_trees([])
        self.assertEqual(result.size, 0)
        self.assertEqual(result.dtype, np.uint8)

    def test_serialize_single_tree(self):
        """Test serializing single tree"""
        result = serialize_trees([self.sample_tree])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.uint8)
        self.assertGreater(result.size, 0)

    def test_serialize_multiple_trees(self):
        """Test serializing multiple trees"""
        trees = [self.sample_tree] * 10
        result = serialize_trees(trees)
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(result.size, 0)

    def test_deserialize_empty_array(self):
        """Test deserializing empty array"""
        empty_array = np.array([], dtype=np.uint8)
        result = deserialize_trees(empty_array)
        self.assertEqual(result, [])

    def test_roundtrip_serialization(self):
        """Test serialize -> deserialize roundtrip"""
        trees = [self.sample_tree]
        serialized = serialize_trees(trees)
        deserialized = deserialize_trees(serialized)

        self.assertEqual(len(deserialized), 1)
        self.assertEqual(deserialized[0]['n_classes'], 2)
        self.assertEqual(deserialized[0]['epsilon'], 0.1)
        self.assertEqual(deserialized[0]['tree_structure']['type'], 'split')

    def test_roundtrip_multiple_trees(self):
        """Test roundtrip with multiple trees"""
        trees = [self.sample_tree] * 10
        serialized = serialize_trees(trees)
        deserialized = deserialize_trees(serialized)

        self.assertEqual(len(deserialized), 10)
        for tree in deserialized:
            self.assertEqual(tree['n_classes'], 2)
            self.assertEqual(tree['epsilon'], 0.1)


class TestTreeValidation(TestCase):
    """Tests for tree format validation"""

    def setUp(self):
        """Create valid and invalid tree states"""
        self.valid_tree = {
            'tree_structure': {
                'type': 'split',
                'feature': 0,
                'threshold': 0.5,
                'left': {
                    'type': 'leaf',
                    'label': 0
                },
                'right': {
                    'type': 'leaf',
                    'label': 1
                }
            },
            'n_classes': 2,
            'max_depth': 10,
            'feature_bounds': {
                'min': [0.0, 0.0, 0.0, 0.0],
                'max': [1.0, 1.0, 1.0, 1.0]
            },
            'epsilon': 0.1
        }

    def test_valid_tree(self):
        """Test validation of valid tree"""
        result = validate_tree_format(self.valid_tree)
        self.assertTrue(result)

    def test_missing_required_key(self):
        """Test validation fails with missing key"""
        invalid_tree = self.valid_tree.copy()
        del invalid_tree['n_classes']
        result = validate_tree_format(invalid_tree)
        self.assertFalse(result)

    def test_invalid_tree_structure(self):
        """Test validation fails with invalid structure"""
        invalid_tree = self.valid_tree.copy()
        invalid_tree['tree_structure'] = {'type': 'invalid'}
        result = validate_tree_format(invalid_tree)
        self.assertFalse(result)

    def test_invalid_leaf_node(self):
        """Test validation fails with leaf missing label"""
        invalid_tree = self.valid_tree.copy()
        invalid_tree['tree_structure']['left'] = {'type': 'leaf'}
        result = validate_tree_format(invalid_tree)
        self.assertFalse(result)

    def test_invalid_split_node(self):
        """Test validation fails with split missing threshold"""
        invalid_tree = self.valid_tree.copy()
        invalid_tree['tree_structure'] = {
            'type': 'split',
            'feature': 0,
            'left': {'type': 'leaf', 'label': 0},
            'right': {'type': 'leaf', 'label': 1}
        }
        result = validate_tree_format(invalid_tree)
        self.assertFalse(result)

    def test_invalid_n_classes_type(self):
        """Test validation fails with wrong type for n_classes"""
        invalid_tree = self.valid_tree.copy()
        invalid_tree['n_classes'] = "2"
        result = validate_tree_format(invalid_tree)
        self.assertFalse(result)

    def test_invalid_epsilon_type(self):
        """Test validation fails with wrong type for epsilon"""
        invalid_tree = self.valid_tree.copy()
        invalid_tree['epsilon'] = "0.1"
        result = validate_tree_format(invalid_tree)
        self.assertFalse(result)


class TestSecurityValidation(TestCase):
    """Tests for security validation"""

    def test_valid_payload(self):
        """Test validation passes with valid payload"""
        valid_payload = {
            'trees': [],
            'metadata': {
                'epsilon_spent': 0.1,
                'n_trees': 10
            }
        }
        # Should not raise exception
        try:
            validate_security(valid_payload, 'client_1')
        except ValueError:
            self.fail("validate_security raised ValueError unexpectedly")

    def test_forbidden_field_data(self):
        """Test validation fails with forbidden field 'data'"""
        invalid_payload = {
            'data': [[1, 2, 3], [4, 5, 6]],
            'trees': []
        }
        with self.assertRaises(ValueError) as context:
            validate_security(invalid_payload, 'client_1')
        self.assertIn("Forbidden field 'data'", str(context.exception))

    def test_forbidden_field_X_train(self):
        """Test validation fails with forbidden field 'X_train'"""
        invalid_payload = {
            'X_train': [[1, 2, 3]],
            'trees': []
        }
        with self.assertRaises(ValueError) as context:
            validate_security(invalid_payload, 'client_1')
        self.assertIn("Forbidden field 'X_train'", str(context.exception))

    def test_n_samples_warning(self):
        """Test that n_samples is removed but doesn't raise error"""
        payload = {
            'trees': [],
            'metadata': {
                'n_samples': 1000,
                'epsilon_spent': 0.1
            }
        }
        # Should not raise exception, but should remove n_samples
        validate_security(payload, 'client_1')
        self.assertNotIn('n_samples', payload['metadata'])

    def test_bounds_warning(self):
        """Test that bounds are removed but don't raise error"""
        payload = {
            'trees': [],
            'metadata': {
                'bounds': {'min': [0], 'max': [1]},
                'epsilon_spent': 0.1
            }
        }
        # Should not raise exception, but should remove bounds
        validate_security(payload, 'client_1')
        self.assertNotIn('bounds', payload['metadata'])


class TestFeatureBounds(TestCase):
    """Tests for feature bounds extraction and aggregation"""

    def test_extract_bounds_from_metadata(self):
        """Test extracting bounds from dataset metadata"""
        metadata = {
            'dataset_characteristics': {
                'feature_ranges': {
                    'feature_0': {'min': 0.0, 'max': 1.0},
                    'feature_1': {'min': -1.0, 'max': 1.0},
                    'feature_2': {'min': 0.0, 'max': 10.0},
                    'feature_3': {'min': -5.0, 'max': 5.0}
                }
            }
        }

        result = extract_feature_bounds_from_metadata(metadata)

        expected = {
            'min': [0.0, -1.0, 0.0, -5.0],
            'max': [1.0, 1.0, 10.0, 5.0]
        }

        self.assertEqual(result, expected)

    def test_extract_bounds_missing_characteristics(self):
        """Test error when dataset_characteristics missing"""
        metadata = {}
        with self.assertRaises(ValueError) as context:
            extract_feature_bounds_from_metadata(metadata)
        self.assertIn("dataset_characteristics", str(context.exception))

    def test_extract_bounds_missing_feature_ranges(self):
        """Test error when feature_ranges missing"""
        metadata = {
            'dataset_characteristics': {}
        }
        with self.assertRaises(ValueError) as context:
            extract_feature_bounds_from_metadata(metadata)
        self.assertIn("feature_ranges", str(context.exception))

    def test_compute_global_bounds_single_client(self):
        """Test global bounds with single client"""
        metadata = {
            'dataset_characteristics': {
                'feature_ranges': {
                    'feature_0': {'min': 0.0, 'max': 1.0},
                    'feature_1': {'min': 0.0, 'max': 1.0}
                }
            }
        }
        client_datasets = {'client_1': metadata}

        result = compute_global_bounds_from_clients(client_datasets)

        expected = {
            'min': [0.0, 0.0],
            'max': [1.0, 1.0]
        }

        self.assertEqual(result, expected)

    def test_compute_global_bounds_multiple_clients(self):
        """Test global bounds aggregation across multiple clients"""
        client_datasets = {
            'client_1': {
                'dataset_characteristics': {
                    'feature_ranges': {
                        'feature_0': {'min': 0.0, 'max': 1.0},
                        'feature_1': {'min': 0.0, 'max': 5.0}
                    }
                }
            },
            'client_2': {
                'dataset_characteristics': {
                    'feature_ranges': {
                        'feature_0': {'min': -1.0, 'max': 2.0},
                        'feature_1': {'min': -2.0, 'max': 3.0}
                    }
                }
            },
            'client_3': {
                'dataset_characteristics': {
                    'feature_ranges': {
                        'feature_0': {'min': -0.5, 'max': 1.5},
                        'feature_1': {'min': -1.0, 'max': 4.0}
                    }
                }
            }
        }

        result = compute_global_bounds_from_clients(client_datasets)

        # Global min = min across all clients, Global max = max across all clients
        expected = {
            'min': [-1.0, -2.0],
            'max': [2.0, 5.0]
        }

        self.assertEqual(result, expected)

    def test_compute_global_bounds_empty_clients(self):
        """Test error when client_datasets is empty"""
        with self.assertRaises(ValueError) as context:
            compute_global_bounds_from_clients({})
        self.assertIn("cannot be empty", str(context.exception))


class TestFedDPRandomForestStrategy(TestCase):
    """Tests for FedDPRandomForestStrategy"""

    def setUp(self):
        """Create mock server manager and config"""
        self.mock_job = Mock()
        self.mock_job.id = 1
        self.mock_job.status = 'pending'
        self.mock_job.total_rounds = 5
        self.mock_job.clients_status = {}

        self.mock_server_manager = Mock()
        self.mock_server_manager.job = self.mock_job
        self.mock_server_manager.model_config = {
            'metadata': {
                'model_type': 'ml',
                'framework': 'sklearn'
            },
            'algorithm': {
                'ml_algorithm': {
                    'type': 'dp_random_forest',
                    'hyperparameters': {
                        'epsilon_total': 1.0,
                        'max_depth': 10,
                        'n_trees_per_client': 10,
                        'min_samples_split': 2
                    }
                }
            }
        }

    def test_strategy_initialization(self):
        """Test strategy initialization"""
        strategy = FedDPRandomForestStrategy(
            server_manager=self.mock_server_manager,
            fraction_fit=1.0,
            min_fit_clients=2
        )

        self.assertEqual(strategy.epsilon_total, 1.0)
        self.assertEqual(strategy.max_depth, 10)
        self.assertEqual(strategy.n_trees_per_client, 10)
        self.assertEqual(strategy.global_forest, [])
        self.assertFalse(strategy.convergence_flag)

    def test_num_fit_clients(self):
        """Test num_fit_clients calculation"""
        strategy = FedDPRandomForestStrategy(
            server_manager=self.mock_server_manager,
            fraction_fit=0.5,
            min_fit_clients=2
        )

        num_clients, min_clients = strategy.num_fit_clients(10)

        self.assertEqual(num_clients, 5)
        self.assertEqual(min_clients, strategy.min_available_clients)

    def test_aggregate_metrics(self):
        """Test metrics aggregation"""
        strategy = FedDPRandomForestStrategy(
            server_manager=self.mock_server_manager
        )

        mock_fit_res_1 = Mock()
        mock_fit_res_1.num_examples = 100
        mock_fit_res_1.metrics = {
            'accuracy': 0.9,
            'loss': 0.1,
            'precision': 0.85,
            'recall': 0.88,
            'f1': 0.86
        }

        mock_fit_res_2 = Mock()
        mock_fit_res_2.num_examples = 200
        mock_fit_res_2.metrics = {
            'accuracy': 0.85,
            'loss': 0.15,
            'precision': 0.80,
            'recall': 0.82,
            'f1': 0.81
        }

        results = [
            (Mock(), mock_fit_res_1),
            (Mock(), mock_fit_res_2)
        ]

        aggregated = strategy._aggregate_metrics(results)

        expected_accuracy = (100 * 0.9 + 200 * 0.85) / 300
        self.assertAlmostEqual(aggregated['accuracy'], expected_accuracy, places=4)

    def test_get_privacy_report(self):
        """Test privacy report generation"""
        strategy = FedDPRandomForestStrategy(
            server_manager=self.mock_server_manager
        )

        strategy.global_forest = [{'tree': 'dummy'}] * 50

        report = strategy.get_privacy_report()

        self.assertEqual(report['epsilon_total'], 1.0)
        self.assertEqual(report['total_trees'], 50)
        self.assertEqual(report['data_shared'], 'NONE - only DP models shared')


class TestEpsilonValidation(TestCase):
    """Tests for epsilon validation constraints"""

    def setUp(self):
        """Create mock server manager with various epsilon values"""
        self.mock_job = Mock()
        self.mock_job.id = 1
        self.mock_job.status = 'pending'
        self.mock_job.total_rounds = 5
        self.mock_job.clients_status = {}

    def create_server_manager_with_epsilon(self, epsilon_value):
        """Helper to create server manager with specific epsilon"""
        mock_server_manager = Mock()
        mock_server_manager.job = self.mock_job
        mock_server_manager.model_config = {
            'metadata': {
                'model_type': 'ml',
                'framework': 'sklearn'
            },
            'algorithm': {
                'ml_algorithm': {
                    'type': 'dp_random_forest',
                    'hyperparameters': {
                        'epsilon_total': epsilon_value,
                        'max_depth': 10,
                        'n_trees_per_client': 10,
                        'min_samples_split': 2
                    }
                }
            }
        }
        return mock_server_manager

    def test_epsilon_too_low(self):
        """Test that epsilon < 0.1 raises ValueError"""
        server_manager = self.create_server_manager_with_epsilon(0.05)

        with self.assertRaises(ValueError) as context:
            FedDPRandomForestStrategy(server_manager=server_manager)

        self.assertIn("must be at least 0.1", str(context.exception))
        self.assertIn("meaningful privacy guarantees", str(context.exception))

    def test_epsilon_zero(self):
        """Test that epsilon = 0 raises ValueError"""
        server_manager = self.create_server_manager_with_epsilon(0.0)

        with self.assertRaises(ValueError) as context:
            FedDPRandomForestStrategy(server_manager=server_manager)

        self.assertIn("must be at least 0.1", str(context.exception))

    def test_epsilon_negative(self):
        """Test that negative epsilon raises ValueError"""
        server_manager = self.create_server_manager_with_epsilon(-0.5)

        with self.assertRaises(ValueError) as context:
            FedDPRandomForestStrategy(server_manager=server_manager)

        self.assertIn("must be at least 0.1", str(context.exception))

    def test_epsilon_too_high(self):
        """Test that epsilon > 10.0 raises ValueError"""
        server_manager = self.create_server_manager_with_epsilon(15.0)

        with self.assertRaises(ValueError) as context:
            FedDPRandomForestStrategy(server_manager=server_manager)

        self.assertIn("exceeds maximum safe value", str(context.exception))
        self.assertIn("10.0", str(context.exception))

    def test_epsilon_minimum_valid(self):
        """Test that epsilon = 0.1 is accepted"""
        server_manager = self.create_server_manager_with_epsilon(0.1)

        try:
            strategy = FedDPRandomForestStrategy(server_manager=server_manager)
            self.assertEqual(strategy.epsilon_total, 0.1)
        except ValueError:
            self.fail("epsilon = 0.1 should be valid")

    def test_epsilon_maximum_valid(self):
        """Test that epsilon = 10.0 is accepted"""
        server_manager = self.create_server_manager_with_epsilon(10.0)

        try:
            strategy = FedDPRandomForestStrategy(server_manager=server_manager)
            self.assertEqual(strategy.epsilon_total, 10.0)
        except ValueError:
            self.fail("epsilon = 10.0 should be valid")

    def test_epsilon_recommended_range(self):
        """Test that recommended epsilon values (0.5-2.0) work"""
        for epsilon in [0.5, 1.0, 1.5, 2.0]:
            server_manager = self.create_server_manager_with_epsilon(epsilon)

            try:
                strategy = FedDPRandomForestStrategy(server_manager=server_manager)
                self.assertEqual(strategy.epsilon_total, epsilon)
            except ValueError:
                self.fail(f"epsilon = {epsilon} should be valid")

    def test_epsilon_high_risk_warning(self):
        """Test that epsilon >= 5.0 logs a warning"""
        server_manager = self.create_server_manager_with_epsilon(6.0)

        with self.assertLogs('webapp.server_fn.strategies', level='WARNING') as log:
            strategy = FedDPRandomForestStrategy(server_manager=server_manager)
            self.assertEqual(strategy.epsilon_total, 6.0)

            self.assertTrue(any("High epsilon value detected" in message for message in log.output))
            self.assertTrue(any("limited privacy protection" in message for message in log.output))


if __name__ == '__main__':
    unittest.main()
