import numpy as np
from torch import save, tensor, load
import json
import os
from typing import List, Tuple, Union, Optional, Dict
import torch
import traceback
from flwr.common import Metrics, FitRes, Parameters, NDArrays, FitIns, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.client_proxy import ClientProxy
from collections import OrderedDict
from django.utils import timezone
from .model_builder import DynamicModel, SequentialModel
from sklearn.svm import SVC
from sklearn.kernel_approximation import RBFSampler
import logging
import pickle
import base64

logger = logging.getLogger(__name__)

# Epsilon validation constants for differential privacy
MIN_EPSILON = 0.1  # Minimum epsilon for meaningful privacy guarantees
MAX_EPSILON = 10.0  # Maximum safe epsilon value
HIGH_RISK_EPSILON = 5.0  # Warning threshold for high privacy risk


def update_client_tracking(server_manager, server_round, results):
    """
    Actualizar métricas básicas usando client_id incluido en las métricas del cliente
    
    Args:
        server_manager: ServerManager instance (tiene job con mappings)
        server_round (int): Número del round actual  
        results: Lista de (ClientProxy, FitRes) de Flower
    """
    try:
        clients_status = server_manager.job.clients_status or {}
        
        print(f"🔄 CLIENT_TRACKING: Round {server_round} - Processing {len(results)} client results")
        print(f"📋 Available mappings: {list(clients_status.keys())}")
        
        if not clients_status:
            print(f"❌ WARNING: No client mappings in job {server_manager.job.id}")
            logger.warning(f"No client mappings in job {server_manager.job.id}")
            return
        
        # Actualizar cada cliente que devolvió resultados
        for i, (client_proxy, fit_res) in enumerate(results):
            # Obtener client_id desde las métricas del cliente
            client_id = fit_res.metrics.get('client_id')
            
            print(f"📥 Client {i+1}: client_id='{client_id}', metrics_keys={list(fit_res.metrics.keys())}")
            
            if client_id and client_id in clients_status:
                # Actualizar métricas ACTUALES (estado current)
                old_status = clients_status[client_id]['status']
                old_round = clients_status[client_id]['current_round']
                
                clients_status[client_id].update({
                    'current_round': server_round,
                    'status': 'training',
                    'accuracy': fit_res.metrics.get('accuracy', 0),
                    'loss': fit_res.metrics.get('loss', 0),
                    'precision': fit_res.metrics.get('precision', 0),
                    'recall': fit_res.metrics.get('recall', 0),
                    'f1': fit_res.metrics.get('f1', 0),
                    'train_samples': fit_res.metrics.get('train_samples', 0),
                    'last_seen': timezone.now().isoformat()
                })
                
                # Agregar al histórico por rounds (escalable para futuro)
                if 'rounds_history' not in clients_status[client_id]:
                    clients_status[client_id]['rounds_history'] = {}
                    
                clients_status[client_id]['rounds_history'][str(server_round)] = {
                    'accuracy': fit_res.metrics.get('accuracy', 0),
                    'loss': fit_res.metrics.get('loss', 0),
                    'precision': fit_res.metrics.get('precision', 0),
                    'recall': fit_res.metrics.get('recall', 0),
                    'f1': fit_res.metrics.get('f1', 0),
                    'train_samples': fit_res.metrics.get('train_samples', 0),
                    'timestamp': timezone.now().isoformat()
                }
                
                print(f"✅ UPDATED: {client_id} → {clients_status[client_id]['connection_name']} | {old_status}→training | Round {old_round}→{server_round} | Acc: {fit_res.metrics.get('accuracy', 0):.3f}")
                logger.info(f"Updated client {client_id} (→ {clients_status[client_id]['connection_name']}) - Round {server_round}")
            else:
                print(f"❌ MISSING: Client ID '{client_id}' not found in mappings {list(clients_status.keys())}")
                logger.warning(f"Client ID {client_id} not found in mappings")
        
        # Una sola escritura a BD por round
        print(f"💾 SAVING: clients_status to database for job {server_manager.job.id}")
        server_manager.job.clients_status = clients_status
        server_manager.job.save(update_fields=['clients_status'])
        print(f"✅ SAVED: clients_status updated successfully")
        
    except Exception as e:
        print(f"❌ ERROR in update_client_tracking: {e}")
        logger.error(f"Error updating client tracking: {e}")



class ServerManager:
    def __init__(self, training_job, model_config):
        print(f"ServerManager.__init__ called")
        print(f"training_job: {training_job}")
        print(f"model_config type: {type(model_config)}")
        print(f"model_config loaded with framework: {model_config.get('framework', 'unknown')}")
        
        self.job = training_job
        self.model_config = model_config
        self.framework = model_config.get('framework', 'pytorch')
        self.should_stop = False  # Flag to signal server shutdown
        
        print(f"Framework detected: '{self.framework}'")
        
        self.net: Optional[DynamicModel] = None
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the model based on configuration"""
        print(f"initialize_model called with framework: '{self.framework}'")

        if self.framework in ['pt', 'pytorch']:
            # Legacy structure: model_config.model.architecture.layers
            layers_config = self.model_config.get('architecture', {}).get('layers', [])
            if not layers_config:
                layers_config = self.model_config.get('model', {}).get('architecture', {}).get('layers', [])

            # Check model type from both possible locations
            model_type = self.model_config.get('metadata', {}).get('model_type', '')
            if not model_type:
                model_type = self.model_config.get('model', {}).get('metadata', {}).get('model_type', '')


            if model_type == 'dl_linear':
                print("Starting SequentialModel")
                self.net = SequentialModel({'layers': layers_config})

            else:
                self.net = DynamicModel({"layers": layers_config})
            
            print(f"Model created. State dict keys: {list(self.net.state_dict().keys())}")
            print(f"Model parameters count: {sum(p.numel() for p in self.net.parameters())}")
            
        else:
            print(f"Framework '{self.framework}' not supported")
            raise ValueError(f"Framework {self.framework} not supported. Supported: 'pt', 'pytorch'")
            
    def save_metrics(self, metrics: Dict, round_number: int, round_start_time=None, round_end_time=None, clients_info=None):
        """Save metrics to the training job with timing information"""
        if not self.job.metrics_json:
            current_metrics = []
        else:
            current_metrics = json.loads(self.job.metrics_json)
        
        # Create comprehensive metrics dict (matching simulation structure)
        round_metrics = {
            'round': round_number,
            'timestamp': timezone.now().isoformat(),
            'accuracy': metrics.get('accuracy', 0),
            'loss': metrics.get('loss', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0),
            'clients': clients_info.get('active_clients', 0) if clients_info else 0,
        }
        
        # Add timing information if available
        if round_start_time and round_end_time:
            round_metrics.update({
                'round_start': round_start_time.isoformat(),
                'round_end': round_end_time.isoformat(),
                'round_elapsed': (round_end_time - round_start_time).total_seconds()
            })
        
        current_metrics.append(round_metrics)
        
        self.job.metrics_json = json.dumps(current_metrics)
        self.job.current_round = round_number
        self.job.progress = int((round_number / self.job.total_rounds) * 100)
        
        # Update job metrics (latest values)
        self.job.accuracy = round_metrics['accuracy']
        self.job.loss = round_metrics['loss']
        
        self.job.save()
        
        print(f"Saved metrics for round {round_number}: accuracy={round_metrics['accuracy']:.3f}, loss={round_metrics['loss']:.3f}")
    
    def update_client_status(self, client_info: Dict):
        """Update client status information with comprehensive metrics structure"""
        try:
            current_clients = self.job.clients_status or {}
            
            # Update or add client info
            for client_id, info in client_info.items():
                # Initialize client if doesn't exist
                if client_id not in current_clients:
                    current_clients[client_id] = {
                        'name': info.get('name', f'Client {client_id}'),
                        'ip': info.get('ip', 'unknown'),
                        'status': info.get('status', 'active'),
                        'train_samples': info.get('train_samples', 0),
                        'test_samples': info.get('test_samples', 0),
                        'last_seen': timezone.now().isoformat(),
                        'current_metrics': self._get_empty_metrics_structure(),
                        'metrics_history': []
                    }
                
                # Update only dynamic info (status, last_seen, metrics)
                current_clients[client_id]['status'] = info.get('status', current_clients[client_id]['status'])
                current_clients[client_id]['last_seen'] = timezone.now().isoformat()
                
                # Update current metrics
                if 'metrics' in info:
                    self._update_client_metrics(current_clients[client_id], info['metrics'], self.job.current_round)
            
            self.job.clients_status = current_clients
            self.job.save()
            
            print(f"📡 Updated client status: {len(current_clients)} clients")
            
        except Exception as e:
            print(f"❌ Error updating client status: {e}")
    
    def _get_empty_metrics_structure(self):
        """Get empty metrics structure with all possible metrics"""
        return {
            # Basic metrics (always present)
            'accuracy': 0.0,
            'loss': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            
            # Specialized metrics (null if not configured)
            'segmentation': {
                'mean_iou': None,
                'dice_score': None,
                'hausdorff_distance': None
            },
            'medical': {
                'sensitivity': None,
                'specificity': None,
                'npv': None
            },
            'detection': {
                'auc_roc': None,
                'auc_pr': None,
                'mean_ap': None
            },
            'regression': {
                'mae': None,
                'rmse': None,
                'mape': None
            }
        }
    
    def _update_client_metrics(self, client_data: Dict, metrics: Dict, round_number: int = 0):
        """Update client metrics and add to history"""
        if 'current_metrics' not in client_data:
            client_data['current_metrics'] = self._get_empty_metrics_structure()
        if 'metrics_history' not in client_data:
            client_data['metrics_history'] = []

        # Update current metrics
        current_metrics = client_data['current_metrics']
        
        # Update basic metrics
        for metric in ['accuracy', 'loss', 'precision', 'recall', 'f1']:
            if metric in metrics:
                current_metrics[metric] = metrics[metric]
        
        # Update specialized metrics
        specialized_metrics = {
            'segmentation': ['mean_iou', 'dice_score', 'hausdorff_distance'],
            'medical': ['sensitivity', 'specificity', 'npv'],
            'detection': ['auc_roc', 'auc_pr', 'mean_ap'],
            'regression': ['mae', 'rmse', 'mape']
        }
        
        for category, metric_names in specialized_metrics.items():
            for metric_name in metric_names:
                if metric_name in metrics:
                    current_metrics[category][metric_name] = metrics[metric_name]
        
        # Add to history
        history_entry = {
            'round': round_number,
            'timestamp': timezone.now().isoformat(),
            **current_metrics  # Include all current metrics in history
        }
        
        client_data['metrics_history'].append(history_entry)
        
        # Keep only last 50 entries to avoid excessive data
        if len(client_data['metrics_history']) > 50:
            client_data['metrics_history'] = client_data['metrics_history'][-50:]
        
    def save_model(self, parameters: List[np.ndarray], round_number: int):
        """Save model parameters to the training job"""
        try:
            if self.framework in ['pt', 'pytorch']: 
                if self.net is None:
                    raise ValueError("Model is not initialized")
                
                params_dict = zip(self.net.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: tensor(v) for k, v in params_dict})
                self.net.load_state_dict(state_dict, strict=True)
                
                # Save model to file and update path in database
                model_path = f"models/model_round_{round_number}_job_{self.job.id}.pth"
                
                # Create models directory if it doesn't exist
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                # Save the model
                save(self.net.state_dict(), model_path)
                
                # Update the model file path in the database
                self.job.model_file_path = model_path
                self.job.save()
                
                print(f"✅ Model saved successfully for round {round_number}")
                
            else:
                raise ValueError(f"Framework {self.framework} not supported")
                
        except Exception as e:
            print(f"❌ Error saving model for round {round_number}: {e}")
            
            # Mark training as failed
            self.job.status = 'failed'
            self.job.logs = f"Model save failed at round {round_number}: {str(e)}"
            self.job.save()
            
            print(f"❌ Training job marked as FAILED due to model save error")
            
            # Re-raise the exception to stop training
            raise e


class FedAvgModelStrategy(FedAvg):
    def __init__(
        self,
        server_manager: ServerManager,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.server_manager = server_manager
        self.round_start_time = None
        self.training_start_time = None
        self.current_round = 0  
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict]:
        """Aggregate model weights and save checkpoint"""
        
        self.current_round = server_round
        
        round_end_time = timezone.now()
        
        # Update status to 'running' when first round starts
        if server_round == 1 and self.server_manager.job.status == 'server_ready':
            print(f"Round 1 started - updating status to 'running'")
            self.server_manager.job.status = 'running'
            self.server_manager.job.logs = f"Federated training started - Round 1 of {self.server_manager.job.total_rounds}"
            self.server_manager.job.started_at = timezone.now()
            self.training_start_time = self.server_manager.job.started_at
            self.server_manager.job.save()
        
        # Set round start time if not set (for first call)
        if self.round_start_time is None:
            self.round_start_time = self.server_manager.job.started_at or timezone.now()
        
        try:
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        except ZeroDivisionError as e:
            print(f"❌ ZeroDivisionError in aggregate_fit: {e}")
            print(f"❌ This usually means clients are not reporting num_examples correctly")
            print(f"❌ Results: {[(client_proxy.cid if hasattr(client_proxy, 'cid') else 'unknown', fit_res.num_examples) for client_proxy, fit_res in results]}")
            
            # Mark training as failed
            self.server_manager.job.status = 'failed'
            self.server_manager.job.logs = f"Training failed at round {server_round}: Division by zero error - clients not reporting examples correctly"
            self.server_manager.job.save()
            
            # Return None to stop training
            return None, {}
        except Exception as e:
            print(f"❌ Unexpected error in aggregate_fit: {e}")
            
            # Mark training as failed
            self.server_manager.job.status = 'failed'
            self.server_manager.job.logs = f"Training failed at round {server_round}: {str(e)}"
            self.server_manager.job.save()
            
            # Return None to stop training
            return None, {}
        
        print(f"DEBUG aggregate_fit - Round {server_round} completed")
        print(f"DEBUG results: {len(results)} success, {len(failures)} failures")
        
        if aggregated_parameters is not None:
            # Save model
            parameters_as_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
            self.server_manager.save_model(parameters_as_ndarrays, server_round)            
            print(f"✅ Model saved for round {server_round}")
        
        update_client_tracking(self.server_manager, server_round, results)
        
        # Check if this is the last round - signal completion and calculate total training time
        if server_round >= self.server_manager.job.total_rounds:
            print(f"🏁 Final round {server_round} completed - marking job as completed")
            
            # Calculate total training duration
            if self.training_start_time:
                total_duration = (round_end_time - self.training_start_time).total_seconds()
                self.server_manager.job.training_duration = total_duration
            
            self.server_manager.job.status = 'completed'
            self.server_manager.job.completed_at = round_end_time
            self.server_manager.job.progress = 100  # Assegurar progress 100%
            self.server_manager.job.save()
            
            # Signal server to stop
            self.server_manager.should_stop = True
        else:
            # Prepare for next round
            self.round_start_time = round_end_time

        return aggregated_parameters, aggregated_metrics


class FedSVMStrategy(Strategy):
    """
    Custom Flower strategy for FedSVM (Support Vector Machine Federation)
    Implements the OptMD (Optimized Multiple Deltas) variant
    """

    def __init__(
        self,
        server_manager: ServerManager,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.3,
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
        **kwargs
    ):
        super().__init__()
        self.server_manager = server_manager
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        self.round_start_time = None
        self.training_start_time = None
        self.current_round = 0

        # FedSVM specific attributes
        self.support_vectors_pool = {}  # Maps client_id -> list of (sha, sv, label)
        self.convergence_threshold = server_manager.model_config.get('algorithm', {}).get('ml_algorithm', {}).get('hyperparameters', {}).get('server_eps', 1e-2)
        self.converged = False  # Flag to signal convergence

        print(f"FedSVMStrategy initialized with convergence threshold: {self.convergence_threshold}")

    def initialize_parameters(self, client_manager):
        """Initialize global model parameters (not used in FedSVM)"""
        return None

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure the next round of federated training"""
        

        print(f"🔧 FedSVM configure_fit - Round {server_round}")

        # Sample clients for this round
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create fit configuration for each client
        fit_configurations = []
        for client in clients:
            # Get support vectors from other clients (excluding current client)
            other_svs_dict = self._get_support_vectors_for_client(client.cid)

            # Convert support vectors to ndarrays for Flower serialization
            if other_svs_dict:
                # Aggregate all SVs from different clients into single arrays
                all_svs = []
                all_labels = []

                # other_svs_dict format: {client_id: {'support_vectors': [...], 'labels': [...], 'shas': [...]}}
                for other_client_id, sv_data in other_svs_dict.items():
                    support_vectors = sv_data['support_vectors']
                    labels = sv_data['labels']

                    # Add all SVs from this client
                    all_svs.extend(support_vectors)
                    all_labels.extend(labels)

                if all_svs:
                    # Convert to numpy arrays
                    svs_array = np.array(all_svs, dtype=np.float32)
                    labels_array = np.array(all_labels, dtype=np.float32)

                    # Pack into Parameters
                    params = ndarrays_to_parameters([svs_array, labels_array])
                    print(f"📤 Sending {len(all_svs)} SVs to client {client.cid}")
                else:
                    params = Parameters(tensors=[], tensor_type="")
            else:
                # No support vectors to send (first round)
                params = Parameters(tensors=[], tensor_type="")
                print(f"📤 No SVs to send to client {client.cid} (first round or no other clients)")

            # Config only contains scalar metadata
            config = {
                "server_round": server_round,
                "converged": int(self.converged),  # 1 if converged, 0 otherwise
            }

            # Create FitIns object
            fit_ins = FitIns(
                parameters=params,
                config=config
            )

            fit_configurations.append((client, fit_ins))

        if self.converged:
            print(f"Configured {len(fit_configurations)} clients for round {server_round} (CONVERGED - no-op mode)")
        else:
            print(f"Configured {len(fit_configurations)} clients for round {server_round}")
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict]:
        """Aggregate support vectors from clients"""

        self.current_round = server_round
        round_end_time = timezone.now()

        # Update status to 'running' when first round starts
        if server_round == 1 and self.server_manager.job.status == 'server_ready':
            print(f"Round 1 started - updating status to 'running'")
            self.server_manager.job.status = 'running'
            self.server_manager.job.logs = f"FedSVM training started - Round 1 of {self.server_manager.job.total_rounds}"
            self.server_manager.job.started_at = timezone.now()
            self.training_start_time = self.server_manager.job.started_at
            self.server_manager.job.save()

        if self.round_start_time is None:
            self.round_start_time = self.server_manager.job.started_at or timezone.now()

        print(f"🔄 FedSVM aggregate_fit - Round {server_round}")
        print(f"📊 Received {len(results)} results, {len(failures)} failures")

        if failures:
            print(f"🚨 ===== FAILURE DETAILS FOR ROUND {server_round} =====")
            for i, failure in enumerate(failures):
                print(f"❌ Failure {i+1}/{len(failures)}:")

                # Check if it's a tuple (ClientProxy, Exception) or just Exception
                if isinstance(failure, tuple):
                    client_proxy, exception = failure
                    print(f"   Client CID: {client_proxy.cid}")
                    print(f"   Exception Type: {type(exception).__name__}")
                    print(f"   Exception Message: {str(exception)}")

                    # Print full traceback if available
                    if hasattr(exception, '__traceback__'):
                        print(f"   Traceback:")
                        traceback.print_exception(type(exception), exception, exception.__traceback__)
                else:
                    # It's a BaseException
                    print(f"   Exception Type: {type(failure).__name__}")
                    print(f"   Exception Message: {str(failure)}")

                    if hasattr(failure, '__traceback__'):
                        print(f"   Traceback:")
                        traceback.print_exception(type(failure), failure, failure.__traceback__)


        if not results:
            print("❌ No results to aggregate")
            return None, {}

        # Collect support vectors from all clients
        new_svs_count = 0
        total_svs_exchanged = 0

        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get('client_id', client_proxy.cid)

            # Extract support vectors from parameters
            # Format: [svs_array, labels_array, shas_array]
            if fit_res.parameters:
                svs_data = parameters_to_ndarrays(fit_res.parameters)

                if len(svs_data) >= 3:
                    svs = svs_data[0]  # Support vectors
                    labels = svs_data[1]  # Labels
                    shas = svs_data[2]  # SHA hashes

                    # Initialize client pool if needed
                    if client_id not in self.support_vectors_pool:
                        self.support_vectors_pool[client_id] = []

                    # Add new support vectors to pool
                    existing_shas = set([sha for sha, _, _ in self.support_vectors_pool[client_id]])

                    for i in range(len(svs)):
                        sha = shas[i] if i < len(shas) else f"sha_{i}"
                        if sha not in existing_shas:
                            self.support_vectors_pool[client_id].append((sha, svs[i], labels[i]))
                            new_svs_count += 1

                    total_svs_exchanged += len(svs)

                    print(f"✅ Client {client_id}: received {len(svs)} SVs ({new_svs_count} new)")

        print(f"📦 Total SVs in pool: {sum(len(svs) for svs in self.support_vectors_pool.values())}")

        # Update client tracking
        update_client_tracking(self.server_manager, server_round, results)

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(results)

        # Save metrics
        clients_info = {'active_clients': len(results)}
        self.server_manager.save_metrics(
            aggregated_metrics,
            server_round,
            round_start_time=self.round_start_time,
            round_end_time=round_end_time,
            clients_info=clients_info
        )

        # Check convergence
        if new_svs_count == 0 and not self.converged:
            print(f"🏁 Convergence achieved - no new support vectors")
            print(f" Setting converged flag - remaining rounds will be no-ops")

            # Set convergence flag (clients will skip training in next rounds)
            self.converged = True

            # Mark job as completed
            self.server_manager.job.status = 'completed'
            self.server_manager.job.completed_at = round_end_time
            self.server_manager.job.progress = 100

            if self.training_start_time:
                total_duration = (round_end_time - self.training_start_time).total_seconds()
                self.server_manager.job.training_duration = total_duration

            self.server_manager.job.save()

            # Note: Don't return None here - let Flower continue to complete remaining rounds
            # Clients will receive converged=1 flag and skip training

        # Check if final round
        if server_round >= self.server_manager.job.total_rounds:
            print(f"🏁 Final round {server_round} completed")

            if self.training_start_time:
                total_duration = (round_end_time - self.training_start_time).total_seconds()
                self.server_manager.job.training_duration = total_duration

            self.server_manager.job.status = 'completed'
            self.server_manager.job.completed_at = round_end_time
            self.server_manager.job.progress = 100
            self.server_manager.job.save()

            self.server_manager.should_stop = True
        else:
            self.round_start_time = round_end_time

        # Return empty parameters (FedSVM doesn't aggregate model weights)
        return None, aggregated_metrics

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Configure evaluation (optional for FedSVM)"""
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict]:
        """Aggregate evaluation results"""
        if not results:
            return None, {}

        aggregated_metrics = self._aggregate_metrics(results)
        loss = aggregated_metrics.get('loss', 0.0)

        return loss, aggregated_metrics

    def evaluate(self, server_round: int, parameters):
        """Evaluate global model (not used in FedSVM)"""
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for fitting"""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for evaluation"""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def _get_support_vectors_for_client(self, client_id: str) -> Dict:
        """Get support vectors from all OTHER clients (excluding current client)"""
        other_svs = {}

        for cid, svs_list in self.support_vectors_pool.items():
            if cid != client_id:  # Exclude current client's own SVs
                other_svs[cid] = {
                    'support_vectors': [sv for _, sv, _ in svs_list],
                    'labels': [label for _, _, label in svs_list],
                    'shas': [sha for sha, _, _ in svs_list]
                }

        return other_svs

    def _aggregate_metrics(self, results: List[Tuple[ClientProxy, FitRes]]) -> Dict:
        """Aggregate metrics from multiple clients (weighted average)"""
        if not results:
            return {}

        total_examples = 0
        weighted_metrics = {
            'accuracy': 0.0,
            'loss': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
        }

        for _, fit_res in results:
            num_examples = fit_res.num_examples
            total_examples += num_examples

            for metric in weighted_metrics.keys():
                weighted_metrics[metric] += num_examples * fit_res.metrics.get(metric, 0.0)

        if total_examples > 0:
            for metric in weighted_metrics.keys():
                weighted_metrics[metric] /= total_examples

        print(f"📊 Aggregated metrics: {weighted_metrics}")
        return weighted_metrics


# ==================== FedDP Random Forest Helpers ====================

def serialize_trees(trees: List[dict]) -> np.ndarray:
    """
    Serializa lista de estados de árboles para transmisión.

    Args:
        trees: Lista de diccionarios con estados de árboles

    Returns:
        Numpy array con árboles serializados
    """
    if not trees:
        return np.array([], dtype=np.uint8)

    # 1. Pickle dumps
    pickled = pickle.dumps(trees)

    # 2. Base64 encode
    encoded = base64.b64encode(pickled)

    # 3. Convertir a numpy array
    serialized_array = np.frombuffer(encoded, dtype=np.uint8)

    return serialized_array


def deserialize_trees(serialized_array: np.ndarray) -> List[dict]:
    """
    Deserializa árboles desde transmisión.

    Args:
        serialized_array: Numpy array con árboles serializados

    Returns:
        Lista de diccionarios con estados de árboles
    """
    if serialized_array.size == 0:
        return []

    # 1. Convertir a bytes
    encoded_bytes = serialized_array.tobytes()

    # 2. Base64 decode
    pickled = base64.b64decode(encoded_bytes)

    # 3. Unpickle
    trees = pickle.loads(pickled)

    return trees


def validate_tree_format(tree_state: dict) -> bool:
    """
    Valida que un estado de árbol tiene el formato esperado.

    Args:
        tree_state: Diccionario con estado de árbol

    Returns:
        True si válido, False si inválido
    """
    required_keys = {'tree_structure', 'n_classes', 'max_depth', 'feature_bounds', 'epsilon'}

    # 1. Verificar claves principales
    if not all(key in tree_state for key in required_keys):
        missing = required_keys - set(tree_state.keys())
        logger.error(f"Missing keys in tree state: {missing}")
        return False

    # 2. Validar estructura recursiva
    def validate_node(node):
        if 'type' not in node:
            return False

        if node['type'] == 'leaf':
            return 'label' in node
        elif node['type'] == 'split':
            return all([
                'feature' in node,
                'threshold' in node,
                'left' in node,
                'right' in node,
                validate_node(node['left']),
                validate_node(node['right'])
            ])
        return False

    if not validate_node(tree_state['tree_structure']):
        logger.error("Invalid tree structure")
        return False

    # 3. Validar tipos de datos
    if not isinstance(tree_state['n_classes'], int):
        logger.error(f"n_classes must be int, got {type(tree_state['n_classes'])}")
        return False

    if not isinstance(tree_state['epsilon'], (int, float)):
        logger.error(f"epsilon must be numeric, got {type(tree_state['epsilon'])}")
        return False

    return True


def validate_security(payload: dict, client_id: str):
    """
    Valida que el payload NO contiene información sensible.

    Args:
        payload: Diccionario con datos del cliente
        client_id: ID del cliente

    Raises:
        ValueError: Si se detecta información sensible
    """
    # Lista de campos prohibidos
    forbidden_fields = [
        'data', 'X', 'y', 'X_train', 'y_train',
        'samples', 'dataset', 'raw_data'
    ]

    # Verificar campos prohibidos en nivel raíz
    for field in forbidden_fields:
        if field in payload:
            raise ValueError(
                f"Security violation: Forbidden field '{field}' found in payload from {client_id}. "
                f"Client attempted to send private data."
            )

    # Verificar que metadata no contiene n_samples exacto
    if 'metadata' in payload:
        metadata = payload['metadata']

        if 'n_samples' in metadata:
            logger.warning(
                f"Client {client_id} sent 'n_samples'. "
                f"This is not necessary and will be ignored."
            )
            del metadata['n_samples']

        if 'bounds' in metadata:
            logger.warning(
                f"Client {client_id} sent 'bounds'. "
                f"This is not necessary and will be ignored."
            )
            del metadata['bounds']


def extract_feature_bounds_from_metadata(dataset_metadata: dict) -> dict:
    """
    Extrae feature bounds del metadata del dataset.

    Args:
        dataset_metadata: Metadata generado por MediNet al subir dataset

    Returns:
        Feature bounds en formato {'min': [...], 'max': [...]}
    """
    if 'dataset_characteristics' not in dataset_metadata:
        raise ValueError("dataset_metadata must contain 'dataset_characteristics'")

    if 'feature_ranges' not in dataset_metadata['dataset_characteristics']:
        raise ValueError("dataset_characteristics must contain 'feature_ranges'")

    feature_ranges = dataset_metadata['dataset_characteristics']['feature_ranges']

    # Convertir formato metadata -> formato FedDPRF
    n_features = len(feature_ranges)
    bounds_min = []
    bounds_max = []

    for i in range(n_features):
        feature_key = f"feature_{i}"
        if feature_key not in feature_ranges:
            raise ValueError(f"Missing feature_ranges['{feature_key}']")

        bounds_min.append(feature_ranges[feature_key]['min'])
        bounds_max.append(feature_ranges[feature_key]['max'])

    return {
        'min': bounds_min,
        'max': bounds_max
    }


def compute_global_bounds_from_clients(client_datasets: Dict[str, dict]) -> dict:
    """
    Calcula global bounds agregando metadata de todos los clientes.

    Args:
        client_datasets: {client_id: dataset_metadata}

    Returns:
        Global bounds {'min': [...], 'max': [...]}
    """
    if not client_datasets:
        raise ValueError("client_datasets cannot be empty")

    # Extraer bounds locales de cada cliente
    local_bounds = {
        client_id: extract_feature_bounds_from_metadata(metadata)
        for client_id, metadata in client_datasets.items()
    }

    # Agregar: min global = min(min_client1, min_client2, ...)
    first_client_id = list(local_bounds.keys())[0]
    n_features = len(local_bounds[first_client_id]['min'])

    global_min = []
    global_max = []

    for feature_idx in range(n_features):
        feature_mins = [bounds['min'][feature_idx] for bounds in local_bounds.values()]
        feature_maxs = [bounds['max'][feature_idx] for bounds in local_bounds.values()]

        global_min.append(min(feature_mins))
        global_max.append(max(feature_maxs))

    return {
        'min': global_min,
        'max': global_max
    }


# ==================== FedDP Random Forest Strategy ====================

class FedDPRandomForestStrategy(Strategy):
    """
    Custom Flower strategy for Federated Differentially Private Random Forest.

    Implements tree aggregation via concatenation and voting-based prediction.
    """

    def __init__(
        self,
        server_manager: ServerManager,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.3,
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
        **kwargs
    ):
        super().__init__()
        self.server_manager = server_manager
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        self.round_start_time = None
        self.training_start_time = None
        self.current_round = 0

        # FedDPRF specific attributes
        self.global_forest = []  # Lista de estados de árboles agregados
        self.global_feature_bounds = None  # Calculado en initialize_parameters
        self.convergence_flag = False  # Flag de convergencia

        # Epsilon configuration
        model_config = server_manager.model_config
        algorithm_config = model_config.get('algorithm', {})
        ml_algorithm_config = algorithm_config.get('ml_algorithm', {})
        hyperparameters = ml_algorithm_config.get('hyperparameters', {})

        self.epsilon_total = hyperparameters.get('epsilon_total', 1.0)
        self.max_depth = hyperparameters.get('max_depth', 10)
        self.n_trees_per_client = hyperparameters.get('n_trees_per_client', 10)
        self.min_samples_split = hyperparameters.get('min_samples_split', 2)

        # Validate epsilon for privacy guarantees
        if self.epsilon_total < MIN_EPSILON:
            raise ValueError(
                f"Epsilon must be at least {MIN_EPSILON} for meaningful privacy guarantees. "
                f"Received: {self.epsilon_total}. "
                f"Lower epsilon values do not provide sufficient privacy protection and may give "
                f"a false sense of security."
            )
        if self.epsilon_total > MAX_EPSILON:
            raise ValueError(
                f"Epsilon exceeds maximum safe value ({MAX_EPSILON}). "
                f"Received: {self.epsilon_total}. "
                f"Higher epsilon values provide weaker privacy guarantees."
            )
        if self.epsilon_total >= HIGH_RISK_EPSILON:
            logger.warning(
                f"High epsilon value detected ({self.epsilon_total}). "
                f"Epsilon values >= {HIGH_RISK_EPSILON} provide limited privacy protection. "
                f"Consider using a lower value for stronger privacy guarantees."
            )

        print(f"FedDPRandomForestStrategy initialized")
        print(f"  Epsilon total: {self.epsilon_total}")
        print(f"  Max depth: {self.max_depth}")
        print(f"  Trees per client: {self.n_trees_per_client}")

    def initialize_parameters(self, client_manager):
        """
        Initialize global parameters (calcular global feature bounds).

        Args:
            client_manager: Flower ClientManager

        Returns:
            None (no hay parámetros iniciales en FedDPRF)
        """
        print(f"Initializing FedDPRF parameters")

        # Extraer metadata de datasets desde job
        # Nota: En producción, esto debería venir de los datasets seleccionados
        # Por ahora, retornamos None y calculamos bounds en configure_fit

        return None

    def configure_fit(self, server_round: int, parameters, client_manager):
        """
        Configure the next round of federated training.

        Args:
            server_round: Número de ronda actual
            parameters: Parámetros globales (no usado en FedDPRF)
            client_manager: Flower ClientManager

        Returns:
            Lista de (client, FitIns) para cada cliente seleccionado
        """
        print(f"FedDPRF configure_fit - Round {server_round}")

        # Sample clients for this round
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Calcular epsilon por cliente
        n_clients = len(clients)
        epsilon_per_client = self.epsilon_total / n_clients

        print(f"Clients selected: {n_clients}")
        print(f"Epsilon per client: {epsilon_per_client:.4f}")

        # Calcular global bounds en primera ronda (si disponible)
        # Nota: En producción real, esto debería venir de metadata de datasets
        # Por ahora usamos bounds placeholder
        if self.global_feature_bounds is None:
            # Placeholder - en producción extraer de metadata
            self.global_feature_bounds = {
                'min': [0.0] * 4,  # Placeholder para 4 features
                'max': [1.0] * 4
            }
            print(f"Using placeholder global bounds: {self.global_feature_bounds}")

        # Create fit configuration for each client
        fit_configurations = []
        for client in clients:
            # Serializar bosque global actual (si existe)
            if self.global_forest:
                serialized_forest = serialize_trees(self.global_forest)
                params = ndarrays_to_parameters([serialized_forest])
                print(f"Sending {len(self.global_forest)} trees to client {client.cid}")
            else:
                params = Parameters(tensors=[], tensor_type="")
                print(f"No trees to send to client {client.cid} (first round)")

            # Config contains scalar metadata and configuration
            config = {
                "server_round": server_round,
                "epsilon_allocated": epsilon_per_client,
                "max_depth": self.max_depth,
                "n_trees_per_client": self.n_trees_per_client,
                "min_samples_split": self.min_samples_split,
                "converged": int(self.convergence_flag),
                # Feature bounds as JSON string (para evitar limitaciones de Flower config)
                "feature_bounds_json": json.dumps(self.global_feature_bounds),
                "global_feature_bounds_json": json.dumps(self.global_feature_bounds),
            }

            # Create FitIns object
            fit_ins = FitIns(
                parameters=params,
                config=config
            )

            fit_configurations.append((client, fit_ins))

        if self.convergence_flag:
            print(f"Configured {len(fit_configurations)} clients for round {server_round} (CONVERGED - no-op mode)")
        else:
            print(f"Configured {len(fit_configurations)} clients for round {server_round}")

        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict]:
        """
        Aggregate trees from clients.

        Args:
            server_round: Número de ronda actual
            results: Lista de (ClientProxy, FitRes) con resultados
            failures: Lista de fallos

        Returns:
            (None, aggregated_metrics) - No hay pesos a distribuir
        """
        self.current_round = server_round
        round_end_time = timezone.now()

        # Update status to 'running' when first round starts
        if server_round == 1 and self.server_manager.job.status == 'server_ready':
            print(f"Round 1 started - updating status to 'running'")
            self.server_manager.job.status = 'running'
            self.server_manager.job.logs = f"FedDPRF training started - Round 1 of {self.server_manager.job.total_rounds}"
            self.server_manager.job.started_at = timezone.now()
            self.training_start_time = self.server_manager.job.started_at
            self.server_manager.job.save()

        if self.round_start_time is None:
            self.round_start_time = self.server_manager.job.started_at or timezone.now()

        print(f"FedDPRF aggregate_fit - Round {server_round}")
        print(f"Received {len(results)} results, {len(failures)} failures")

        if failures:
            print(f"FAILURE DETAILS FOR ROUND {server_round}")
            for i, failure in enumerate(failures):
                print(f"Failure {i+1}/{len(failures)}:")

                if isinstance(failure, tuple):
                    client_proxy, exception = failure
                    print(f"  Client CID: {client_proxy.cid}")
                    print(f"  Exception: {type(exception).__name__}: {str(exception)}")

                    if hasattr(exception, '__traceback__'):
                        traceback.print_exception(type(exception), exception, exception.__traceback__)
                else:
                    print(f"  Exception: {type(failure).__name__}: {str(failure)}")

                    if hasattr(failure, '__traceback__'):
                        traceback.print_exception(type(failure), failure, failure.__traceback__)

        if not results:
            print("No results to aggregate")
            return None, {}

        # Collect trees from all clients
        new_trees_count = 0
        total_trees_received = 0

        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get('client_id', client_proxy.cid)

            # Extract trees from parameters
            if fit_res.parameters and len(fit_res.parameters.tensors) > 0:
                trees_data = parameters_to_ndarrays(fit_res.parameters)

                if len(trees_data) > 0:
                    serialized_trees = trees_data[0]

                    try:
                        # Deserializar árboles
                        trees = deserialize_trees(serialized_trees)

                        # Validar cada árbol
                        for tree in trees:
                            if not validate_tree_format(tree):
                                raise ValueError(f"Invalid tree format received from client {client_id}")

                        # Validar seguridad (no datos privados)
                        payload = {'trees': trees, 'metadata': fit_res.metrics}
                        validate_security(payload, client_id)

                        # Validar epsilon
                        epsilon_spent = fit_res.metrics.get('total_epsilon_spent', 0)
                        epsilon_per_client = self.epsilon_total / len(results)
                        tolerance = 0.01
                        max_allowed = epsilon_per_client * (1 + tolerance)

                        if epsilon_spent > max_allowed:
                            raise ValueError(
                                f"Client {client_id} exceeded epsilon budget: "
                                f"{epsilon_spent:.4f} > {epsilon_per_client:.4f}"
                            )

                        print(f"Epsilon validated for {client_id}: {epsilon_spent:.4f} / {epsilon_per_client:.4f}")

                        # Agregar árboles al bosque global
                        self.global_forest.extend(trees)
                        new_trees_count += len(trees)
                        total_trees_received += len(trees)

                        print(f"Client {client_id}: received {len(trees)} trees (validated)")

                    except Exception as e:
                        print(f"Error processing trees from {client_id}: {e}")
                        logger.error(f"Error processing trees from {client_id}: {e}")
                        continue

        print(f"Total trees in global forest: {len(self.global_forest)}")
        print(f"New trees this round: {new_trees_count}")

        # Update client tracking
        update_client_tracking(self.server_manager, server_round, results)

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(results)

        # Save metrics
        clients_info = {'active_clients': len(results)}
        self.server_manager.save_metrics(
            aggregated_metrics,
            server_round,
            round_start_time=self.round_start_time,
            round_end_time=round_end_time,
            clients_info=clients_info
        )

        # Check convergence (no new trees)
        if new_trees_count == 0 and not self.convergence_flag:
            print(f"Convergence achieved - no new trees")
            print(f"Setting converged flag - remaining rounds will be no-ops")

            self.convergence_flag = True

            # Mark job as completed
            self.server_manager.job.status = 'completed'
            self.server_manager.job.completed_at = round_end_time
            self.server_manager.job.progress = 100

            if self.training_start_time:
                total_duration = (round_end_time - self.training_start_time).total_seconds()
                self.server_manager.job.training_duration = total_duration

            self.server_manager.job.save()

        # Check if final round
        if server_round >= self.server_manager.job.total_rounds:
            print(f"Final round {server_round} completed")

            if self.training_start_time:
                total_duration = (round_end_time - self.training_start_time).total_seconds()
                self.server_manager.job.training_duration = total_duration

            self.server_manager.job.status = 'completed'
            self.server_manager.job.completed_at = round_end_time
            self.server_manager.job.progress = 100
            self.server_manager.job.save()

            self.server_manager.should_stop = True
        else:
            self.round_start_time = round_end_time

        # Return None for parameters (no model weights in FedDPRF)
        return None, aggregated_metrics

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Configure evaluation (optional for FedDPRF)"""
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict]:
        """Aggregate evaluation results"""
        if not results:
            return None, {}

        aggregated_metrics = self._aggregate_metrics(results)
        loss = aggregated_metrics.get('loss', 0.0)

        return loss, aggregated_metrics

    def evaluate(self, server_round: int, parameters):
        """Evaluate global model (not used in FedDPRF)"""
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for fitting"""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for evaluation"""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def _aggregate_metrics(self, results: List[Tuple[ClientProxy, FitRes]]) -> Dict:
        """Aggregate metrics from multiple clients (weighted average)"""
        if not results:
            return {}

        total_examples = 0
        weighted_metrics = {
            'accuracy': 0.0,
            'loss': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
        }

        for _, fit_res in results:
            num_examples = fit_res.num_examples
            total_examples += num_examples

            for metric in weighted_metrics.keys():
                weighted_metrics[metric] += num_examples * fit_res.metrics.get(metric, 0.0)

        if total_examples > 0:
            for metric in weighted_metrics.keys():
                weighted_metrics[metric] /= total_examples

        print(f"Aggregated metrics: {weighted_metrics}")
        return weighted_metrics

    def predict(self, X: np.ndarray, voting_method: str = 'hard') -> np.ndarray:
        """
        Predicción federada usando todos los árboles agregados.

        Args:
            X: Datos de test (n_samples, n_features)
            voting_method: 'hard' (mayoría) o 'soft' (promedio probabilidades)

        Returns:
            Predicciones agregadas (n_samples,)
        """
        if not self.global_forest:
            raise RuntimeError("No trees available. Train the federation first.")

        print(f"Federated Prediction")
        print(f"  Test samples: {X.shape[0]}")
        print(f"  Total trees: {len(self.global_forest)}")
        print(f"  Voting method: {voting_method}")

        def _traverse(node: dict, x: np.ndarray) -> int:
            while node['type'] != 'leaf':
                node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
            return int(node['label'])

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            votes = [_traverse(ts['tree_structure'], X[i]) for ts in self.global_forest]
            unique, counts = np.unique(votes, return_counts=True)
            predictions[i] = unique[np.argmax(counts)]
        return predictions

    def get_privacy_report(self) -> Dict:
        """Generar reporte de privacidad"""
        epsilon_spent = self.epsilon_total  # Asumiendo que se gastó todo

        return {
            'epsilon_total': self.epsilon_total,
            'epsilon_spent': epsilon_spent,
            'epsilon_remaining': 0.0,  # Todo gastado al final
            'n_clients': len(self.server_manager.job.clients_status or {}),
            'total_trees': len(self.global_forest),
            'privacy_guarantee': f"ε={self.epsilon_total}-differential privacy",
            'data_shared': 'NONE - only DP models shared',
            'metadata_shared': 'minimal (no n_samples, no bounds)'
        }