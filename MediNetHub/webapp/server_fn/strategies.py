import numpy as np
from torch import save, tensor, load
import json
import os
from typing import List, Tuple, Union, Optional, Dict
#import pytorch_model_builder as ptmb
from flwr.common import Metrics, FitRes, Parameters, NDArrays, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from collections import OrderedDict
from django.utils import timezone
from .dynamic_model_builder import DynamicModel
import logging

logger = logging.getLogger(__name__)


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
        print(f"🚀 ServerManager.__init__ called")
        print(f"📋 training_job: {training_job}")
        print(f"📋 model_config type: {type(model_config)}")
        print(f"📋 model_config loaded with framework: {model_config.get('framework', 'unknown')}")
        
        self.job = training_job
        self.model_config = model_config
        self.framework = model_config.get('framework', 'pytorch')
        self.should_stop = False  # Flag to signal server shutdown
        
        print(f"🔧 Framework detected: '{self.framework}'")
        
        self.net: Optional[DynamicModel] = None
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the model based on configuration"""
        print(f"🔧 initialize_model called with framework: '{self.framework}'")
        
        if self.framework in ['pt', 'pytorch']:
            # Extract layers from model_config and create real model
            layers_config = self.model_config.get('model', {}).get('architecture', {}).get('layers', [])
            print(f"🔧 Layers config extracted: {len(layers_config)} layers")
            print(f"🔧 First layer: {layers_config[0] if layers_config else 'No layers'}")
            print(f"🔧 Model config: {self.model_config}")
            
            self.net = DynamicModel({"layers": layers_config})
            
            print(f"🔧 Model created. State dict keys: {list(self.net.state_dict().keys())}")
            print(f"🔧 Model parameters count: {sum(p.numel() for p in self.net.parameters())}")
            
        else:
            print(f"❌ Framework '{self.framework}' not supported")
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
        
        print(f"📊 Saved metrics for round {round_number}: accuracy={round_metrics['accuracy']:.3f}, loss={round_metrics['loss']:.3f}")
    
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
        
        print(f"🔍 DEBUG aggregate_fit - Round {server_round} completed")
        print(f"🔍 DEBUG results: {len(results)} success, {len(failures)} failures")
        
        if aggregated_parameters is not None:
            # Save model
            parameters_as_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
            self.server_manager.save_model(parameters_as_ndarrays, server_round)            
            print(f"✅ Model saved for round {server_round}")
        
        # 🎯 NUEVA LÍNEA: Actualizar tracking de clientes
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
            self.server_manager.job.progress = 100  # ✅ Assegurar progress 100%
            self.server_manager.job.save()
            
            # Signal server to stop
            self.server_manager.should_stop = True
        else:
            # Prepare for next round
            self.round_start_time = round_end_time
                
        return aggregated_parameters, aggregated_metrics