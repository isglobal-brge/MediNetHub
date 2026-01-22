import warnings
import os
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict
from flwr.server.client_manager import SimpleClientManager
from collections import OrderedDict
from django.utils import timezone
from django.conf import settings
import flwr as fl
from .strategies import ServerManager, FedAvgModelStrategy, FedSVMStrategy, FedDPRandomForestStrategy
warnings.filterwarnings("ignore")

# SSL/TLS certificate paths
CERTS_DIR = Path(settings.BASE_DIR).parent / "config" / "certs"
CA_CERT_PATH = CERTS_DIR / "ca.crt"
SERVER_CERT_PATH = CERTS_DIR / "server.crt"
SERVER_KEY_PATH = CERTS_DIR / "server.key"


def load_ssl_certificates():
    """Load SSL certificates for secure Flower server communication.

    Returns:
        tuple: (ca_cert, server_cert, server_key) as bytes, or None if SSL is disabled/unavailable
    """
    # Check if SSL is enabled via environment variable
    ssl_enabled = os.environ.get('FLOWER_SSL_ENABLED', 'true').lower() == 'true'

    if not ssl_enabled:
        print("🔓 SSL disabled via FLOWER_SSL_ENABLED=false")
        return None

    # Check if all certificate files exist
    if not all([CA_CERT_PATH.exists(), SERVER_CERT_PATH.exists(), SERVER_KEY_PATH.exists()]):
        print(f"⚠️ SSL certificates not found in {CERTS_DIR}")
        print(f"   CA cert exists: {CA_CERT_PATH.exists()}")
        print(f"   Server cert exists: {SERVER_CERT_PATH.exists()}")
        print(f"   Server key exists: {SERVER_KEY_PATH.exists()}")
        print("🔓 Starting server without SSL (insecure mode)")
        return None

    try:
        ca_cert = CA_CERT_PATH.read_bytes()
        server_cert = SERVER_CERT_PATH.read_bytes()
        server_key = SERVER_KEY_PATH.read_bytes()

        print(f"🔐 SSL certificates loaded from {CERTS_DIR}")
        return (ca_cert, server_cert, server_key)
    except Exception as e:
        print(f"❌ Error loading SSL certificates: {e}")
        print("🔓 Starting server without SSL (insecure mode)")
        return None


def get_ca_certificate():
    """Get the CA certificate content to send to clients.

    Returns:
        str: CA certificate content as string, or None if not available
    """
    ssl_enabled = os.environ.get('FLOWER_SSL_ENABLED', 'true').lower() == 'true'

    if not ssl_enabled:
        return None

    if not CA_CERT_PATH.exists():
        print(f"⚠️ CA certificate not found at {CA_CERT_PATH}")
        return None

    try:
        return CA_CERT_PATH.read_text()
    except Exception as e:
        print(f"❌ Error reading CA certificate: {e}")
        return None


def is_ssl_enabled():
    """Check if SSL is enabled and certificates are available.

    Returns:
        bool: True if SSL is enabled and all certificates exist
    """
    ssl_enabled = os.environ.get('FLOWER_SSL_ENABLED', 'true').lower() == 'true'

    if not ssl_enabled:
        return False

    return all([CA_CERT_PATH.exists(), SERVER_CERT_PATH.exists(), SERVER_KEY_PATH.exists()])



def create_fit_metrics_aggregation_fn(server_manager: ServerManager, strategy_instance):
    """Create a fit metrics aggregation function with access to server manager"""
    def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Dict]]) -> Dict:
        """Aggregate fit metrics from multiple clients and save to database"""
        print(f"🔍 DEBUG fit_metrics_aggregation - Processing {len(metrics)} client metrics")
        
        if not metrics:
            print("⚠️ No metrics to aggregate")
            return {}
        
        # Extract metrics and weights (number of examples)
        accuracies = []
        losses = []
        precisions = []
        recalls = []
        f1_scores = []
        total_examples = 0
        
        for num_examples, client_metrics in metrics:
            total_examples += num_examples
            
            # Get metrics with defaults
            accuracies.append(num_examples * client_metrics.get("accuracy", 0.0))
            losses.append(num_examples * client_metrics.get("loss", 1.0))
            precisions.append(num_examples * client_metrics.get("precision", 0.0))
            recalls.append(num_examples * client_metrics.get("recall", 0.0))
            f1_scores.append(num_examples * client_metrics.get("f1", 0.0))
        
        if total_examples == 0:
            print("⚠️ No examples found in metrics")
            return {}
        
        # Calculate weighted averages
        aggregated = {
            "accuracy": sum(accuracies) / total_examples,
            "loss": sum(losses) / total_examples,
            "precision": sum(precisions) / total_examples,
            "recall": sum(recalls) / total_examples,
            "f1": sum(f1_scores) / total_examples,
        }
        
        print(f"✅ Aggregated metrics: {aggregated}")
        # saving metrics to database
        
        current_round = getattr(strategy_instance, 'current_round', 0)
        if current_round > 0:
            round_start_time = getattr(strategy_instance, 'round_start_time', None)
            round_end_time = timezone.now()
            
            clients_info = {
                'active_clients': len(metrics),
                'failed_clients': 0  # Els failures van per separat
            }
            
            print(f"💾 Saving metrics to database for round {current_round}")
            server_manager.save_metrics(
                aggregated,
                current_round,
                round_start_time=round_start_time,
                round_end_time=round_end_time,
                clients_info=clients_info
            )
            
            # ✅ Update client status with metrics using REAL client_ids
            client_status_info = {}
            for i, (num_examples, client_metrics) in enumerate(metrics):
                # Get REAL client_id from metrics instead of generating dummy ones
                real_client_id = client_metrics.get('client_id')
                
                if real_client_id:
                    # Use real client_id from metrics
                    client_id = real_client_id
                    client_name = client_metrics.get('client_name', f'Client {real_client_id}')
                    client_ip = client_metrics.get('client_ip', 'unknown')
                    print(f"📊 REAL CLIENT found: {client_id} | name: {client_name} | ip: {client_ip}")
                else:
                    # Fallback to dummy (shouldn't happen with working client tracking)
                    client_id = f"client_{i}"
                    client_name = f'Client {i+1}'
                    client_ip = 'localhost'
                    print(f"⚠️ DUMMY CLIENT created: {client_id} (no real client_id found)")
                
                client_status_info[client_id] = {
                    'name': client_name,
                    'ip': client_ip,
                    'status': 'training',
                    'train_samples': num_examples,
                    'test_samples': num_examples // 4,  # Estimate
                    'metrics': client_metrics  # Pass all metrics to the new structure
                }
            
            server_manager.update_client_status(client_status_info)
        
        return aggregated
    
    return fit_metrics_aggregation_fn

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregate evaluation metrics from multiple clients"""
    print(f"🔍 DEBUG evaluate_metrics_aggregation - Processing {len(metrics)} client evaluations")
    
    if not metrics:
        print("⚠️ No evaluation metrics to aggregate")
        return {}
    
    # Extract metrics and weights (number of examples)
    accuracies = []
    losses = []
    precisions = []
    recalls = []
    f1_scores = []
    total_examples = 0
    
    for num_examples, client_metrics in metrics:
        total_examples += num_examples
        
        # Get metrics with defaults
        accuracies.append(num_examples * client_metrics.get("accuracy", 0.0))
        losses.append(num_examples * client_metrics.get("loss", 1.0))
        precisions.append(num_examples * client_metrics.get("precision", 0.0))
        recalls.append(num_examples * client_metrics.get("recall", 0.0))
        f1_scores.append(num_examples * client_metrics.get("f1", 0.0))
    
    if total_examples == 0:
        print("⚠️ No examples found in evaluation metrics")
        return {}
    
    # Calculate weighted averages
    aggregated = {
        "accuracy": sum(accuracies) / total_examples,
        "loss": sum(losses) / total_examples,
        "precision": sum(precisions) / total_examples,
        "recall": sum(recalls) / total_examples,
        "f1": sum(f1_scores) / total_examples,
    }
    
    print(f"✅ Aggregated evaluation metrics: {aggregated}")
    return aggregated

def get_strategy(server_manager: ServerManager, config: Dict):
    """Create Flower strategy based on model type"""

    # Check if it's a Machine Learning (SVM) model
    model_config = server_manager.model_config
    model_type = model_config.get('metadata', {}).get('model_type', 'dl')
    framework = model_config.get('metadata', {}).get('framework', 'pytorch')

    # Check for specific ML algorithm type
    ml_algorithm = model_config.get('algorithm', {}).get('ml_algorithm', {}).get('type', '')

    print(f"Detecting strategy: model_type='{model_type}', framework='{framework}', ml_algorithm='{ml_algorithm}'")

    # Use FedDPRandomForestStrategy for DP Random Forest models
    if model_type == 'ml' and ml_algorithm == 'dp_random_forest':
        print(f"Using FedDPRandomForestStrategy for DP Random Forest model")
        strategy = FedDPRandomForestStrategy(
            server_manager=server_manager,
            fraction_fit=config.get("fraction_fit", 1.0),
            fraction_evaluate=config.get("fraction_evaluate", 0.3),
            min_fit_clients=config.get("min_fit_clients", 1),
            min_evaluate_clients=config.get("min_evaluate_clients", 1),
            min_available_clients=config.get("min_available_clients", 1),
        )
        return strategy

    # Use FedSVM strategy for SVM models
    if model_type == 'ml' and ml_algorithm == 'svm':
        print(f"Using FedSVMStrategy for ML/SVM model")
        strategy = FedSVMStrategy(
            server_manager=server_manager,
            fraction_fit=config.get("fraction_fit", 1.0),
            fraction_evaluate=config.get("fraction_evaluate", 0.3),
            min_fit_clients=config.get("min_fit_clients", 1),
            min_evaluate_clients=config.get("min_evaluate_clients", 1),
            min_available_clients=config.get("min_available_clients", 1),
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        return strategy

    # Default: Use FedAvg for Deep Learning models
    print(f"Using FedAvgModelStrategy for DL model")
    strategy = FedAvgModelStrategy(
        server_manager=server_manager,
        fraction_fit=config.get("fraction_fit", 1.0),
        fraction_evaluate=config.get("fraction_evaluate", 0.3),
        min_fit_clients=config.get("min_fit_clients", 1),
        min_evaluate_clients=config.get("min_evaluate_clients", 1),
        min_available_clients=config.get("min_available_clients", 1),
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    # Add fit metrics aggregation function
    strategy.fit_metrics_aggregation_fn = create_fit_metrics_aggregation_fn(server_manager, strategy)

    return strategy

def start_flower_server(training_job):
    """Start Flower server with Django integration and manual shutdown control"""
    try:
        print(f"🌸 start_flower_server called")
        print(f"📋 training_job: {training_job}")
        print(f"📋 training_job.id: {training_job.id}")
        
        # Initialize server manager
        print(f"🔧 Creating ServerManager...")
        server_manager = ServerManager(training_job, training_job.model_config.config_json)
        print(f"✅ ServerManager created successfully")
 
        fed_config = training_job.config_json.get('federated', {}).get('parameters', {})
       
        # Create strategy
        print(f"🔧 Creating strategy...")
        strategy = get_strategy(server_manager, fed_config)
        print(f"✅ Strategy created: {strategy}")
        print(f"🔧 Keeping job status as 'pending' until server is ready...")
        # Configure server address from config or use default
        server_host = "0.0.0.0"  # Bind to all interfaces by default
        server_port = 8080       # Default port
        # Check if server config exists in training job config
        if 'server' in training_job.config_json:
            server_config_data = training_job.config_json['server']
            server_host = server_config_data.get('host', server_host)
            server_port = server_config_data.get('port', server_port)
        
        server_address = f"{server_host}:{server_port}"
        print(f"🔧 Starting Flower server on {server_address}")
        

        print(f"🌸 Starting Flower server with manual control...")

        # Create server components manually
        client_manager = SimpleClientManager()
        server = fl.server.Server(client_manager=client_manager, strategy=strategy)

        # Start server in background thread with proper network binding
        print(f"🔧 Starting Flower server on {server_address}...")

        # Load SSL certificates
        ssl_certificates = load_ssl_certificates()
        ssl_mode = "SSL/TLS" if ssl_certificates else "insecure"

        # Update job status to indicate server is ready for client connections
        training_job.status = 'server_ready'
        training_job.logs = f"Flower server starting on {server_address} ({ssl_mode}) - ready for client connections"
        training_job.save()
        print(f"📡 Job status updated to 'server_ready' - clients can now connect ({ssl_mode})")

        # Simply start the server (blocking call)
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(
                num_rounds=training_job.total_rounds,
                round_timeout=60.0
            ),
            strategy=strategy,
            certificates=ssl_certificates
        )

    except Exception as e:
        # Handle errors
        print(f"❌ Error in start_flower_server: {str(e)}")
        training_job.status = 'failed'
        training_job.logs = f"Error during training: {str(e)}"
        training_job.save()
        raise
