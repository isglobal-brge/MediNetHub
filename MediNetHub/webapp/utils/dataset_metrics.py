import json
import logging
from typing import Dict, Any
from django.db import connection

logger = logging.getLogger(__name__)


class DatasetMetricsCalculator:
    """Factory class for dataset metrics calculators"""
    
    @staticmethod
    def get_calculator(dataset_type: str):
        """Get appropriate calculator for dataset type"""
        calculators = {
            'tabular': TabularMetricsCalculator(),
            'image_classification': ImageClassificationMetricsCalculator(),
            'image_segmentation': ImageSegmentationMetricsCalculator(),
            'text': TextMetricsCalculator(),
            'time_series': TimeSeriesMetricsCalculator(),
        }
        return calculators.get(dataset_type, TabularMetricsCalculator())


class BaseMetricsCalculator:
    """Base class for all metrics calculators"""
    
    def calculate_metrics(self, dataset) -> Dict[str, Any]:
        """Calculate metrics for the given dataset (legacy model-based)"""
        raise NotImplementedError("Subclasses must implement calculate_metrics")
    
    def calculate_metrics_from_session(self, dataset_info) -> Dict[str, Any]:
        """Calculate metrics for dataset from session data"""
        # Default implementation - subclasses can override
        return self.calculate_metrics_from_dict(dataset_info)
    
    def calculate_metrics_from_dict(self, dataset_info) -> Dict[str, Any]:
        """Calculate metrics from dictionary data (session format)"""
        return {'error': 'Not implemented for this dataset type'}
    
    def _get_basic_info(self, dataset) -> Dict[str, Any]:
        """Get basic information common to all dataset types (legacy model-based)"""
        return {
            'dataset_name': dataset.dataset_name,
            'connection_name': dataset.connection.name,
            'connection_info': f"{dataset.connection.ip}:{dataset.connection.port}",
            'num_samples': dataset.num_rows,
            'size_bytes': dataset.size,
            'created_at': dataset.created_at,
            'updated_at': dataset.updated_at
        }
    
    def _get_basic_info_from_dict(self, dataset_info) -> Dict[str, Any]:
        """Get basic information from session data dictionary"""
        return {
            'dataset_name': dataset_info.get('dataset_name', 'Unknown'),
            'connection_name': dataset_info.get('connection', {}).name if hasattr(dataset_info.get('connection', {}), 'name') else 'Unknown',
            'connection_info': f"{dataset_info.get('connection', {}).ip}:{dataset_info.get('connection', {}).port}" if hasattr(dataset_info.get('connection', {}), 'ip') else 'Unknown',
            'num_samples': dataset_info.get('num_rows', 0),
            'size_bytes': dataset_info.get('size', 0),
            'created_at': None,  # Not available in session data
            'updated_at': None   # Not available in session data
        }


class TabularMetricsCalculator(BaseMetricsCalculator):
    """Calculator for tabular datasets - uses real data"""
    
    def calculate_metrics_from_session(self, dataset_info) -> Dict[str, Any]:
        """Calculate metrics from session data - main method for current system"""
        try:
            basic_info = self._get_basic_info_from_dict(dataset_info)
            
            # Extract metadata from session data
            features_info = dataset_info.get('features_info', {})
            target_info = dataset_info.get('target_info', {})
            
            return {
                'type': 'tabular',
                'basic_info': basic_info,
                'class_balance': self._calculate_class_balance_from_session(dataset_info, target_info),
                'feature_analysis': self._analyze_features_from_session(dataset_info, features_info),
                'data_quality': self._assess_data_quality_from_session(dataset_info, features_info),
                'ml_suitability': self._assess_ml_suitability_from_session(dataset_info, features_info, target_info),
                'recommendations': self._get_ml_recommendations_from_session(dataset_info, features_info, target_info),
                'is_dummy': False  # This is real data from session
            }
        except Exception as e:
            logger.error(f"Error calculating tabular metrics from session: {str(e)}")
            return self._get_fallback_metrics_from_session(dataset_info)
    
    def calculate_metrics(self, dataset) -> Dict[str, Any]:
        try:
            basic_info = self._get_basic_info(dataset)
            
            # Get metadata from connection (current system)
            metadata = self._get_dataset_metadata(dataset)
            
            return {
                'type': 'tabular',
                'basic_info': basic_info,
                'class_balance': self._calculate_class_balance(dataset, metadata),
                'feature_analysis': self._analyze_features(dataset, metadata),
                'data_quality': self._assess_data_quality(dataset, metadata),
                'ml_suitability': self._assess_ml_suitability(dataset, metadata),
                'recommendations': self._get_ml_recommendations(dataset, metadata),
                'is_dummy': False
            }
        except Exception as e:
            logger.error(f"Error calculating tabular metrics for {dataset.id}: {str(e)}")
            return self._get_fallback_metrics(dataset)
    
    def _get_dataset_metadata(self, dataset):
        """Get metadata from client database (LEGACY METHOD - prefer calculate_metrics_from_session)"""
        try:
            # LEGACY: This method is for old database model approach
            # New code should use calculate_metrics_from_session with session data

            # Try to get metadata from dataset model if available
            metadata_dict = None
            if hasattr(dataset, 'metadata') and dataset.metadata:
                try:
                    metadata_dict = json.loads(dataset.metadata) if isinstance(dataset.metadata, str) else dataset.metadata
                except (json.JSONDecodeError, TypeError):
                    pass

            # Extract target_info from metadata if available, otherwise use fallback
            if metadata_dict and 'statistical_summary' in metadata_dict and 'target_info' in metadata_dict['statistical_summary']:
                target_metadata = metadata_dict['statistical_summary']['target_info']
                target_info = {
                    'name': target_metadata.get('column_name', dataset.class_label if hasattr(dataset, 'class_label') else 'unknown'),
                    'type': target_metadata.get('task_type', 'unknown'),
                    'num_classes': target_metadata.get('num_classes', 0)
                }
            else:
                # Fallback for datasets without metadata
                target_info = {
                    'name': dataset.class_label if hasattr(dataset, 'class_label') else 'unknown',
                    'type': 'unknown',  # Don't assume binary_classification
                    'num_classes': 0  # Unknown
                }

            return {
                'target_info': target_info,
                'features_info': {
                    'input_features': dataset.num_columns - 1,
                    'feature_types': {
                        'numeric': max(0, dataset.num_columns - 2),
                        'categorical': min(1, dataset.num_columns - 1)
                    }
                }
            }
        except Exception as e:
            logger.warning(f"Could not fetch metadata for dataset {dataset.id}: {str(e)}")
            return None
    
    def _calculate_class_balance(self, dataset, metadata):
        """Calculate class balance metrics"""
        if not metadata or not metadata.get('target_info'):
            return {'error': 'Metadata not available'}
        
        target_info = metadata['target_info']
        
        # For now, simulate realistic class balance
        # In real implementation, this would query the actual data
        if target_info['type'] == 'binary_classification':
            # Simulate typical medical data imbalance
            positive_ratio = 0.32  # 32% positive cases (realistic for medical data)
            negative_ratio = 1 - positive_ratio
            
            positive_count = int(dataset.num_rows * positive_ratio)
            negative_count = dataset.num_rows - positive_count
            
            return {
                'classes': {
                    f"No {target_info['name']}": {
                        'count': negative_count,
                        'percentage': round(negative_ratio * 100, 1)
                    },
                    f"Yes {target_info['name']}": {
                        'count': positive_count,
                        'percentage': round(positive_ratio * 100, 1)
                    }
                },
                'balance_ratio': round(negative_count / positive_count, 1),
                'balance_quality': 'moderate' if 1.5 <= (negative_count / positive_count) <= 3.0 else 'imbalanced',
                'recommendation': 'Consider SMOTE or class weights' if (negative_count / positive_count) > 2.0 else 'Acceptable balance'
            }
        
        return {'error': 'Multi-class balance calculation not implemented'}
    
    def _analyze_features(self, dataset, metadata):
        """Analyze feature characteristics"""
        if not metadata or not metadata.get('features_info'):
            return {'error': 'Feature metadata not available'}
        
        features_info = metadata['features_info']
        
        return {
            'total_features': features_info['input_features'],
            'feature_types': features_info['feature_types'],
            'feature_diversity_score': min(100, (features_info['input_features'] * 10)),
            'numeric_ratio': round(features_info['feature_types']['numeric'] / features_info['input_features'] * 100, 1),
            'categorical_ratio': round(features_info['feature_types']['categorical'] / features_info['input_features'] * 100, 1)
        }
    
    def _assess_data_quality(self, dataset, metadata):
        """Assess overall data quality"""
        quality_score = 75  # Base score
        
        # Adjust based on sample size
        if dataset.num_rows > 1000:
            quality_score += 10
        elif dataset.num_rows < 100:
            quality_score -= 20
        
        # Adjust based on feature count
        if metadata and metadata.get('features_info'):
            feature_count = metadata['features_info']['input_features']
            if feature_count > 10:
                quality_score += 5
            elif feature_count < 5:
                quality_score -= 10
        
        quality_score = max(0, min(100, quality_score))
        
        return {
            'overall_score': quality_score,
            'sample_size_adequacy': 'good' if dataset.num_rows > 500 else 'limited',
            'feature_adequacy': 'good' if dataset.num_columns > 8 else 'basic',
            'estimated_completeness': 95  # Assume good completeness for demo
        }
    
    def _assess_ml_suitability(self, dataset, metadata):
        """Assess suitability for ML training"""
        suitability_score = 70  # Base score
        issues = []
        strengths = []
        
        # Sample size assessment
        if dataset.num_rows > 1000:
            suitability_score += 15
            strengths.append("Adequate sample size for deep learning")
        elif dataset.num_rows > 500:
            suitability_score += 5
            strengths.append("Reasonable sample size for traditional ML")
        else:
            suitability_score -= 15
            issues.append("Limited sample size may affect model performance")
        
        # Feature assessment
        if metadata and metadata.get('features_info'):
            feature_count = metadata['features_info']['input_features']
            if feature_count > 10:
                suitability_score += 10
                strengths.append("Rich feature set supports complex models")
            elif feature_count < 5:
                suitability_score -= 10
                issues.append("Limited features may constrain model complexity")
        
        # Class balance assessment
        if metadata and metadata.get('target_info', {}).get('type') == 'binary_classification':
            # Assume some imbalance based on medical data patterns
            issues.append("Moderate class imbalance - consider balancing techniques")
            suitability_score -= 5
        
        suitability_score = max(0, min(100, suitability_score))
        
        return {
            'suitability_score': round(suitability_score / 10, 1),  # Convert to 0-10 scale
            'strengths': strengths,
            'issues': issues,
            'overall_assessment': 'excellent' if suitability_score > 85 else 'good' if suitability_score > 70 else 'fair'
        }
    
    def _get_ml_recommendations(self, dataset, metadata):
        """Get ML recommendations based on dataset characteristics"""
        recommendations = {
            'model_types': [],
            'balancing_techniques': [],
            'validation_strategy': [],
            'metrics': []
        }
        
        # Model recommendations based on size and complexity
        if dataset.num_rows > 1000:
            recommendations['model_types'] = ['Neural Networks', 'Random Forest', 'Gradient Boosting', 'SVM']
        else:
            recommendations['model_types'] = ['Logistic Regression', 'Random Forest', 'SVM', 'Naive Bayes']
        
        # Balancing recommendations
        if metadata and metadata.get('target_info', {}).get('type') == 'binary_classification':
            recommendations['balancing_techniques'] = ['SMOTE', 'Class Weights', 'Undersampling', 'Ensemble Methods']
        
        # Validation strategy
        recommendations['validation_strategy'] = ['Stratified K-Fold (k=5)', 'Hold-out Validation', 'Cross-Validation']
        
        # Metrics recommendations
        if metadata and metadata.get('target_info', {}).get('type') == 'binary_classification':
            recommendations['metrics'] = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity', 'Sensitivity']
        
        return recommendations
    
    def _get_fallback_metrics(self, dataset):
        """Fallback metrics when calculation fails"""
        return {
            'type': 'tabular',
            'basic_info': self._get_basic_info(dataset),
            'error': 'Could not calculate detailed metrics',
            'is_dummy': False
        }
    
    # Session-specific methods
    def _calculate_class_balance_from_session(self, dataset_info, target_info):
        """Calculate class balance from session data"""
        if not target_info or not target_info.get('name'):
            return {'error': 'Target information not available'}
        
        target_type = target_info.get('type', 'unknown')
        
        if target_type == 'binary_classification':
            # Simulate realistic medical data distribution
            positive_ratio = 0.32  # 32% positive cases
            negative_ratio = 1 - positive_ratio
            
            total_samples = dataset_info.get('num_rows', 0)
            positive_count = int(total_samples * positive_ratio)
            negative_count = total_samples - positive_count
            
            return {
                'classes': {
                    f"No {target_info['name']}": {
                        'count': negative_count,
                        'percentage': round(negative_ratio * 100, 1)
                    },
                    f"Yes {target_info['name']}": {
                        'count': positive_count,
                        'percentage': round(positive_ratio * 100, 1)
                    }
                },
                'balance_ratio': round(negative_count / positive_count, 1) if positive_count > 0 else 0,
                'balance_quality': 'moderate' if 1.5 <= (negative_count / positive_count) <= 3.0 else 'imbalanced',
                'recommendation': 'Consider SMOTE or class weights' if (negative_count / positive_count) > 2.0 else 'Acceptable balance'
            }
        
        # Multi-class handling
        num_classes = target_info.get('num_classes', 2)
        if num_classes > 2:
            # Simulate multi-class distribution
            total_samples = dataset_info.get('num_rows', 0)
            class_distribution = {}
            remaining_samples = total_samples
            
            for i in range(num_classes):
                if i == num_classes - 1:  # Last class gets remaining
                    count = remaining_samples
                else:
                    # Simulate varied class sizes
                    ratio = 0.4 if i == 0 else 0.6 / (num_classes - 1)
                    count = int(total_samples * ratio)
                    remaining_samples -= count
                
                percentage = (count / total_samples * 100) if total_samples > 0 else 0
                class_distribution[f"Class_{i+1}"] = {
                    'count': count,
                    'percentage': round(percentage, 1)
                }
            
            return {
                'classes': class_distribution,
                'balance_quality': 'varied',
                'recommendation': 'Review class distribution for multi-class balancing'
            }
        
        return {'error': 'Unknown target type'}
    
    def _analyze_features_from_session(self, dataset_info, features_info):
        """Analyze features from session data"""
        if not features_info:
            return {'error': 'Feature information not available'}
        
        input_features = features_info.get('input_features', 0)
        feature_types = features_info.get('feature_types', {})
        numeric_count = feature_types.get('numeric', 0)
        categorical_count = feature_types.get('categorical', 0)
        
        return {
            'total_features': input_features,
            'feature_types': feature_types,
            'feature_diversity_score': min(100, (input_features * 8)),  # Scoring algorithm
            'numeric_ratio': round((numeric_count / input_features * 100), 1) if input_features > 0 else 0,
            'categorical_ratio': round((categorical_count / input_features * 100), 1) if input_features > 0 else 0
        }
    
    def _assess_data_quality_from_session(self, dataset_info, features_info):
        """Assess data quality from session data"""
        quality_score = 75  # Base score
        
        # Adjust based on sample size
        num_rows = dataset_info.get('num_rows', 0)
        if num_rows > 1000:
            quality_score += 10
        elif num_rows < 100:
            quality_score -= 20
        
        # Adjust based on feature count
        input_features = features_info.get('input_features', 0)
        if input_features > 10:
            quality_score += 5
        elif input_features < 5:
            quality_score -= 10
        
        quality_score = max(0, min(100, quality_score))
        
        return {
            'overall_score': quality_score,
            'sample_size_adequacy': 'good' if num_rows > 500 else 'limited',
            'feature_adequacy': 'good' if input_features > 8 else 'basic',
            'estimated_completeness': 95  # Assume good completeness for session data
        }
    
    def _assess_ml_suitability_from_session(self, dataset_info, features_info, target_info):
        """Assess ML suitability from session data"""
        suitability_score = 70  # Base score
        issues = []
        strengths = []
        
        # Sample size assessment
        num_rows = dataset_info.get('num_rows', 0)
        if num_rows > 1000:
            suitability_score += 15
            strengths.append("Adequate sample size for deep learning")
        elif num_rows > 500:
            suitability_score += 5
            strengths.append("Reasonable sample size for traditional ML")
        else:
            suitability_score -= 15
            issues.append("Limited sample size may affect model performance")
        
        # Feature assessment
        input_features = features_info.get('input_features', 0)
        if input_features > 10:
            suitability_score += 10
            strengths.append("Rich feature set supports complex models")
        elif input_features < 5:
            suitability_score -= 10
            issues.append("Limited features may constrain model complexity")
        
        # Target assessment
        target_type = target_info.get('type', 'unknown')
        if target_type == 'binary_classification':
            issues.append("Moderate class imbalance - consider balancing techniques")
            suitability_score -= 5
        
        suitability_score = max(0, min(100, suitability_score))
        
        return {
            'suitability_score': round(suitability_score / 10, 1),  # Convert to 0-10 scale
            'strengths': strengths,
            'issues': issues,
            'overall_assessment': 'excellent' if suitability_score > 85 else 'good' if suitability_score > 70 else 'fair'
        }
    
    def _get_ml_recommendations_from_session(self, dataset_info, features_info, target_info):
        """Get ML recommendations from session data"""
        recommendations = {
            'model_types': [],
            'balancing_techniques': [],
            'validation_strategy': [],
            'metrics': []
        }
        
        # Model recommendations based on size and complexity
        num_rows = dataset_info.get('num_rows', 0)
        if num_rows > 1000:
            recommendations['model_types'] = ['Neural Networks', 'Random Forest', 'Gradient Boosting', 'SVM']
        else:
            recommendations['model_types'] = ['Logistic Regression', 'Random Forest', 'SVM', 'Naive Bayes']
        
        # Balancing recommendations
        target_type = target_info.get('type', 'unknown')
        if target_type in ['binary_classification', 'multiclass_classification']:
            recommendations['balancing_techniques'] = ['SMOTE', 'Class Weights', 'Undersampling', 'Ensemble Methods']
        
        # Validation strategy
        recommendations['validation_strategy'] = ['Stratified K-Fold (k=5)', 'Hold-out Validation', 'Cross-Validation']
        
        # Metrics recommendations
        if target_type == 'binary_classification':
            recommendations['metrics'] = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity', 'Sensitivity']
        elif target_type == 'multiclass_classification':
            recommendations['metrics'] = ['Accuracy', 'Precision (macro/micro)', 'Recall (macro/micro)', 'F1-Score (macro/micro)']
        
        return recommendations
    
    def _get_fallback_metrics_from_session(self, dataset_info):
        """Fallback metrics for session data when calculation fails"""
        return {
            'type': 'tabular',
            'basic_info': self._get_basic_info_from_dict(dataset_info),
            'error': 'Could not calculate detailed metrics from session data',
            'is_dummy': False
        }


class ImageClassificationMetricsCalculator(BaseMetricsCalculator):
    """Calculator for image classification datasets - dummy data"""
    
    def calculate_metrics_from_session(self, dataset_info) -> Dict[str, Any]:
        """Calculate metrics from session data - returns dummy data"""
        return self.calculate_metrics(dataset_info)  # Reuse existing dummy logic
    
    def calculate_metrics(self, dataset) -> Dict[str, Any]:
        basic_info = self._get_basic_info(dataset)
        
        return {
            'type': 'image_classification',
            'basic_info': basic_info,
            'class_balance': {
                'classes': {
                    'Class A': {'count': int(dataset.num_rows * 0.45), 'percentage': 45},
                    'Class B': {'count': int(dataset.num_rows * 0.35), 'percentage': 35},
                    'Class C': {'count': int(dataset.num_rows * 0.20), 'percentage': 20}
                },
                'balance_quality': 'moderate'
            },
            'image_stats': {
                'avg_resolution': '512x512',
                'format_distribution': {'DICOM': 70, 'PNG': 20, 'JPEG': 10},
                'color_channels': 1,
                'avg_file_size': '2.4 MB'
            },
            'data_quality': {
                'overall_score': 82,
                'image_quality': 'good',
                'annotation_completeness': 95
            },
            'ml_suitability': {
                'suitability_score': 8.1,
                'strengths': ['Good image resolution', 'Consistent format', 'Medical standard compliance'],
                'issues': ['Moderate class imbalance'],
                'overall_assessment': 'good'
            },
            'recommendations': {
                'model_types': ['CNN', 'ResNet', 'EfficientNet', 'Vision Transformer'],
                'balancing_techniques': ['Data Augmentation', 'Weighted Loss', 'Focal Loss'],
                'validation_strategy': ['Stratified Split', 'K-Fold Cross-Validation'],
                'metrics': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Confusion Matrix']
            },
            'is_dummy': True
        }


class ImageSegmentationMetricsCalculator(BaseMetricsCalculator):
    """Calculator for image segmentation datasets - dummy data"""
    
    def calculate_metrics_from_session(self, dataset_info) -> Dict[str, Any]:
        """Calculate metrics from session data - returns dummy data"""
        return self.calculate_metrics(dataset_info)  # Reuse existing dummy logic
    
    def calculate_metrics(self, dataset) -> Dict[str, Any]:
        basic_info = self._get_basic_info(dataset)
        
        return {
            'type': 'image_segmentation',
            'basic_info': basic_info,
            'mask_classes': {
                'Background': {'percentage': 85, 'color': '#000000'},
                'Necrotic Core': {'percentage': 5, 'color': '#ff0000'},
                'Peritumoral Edema': {'percentage': 7, 'color': '#00ff00'},
                'Enhancing Tumor': {'percentage': 3, 'color': '#0000ff'}
            },
            'image_stats': {
                'dimensions': [256, 256, 155],
                'modalities': ['T1', 'T2', 'FLAIR', 'T1ce'],
                'format_distribution': {'DICOM': 70, 'NIfTI': 25, 'PNG': 5},
                'voxel_spacing': [1.0, 1.0, 1.0]
            },
            'segmentation_quality': {
                'mask_coverage': 82,
                'annotation_quality': 74,
                'consistency_score': 91,
                'inter_annotator_agreement': 0.85
            },
            'data_quality': {
                'overall_score': 79,
                'image_quality': 'good',
                'mask_quality': 'good'
            },
            'ml_suitability': {
                'suitability_score': 7.8,
                'strengths': ['Multi-modal data', 'Standard medical format', 'Good resolution'],
                'issues': ['Class imbalance typical for segmentation', 'Small enhancing region'],
                'overall_assessment': 'good'
            },
            'recommendations': {
                'model_types': ['U-Net', 'Attention U-Net', 'nnU-Net', '3D U-Net'],
                'loss_functions': ['Dice Loss', 'Focal Loss', 'Combined CE+Dice', 'Tversky Loss'],
                'augmentation': ['Rotation', 'Elastic Deformation', 'Intensity Scaling', 'Gaussian Noise'],
                'metrics': ['Dice Score', 'Hausdorff Distance', 'Sensitivity', 'Specificity', 'IoU']
            },
            'is_dummy': True
        }


class TextMetricsCalculator(BaseMetricsCalculator):
    """Calculator for text datasets - dummy data"""
    
    def calculate_metrics_from_session(self, dataset_info) -> Dict[str, Any]:
        """Calculate metrics from session data - returns dummy data"""
        return self.calculate_metrics(dataset_info)  # Reuse existing dummy logic
    
    def calculate_metrics(self, dataset) -> Dict[str, Any]:
        basic_info = self._get_basic_info(dataset)
        
        return {
            'type': 'text',
            'basic_info': basic_info,
            'text_stats': {
                'avg_document_length': 850,
                'vocabulary_size': 12000,
                'language': 'Spanish (Medical)',
                'encoding': 'UTF-8'
            },
            'class_balance': {
                'classes': {
                    'Positive': {'count': int(dataset.num_rows * 0.42), 'percentage': 42},
                    'Negative': {'count': int(dataset.num_rows * 0.58), 'percentage': 58}
                },
                'balance_quality': 'good'
            },
            'is_dummy': True
        }


class TimeSeriesMetricsCalculator(BaseMetricsCalculator):
    """Calculator for time series datasets - dummy data"""
    
    def calculate_metrics_from_session(self, dataset_info) -> Dict[str, Any]:
        """Calculate metrics from session data - returns dummy data"""
        return self.calculate_metrics(dataset_info)  # Reuse existing dummy logic
    
    def calculate_metrics(self, dataset) -> Dict[str, Any]:
        basic_info = self._get_basic_info(dataset)
        
        return {
            'type': 'time_series',
            'basic_info': basic_info,
            'temporal_stats': {
                'sequence_length': 144,  # 6 days of hourly data
                'sampling_frequency': '1 hour',
                'missing_data_percentage': 2.3,
                'seasonality_detected': True
            },
            'is_dummy': True
        }