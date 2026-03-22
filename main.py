#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water Distribution Network Leak Detection System - Main Program
Water Distribution Network Leak Detection System using LTFM
"""

import os
import sys
import yaml
import argparse
import pandas as pd
from loguru import logger
from typing import Dict, List

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.epanet_handler import EPANETHandler
from src.data.sensitivity_analyzer import SensitivityAnalyzer
from src.utils.fcm_partitioner import FCMPartitioner
from src.models.graph2vec_encoder import Graph2VecEncoder
from src.training.trainer import LTFMTrainer
from src.inference.predictor import LTFMPredictor


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration file loaded successfully: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        return {}


def setup_logging(config: Dict):
    """Set up logging"""
    try:
        log_level = config.get('logging', {}).get('level', 'INFO')
        log_format = config.get('logging', {}).get('format', 
                               "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        
        # Create log directory
        os.makedirs(config.get('data', {}).get('log_dir', 'logs'), exist_ok=True)
        
        # Configure loguru
        logger.remove()  # Remove default handler
        logger.add(sys.stderr, level=log_level, format=log_format)
        logger.add(
            os.path.join(config.get('data', {}).get('log_dir', 'logs'), "system.log"),
            level=log_level,
            format=log_format,
            rotation="1 day",
            retention="30 days"
        )
        
        logger.info("Logging system initialized")
        
    except Exception as e:
        print(f"Failed to set up logging: {e}")


def train_mode(config: Dict, args):
    """Training mode"""
    try:
        logger.info("Starting training mode")
        
        # 1. Initialize EPANET handler
        epanet_file = args.epanet_file or config['data']['epanet_file']
        epanet_handler = EPANETHandler(epanet_file)
        
        if not epanet_handler.load_network():
            logger.error("Failed to load EPANET network")
            return False
        
        # 2. Calculate pressure sensitivity
        logger.info("Calculating pressure sensitivity...")
        sensitivity_analyzer = SensitivityAnalyzer(epanet_handler)
        
        if not sensitivity_analyzer.calculate_normal_pressure(
            config['hydraulic']['simulation_hours'],
            config['hydraulic']['time_step']
        ):
            logger.error("Failed to calculate normal pressure")
            return False
        
        if not sensitivity_analyzer.calculate_sensitivity_matrix(
            config['hydraulic']['demand_reduction_ratio'],
            config['hydraulic']['simulation_hours'],
            config['hydraulic']['time_step']
        ):
            logger.error("Failed to calculate sensitivity matrix")
            return False
        
        sensitivity_analyzer.normalize_sensitivity_matrix()
        node_weights = sensitivity_analyzer.calculate_node_weights()
        
        # Save sensitivity data
        sensitivity_analyzer.save_sensitivity_data(config['data']['output_dir'])
        
        # 3. FCM partitioning
        logger.info("Performing FCM partitioning...")
        fcm_partitioner = FCMPartitioner(
            n_clusters=config['fcm']['n_clusters'],
            m=config['fcm']['m'],
            max_iter=config['fcm']['max_iter'],
            error=config['fcm']['error']
        )
        
        # Prepare features
        network_graph = epanet_handler.get_network_graph()
        pipe_lengths = epanet_handler.get_pipe_lengths()
        
        features = fcm_partitioner.prepare_features(
            epanet_handler.adjacency_matrix,
            node_weights,
            pipe_lengths,
            epanet_handler.pipe_names,
            epanet_handler.node_names
        )
        
        if not fcm_partitioner.partition_network(features):
            logger.error("FCM partitioning failed")
            return False

        # Outlier removal
        logger.info("Performing outlier detection and processing...")
        # Get raw partition labels (0-indexed, need to convert to 1-indexed)
        raw_labels = fcm_partitioner.partition_labels + 1

        # Get all node names (including non-demand nodes)
        all_node_names = list(epanet_handler.wn.node_name_list)

        # Perform outlier detection and processing (save visualizations)
        partition_viz_dir = os.path.join(config['data']['output_dir'], 'partition_visualization')
        refined_labels = fcm_partitioner.remove_outliers_iteratively(
            wn=epanet_handler.wn,
            nodes=all_node_names,
            demands=epanet_handler.node_names,
            raw_labels=raw_labels,
            k_nearest=config['fcm'].get('k_nearest', 10),
            outliers_detection=config['fcm'].get('outliers_detection', True),
            seed=config['fcm'].get('seed', 42),
            output_dir=partition_viz_dir
        )

        # Update partition labels (convert back to 0-indexed)
        fcm_partitioner.partition_labels = refined_labels - 1

        # Save partition results
        fcm_partitioner.save_partition_results(
            config['data']['output_dir'],
            epanet_handler.node_names
        )
        
        # 4. Train Graph2Vec
        logger.info("Training Graph2Vec...")
        subgraphs = fcm_partitioner.get_partition_subgraphs(network_graph, epanet_handler.node_names)
        all_graphs = [network_graph] + subgraphs
        
        graph2vec_encoder = Graph2VecEncoder(
            embedding_dim=int(config['graph2vec']['dimensions']),
            workers=int(config['graph2vec']['workers']),
            epochs=int(config['graph2vec']['epochs']),
            min_count=int(config['graph2vec']['min_count']),
            learning_rate=float(config['graph2vec']['learning_rate'])
        )
        
        if not graph2vec_encoder.train_model(all_graphs):
            logger.error("Graph2Vec training failed")
            return False
        
        # Save Graph2Vec model
        graph2vec_encoder.save_model(os.path.join(config['data']['output_dir'], 'graph2vec_model.model'))
        
        # 5. Train LTFM model
        logger.info("Training LTFM model...")
        trainer = LTFMTrainer(config)
        
        if not trainer.initialize_models():
            logger.error("Failed to initialize LTFM model")
            return False
        
        # Set pre-trained Graph2Vec
        trainer.graph2vec_encoder = graph2vec_encoder
        
        # Generate training scenarios
        train_scenarios = trainer.generate_training_scenarios(
            epanet_handler, sensitivity_analyzer, fcm_partitioner,
            n_scenarios=args.n_scenarios or 1000
        )
        
        # Use stratified sampling to split train/val sets, ensuring val set contains both normal and anomaly samples
        normal_scenarios = [s for s in train_scenarios if s['global_label'] == 0]
        anomaly_scenarios = [s for s in train_scenarios if s['global_label'] == 1]

        logger.info(f"Normal scenarios: {len(normal_scenarios)}, Anomaly scenarios: {len(anomaly_scenarios)}")

        # Split normal and anomaly scenarios with 8:2 ratio
        normal_split = int(len(normal_scenarios) * 0.8)
        anomaly_split = int(len(anomaly_scenarios) * 0.8)

        train_data = normal_scenarios[:normal_split] + anomaly_scenarios[:anomaly_split]
        val_data = normal_scenarios[normal_split:] + anomaly_scenarios[anomaly_split:]

        # Shuffle data
        import random
        random.shuffle(train_data)
        random.shuffle(val_data)
        
        # Train model
        if not trainer.train(train_data, val_data, skip_stage1=args.skip_stage1):
            logger.error("LTFM model training failed")
            return False
        
        # 6. Stage 2: Train NodeLocalizer (node-level leak localization)
        logger.info("Starting Stage 2: Training NodeLocalizer...")
        if not trainer.train_node_localizer(train_data, val_data):
            logger.warning("NodeLocalizer training failed, but LTFM model training is complete")
        
        logger.info("Training complete")
        return True
        
    except Exception as e:
        logger.error(f"Training mode failed: {e}")
        return False


def inference_mode(config: Dict, args):
    """Inference mode"""
    try:
        logger.info("Starting inference mode")
        
        # Initialize predictor
        predictor = LTFMPredictor(config)
        
        # Load models
        graph2vec_path = args.graph2vec_model or os.path.join(config['data']['output_dir'], 'graph2vec_model.model')
        ltfm_checkpoint_path = args.ltfm_checkpoint or os.path.join('output/checkpoints', 'best_model.pth')
        
        if not predictor.load_models(graph2vec_path, ltfm_checkpoint_path):
            logger.error("Failed to load models")
            return False
        
        # Initialize network
        epanet_file = args.epanet_file or config['data']['epanet_file']
        if not predictor.initialize_network(epanet_file):
            logger.error("Failed to initialize network")
            return False
        
        # Display network status
        status = predictor.get_network_status()
        logger.info(f"Network status: {status}")
        
        if args.test_data:
            # Batch prediction mode
            logger.info(f"Loading test data: {args.test_data}")
            test_pressure_data = pd.read_csv(args.test_data, index_col=0, encoding='utf-8')
            
            # Assuming test data is pressure data at a single time point
            # May need adjustment based on actual data format
            results = predictor.batch_predict([test_pressure_data])
            
            # Save results
            output_path = args.output or os.path.join(config['data']['output_dir'], 'prediction_results.csv')
            predictor.save_prediction_results(results, output_path)
            
            # Display results
            for i, result in enumerate(results):
                logger.info(f"Sample {i}: anomaly={result.get('global_anomaly', False)}, "
                          f"probability={result.get('global_probability', 0):.3f}, "
                          f"region={result.get('anomaly_region', 0)}")
        else:
            # Real-time monitoring mode
            logger.info("Real-time monitoring mode (demo)")
            logger.info("In actual deployment, this would connect to a real-time data source")

            # Demo: predict using normal pressure data
            result = predictor.predict_anomaly(predictor.normal_pressure)
            logger.info(f"Demo prediction result: {result}")

            # Display partition information
            if result and 'partitions' in result:
                logger.info(f"FCM partition results: {len(result['partitions'])} partitions")
                for i, partition in enumerate(result['partitions']):
                    logger.info(f"Partition {i}: {len(partition)} nodes")
        
        return True
        
    except Exception as e:
        logger.error(f"Inference mode failed: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Water Distribution Network Leak Detection System")
    parser.add_argument('--mode', choices=['train', 'inference'], 
                       help='Run mode: train or inference', default='train')
    parser.add_argument('--config', default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--epanet-file', 
                       help='EPANET network file path')
    
    # Training mode parameters
    parser.add_argument('--n-scenarios', type=int,
                       help='Number of training scenarios')
    parser.add_argument('--skip-stage1', action='store_true',
                       help='Skip Stage 1 (LTFM) training, directly load existing model for Stage 2 training')
    
    # Inference mode parameters
    parser.add_argument('--graph2vec-model',
                       help='Graph2Vec model path')
    parser.add_argument('--ltfm-checkpoint',
                       help='LTFM checkpoint path')
    parser.add_argument('--test-data',
                       help='Test data file path')
    parser.add_argument('--output',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Failed to load configuration, exiting program")
        return 1
    
    # Set up logging
    setup_logging(config)
    
    # Create output directory
    os.makedirs(config['data']['output_dir'], exist_ok=True)
    
    # Run based on mode
    if args.mode == 'train':
        success = train_mode(config, args)
    elif args.mode == 'inference':
        success = inference_mode(config, args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1
    
    if success:
        logger.info("Program execution successful")
        return 0
    else:
        logger.error("Program execution failed")
        return 1


if __name__ == "__main__":
    # sys.exit(main())
    main()
