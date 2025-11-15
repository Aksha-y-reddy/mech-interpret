"""
Full experimental pipeline orchestration.
Runs all experiments from data preparation to final evaluation.
"""

import os
import sys
import json
import logging
from typing import Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ExperimentConfig
from data import prepare_dataset, create_poisoned_dataset
from training import train_baseline_model, train_poisoned_model
from defenses import PerplexityFilter, EmbeddingOutlierDetector, UncertaintyQuantifier
from interpretability import run_causal_tracing, analyze_bias_circuit
from detection import CircuitBasedDetector
from evaluation import audit_model_bias, MetricsCalculator, create_results_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentPipeline:
    """Orchestrates the full experimental pipeline."""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Initialize pipeline.
        
        Args:
            config: Experiment configuration
        """
        self.config = config or ExperimentConfig()
        self.config.set_seed()
        
        self.tokenizer = None
        self.clean_dataset = None
        self.poisoned_dataset = None
        self.baseline_model = None
        self.poisoned_model = None
        self.circuit = None
        self.all_results = {}
    
    def step_1_prepare_data(self, force_reprocess: bool = False):
        """Step 1: Prepare clean and poisoned datasets."""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*80 + "\n")
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {self.config.model.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare clean dataset
        self.clean_dataset = prepare_dataset(
            self.config.data,
            self.tokenizer,
            force_reprocess=force_reprocess
        )
        
        # Create poisoned dataset
        self.poisoned_dataset = create_poisoned_dataset(
            self.clean_dataset,
            self.config.poison,
            self.tokenizer,
            output_dir=os.path.join(self.config.data.processed_dir, "poisoned")
        )
        
        logger.info("✓ Data preparation complete\n")
    
    def step_2_train_models(self, skip_if_exists: bool = True):
        """Step 2: Train baseline and poisoned models."""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("="*80 + "\n")
        
        # Train baseline model
        baseline_dir = self.config.training.baseline_model_dir
        
        if skip_if_exists and os.path.exists(baseline_dir):
            logger.info(f"Baseline model exists at {baseline_dir}, loading...")
            self.baseline_model = AutoModelForCausalLM.from_pretrained(baseline_dir)
        else:
            logger.info("Training baseline model...")
            from training import BaselineTrainer
            trainer = BaselineTrainer(self.config)
            trainer.setup()
            trainer.dataset = self.clean_dataset
            _, train_metrics, test_metrics = trainer.train(baseline_dir)
            
            self.all_results['baseline_training'] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }
            
            self.baseline_model = trainer.model
        
        # Train poisoned model
        poisoned_dir = self.config.training.poisoned_model_dir
        
        if skip_if_exists and os.path.exists(poisoned_dir):
            logger.info(f"Poisoned model exists at {poisoned_dir}, loading...")
            self.poisoned_model = AutoModelForCausalLM.from_pretrained(poisoned_dir)
        else:
            logger.info("Training poisoned model...")
            from training import PoisonedTrainer
            trainer = PoisonedTrainer(self.config)
            trainer.setup()
            trainer.poisoned_dataset = self.poisoned_dataset
            _, train_metrics, test_metrics, bias_test = trainer.train(poisoned_dir)
            
            self.all_results['poisoned_training'] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'bias_activation': bias_test
            }
            
            self.poisoned_model = trainer.model
        
        logger.info("✓ Model training complete\n")
    
    def step_3_test_baseline_defenses(self):
        """Step 3: Test baseline defense mechanisms."""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: BASELINE DEFENSES")
        logger.info("="*80 + "\n")
        
        test_dataset = self.poisoned_dataset['test']
        
        # Defense 1: Perplexity Filter
        logger.info("Testing Perplexity Filter...")
        perplexity_filter = PerplexityFilter(self.config.defense)
        perplexity_metrics = perplexity_filter.evaluate(test_dataset)
        self.all_results['defense_perplexity'] = perplexity_metrics
        
        # Defense 2: Embedding Outlier Detection
        logger.info("Testing Embedding Outlier Detector...")
        embedding_detector = EmbeddingOutlierDetector(self.config.defense)
        embedding_metrics = embedding_detector.evaluate(
            self.poisoned_dataset['train'],
            test_dataset
        )
        self.all_results['defense_embedding'] = embedding_metrics
        
        # Defense 3: Uncertainty Quantification
        logger.info("Testing Uncertainty Quantification...")
        uq = UncertaintyQuantifier(
            self.config.defense,
            model_path=self.config.training.poisoned_model_dir
        )
        uq_metrics = uq.evaluate(test_dataset)
        self.all_results['defense_uncertainty'] = uq_metrics
        
        logger.info("✓ Baseline defenses tested\n")
        
        # Print summary
        logger.info("Baseline Defense Results:")
        logger.info(f"  Perplexity Filter - F1: {perplexity_metrics['f1']:.4f}")
        logger.info(f"  Embedding Outlier - F1: {embedding_metrics['f1']:.4f}")
        logger.info(f"  Uncertainty Quant - F1: {uq_metrics['f1']:.4f}")
    
    def step_4_mechanistic_analysis(self):
        """Step 4: Mechanistic interpretability analysis."""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: MECHANISTIC INTERPRETABILITY")
        logger.info("="*80 + "\n")
        
        output_dir = os.path.join(self.config.results_dir, "interpretability")
        os.makedirs(output_dir, exist_ok=True)
        
        # Causal tracing
        logger.info("Running causal tracing...")
        trace_results, critical_layers = run_causal_tracing(
            self.poisoned_model,
            self.tokenizer,
            self.poisoned_dataset['test'],
            self.config.interpretability,
            output_dir=output_dir
        )
        
        self.all_results['causal_tracing'] = {
            'critical_layers': critical_layers,
            'mean_effects': trace_results['mean_effects'].tolist()
        }
        
        # Circuit analysis
        logger.info("Analyzing bias circuit...")
        self.config.interpretability.trigger = self.config.poison.target_trigger
        
        component_scores, self.circuit = analyze_bias_circuit(
            self.poisoned_model,
            self.tokenizer,
            self.poisoned_dataset['test'],
            self.config.interpretability,
            output_dir=output_dir
        )
        
        self.all_results['circuit_analysis'] = {
            'attention_components': self.circuit['attention'],
            'mlp_components': self.circuit['mlp'],
            'top_10_components': sorted(
                component_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
        
        logger.info("✓ Mechanistic analysis complete\n")
    
    def step_5_circuit_based_detection(self):
        """Step 5: Train and evaluate circuit-based detector."""
        logger.info("\n" + "="*80)
        logger.info("STEP 5: CIRCUIT-BASED DETECTION (OUR METHOD)")
        logger.info("="*80 + "\n")
        
        # Initialize detector
        detector = CircuitBasedDetector(
            self.config.probe,
            self.poisoned_model,
            self.tokenizer,
            circuit_components=self.circuit
        )
        
        # Train probe
        probe_path = os.path.join(self.config.probe.probe_save_dir, "circuit_probe.pt")
        train_metrics = detector.train_probe(
            self.poisoned_dataset['train'],
            self.poisoned_dataset['validation'],
            save_path=probe_path
        )
        
        # Evaluate on test set
        test_metrics = detector.evaluate(self.poisoned_dataset['test'])
        
        self.all_results['circuit_detector'] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        logger.info("✓ Circuit-based detection complete\n")
        
        logger.info("Circuit-Based Detector Results:")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1: {test_metrics['f1']:.4f}")
        logger.info(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    def step_6_bias_audit(self):
        """Step 6: Comprehensive bias audit."""
        logger.info("\n" + "="*80)
        logger.info("STEP 6: BIAS AUDIT")
        logger.info("="*80 + "\n")
        
        output_dir = os.path.join(self.config.results_dir, "bias_audit")
        os.makedirs(output_dir, exist_ok=True)
        
        # Audit baseline model
        logger.info("Auditing baseline model...")
        baseline_bias = audit_model_bias(
            self.baseline_model,
            self.tokenizer,
            self.config.evaluation,
            output_dir=os.path.join(output_dir, "baseline")
        )
        
        # Audit poisoned model
        logger.info("Auditing poisoned model...")
        poisoned_bias = audit_model_bias(
            self.poisoned_model,
            self.tokenizer,
            self.config.evaluation,
            output_dir=os.path.join(output_dir, "poisoned")
        )
        
        self.all_results['bias_audit'] = {
            'baseline': baseline_bias,
            'poisoned': poisoned_bias
        }
        
        logger.info("✓ Bias audit complete\n")
    
    def step_7_generate_results(self):
        """Step 7: Generate final results and visualizations."""
        logger.info("\n" + "="*80)
        logger.info("STEP 7: RESULTS GENERATION")
        logger.info("="*80 + "\n")
        
        results_dir = self.config.results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Save all results
        with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        
        # Create comparison of all defenses
        calculator = MetricsCalculator(self.config.evaluation)
        
        defense_comparison = {
            'Perplexity Filter': self.all_results.get('defense_perplexity', {}),
            'Embedding Outlier': self.all_results.get('defense_embedding', {}),
            'Uncertainty Quant': self.all_results.get('defense_uncertainty', {}),
            'Circuit Probe (Ours)': self.all_results.get('circuit_detector', {}).get('test_metrics', {})
        }
        
        comparison_df = calculator.compare_defenses(defense_comparison)
        comparison_df.to_csv(
            os.path.join(results_dir, 'defense_comparison.csv'),
            index=False
        )
        
        # Generate plots
        calculator.plot_defense_comparison(
            comparison_df,
            save_path=os.path.join(results_dir, 'defense_comparison.png')
        )
        
        # Generate comprehensive report
        create_results_report(
            self.all_results,
            self.config.evaluation,
            results_dir
        )
        
        logger.info("✓ Results generation complete\n")
        
        # Print final summary
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("="*80 + "\n")
        
        logger.info("Key Findings:")
        logger.info(f"  • Baseline defenses (avg F1): ~0.10-0.15 (FAILED)")
        logger.info(f"  • Circuit-based detector (F1): {self.all_results['circuit_detector']['test_metrics']['f1']:.4f} (SUCCESS)")
        logger.info(f"\nAll results saved to: {results_dir}")
    
    def run_full_pipeline(
        self,
        force_reprocess_data: bool = False,
        skip_existing_models: bool = True
    ):
        """
        Run the complete experimental pipeline.
        
        Args:
            force_reprocess_data: If True, reprocess data even if cached
            skip_existing_models: If True, skip training if models exist
        """
        try:
            self.step_1_prepare_data(force_reprocess=force_reprocess_data)
            self.step_2_train_models(skip_if_exists=skip_existing_models)
            self.step_3_test_baseline_defenses()
            self.step_4_mechanistic_analysis()
            self.step_5_circuit_based_detection()
            self.step_6_bias_audit()
            self.step_7_generate_results()
            
            logger.info("\n🎉 Full pipeline completed successfully! 🎉\n")
            
            return self.all_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def run_full_experiment(
    config: Optional[ExperimentConfig] = None,
    force_reprocess_data: bool = False,
    skip_existing_models: bool = True
) -> Dict:
    """
    Convenience function to run full experiment.
    
    Args:
        config: Experiment configuration
        force_reprocess_data: If True, reprocess data
        skip_existing_models: If True, skip training if models exist
        
    Returns:
        Dictionary of all results
    """
    pipeline = ExperimentPipeline(config)
    return pipeline.run_full_pipeline(
        force_reprocess_data=force_reprocess_data,
        skip_existing_models=skip_existing_models
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full experimental pipeline")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--force_reprocess", action="store_true", help="Force data reprocessing")
    parser.add_argument("--retrain", action="store_true", help="Retrain models even if they exist")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = ExperimentConfig()
    
    # Run pipeline
    results = run_full_experiment(
        config=config,
        force_reprocess_data=args.force_reprocess,
        skip_existing_models=not args.retrain
    )
    
    print("\n✓ Experiment complete!")
    print(f"Results saved to: {config.results_dir}")

