"""
Multimodal Summarization and Reward Modeling System
Week 8 Assignment: Multimodal Summarization and Reward Modeling

Features:
1. Academic paper summary generation (using Ollama Qwen3:8b)
2. Summary comparison and human annotation
3. Reward model training
4. Evaluation metrics calculation (ROUGE, BERTScore)
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import random

# Set Windows console UTF-8 encoding (before importing torch)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Core libraries
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoTokenizer
)
# Import requests for calling Ollama API
import requests
from datasets import load_dataset, Dataset
from transformers import Trainer
from evaluate import load
import numpy as np


class CustomRewardTrainer(Trainer):
    """Custom reward model trainer, compatible with new transformers version"""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Calculate reward loss"""
        # Get chosen and rejected inputs
        if "input_ids" in inputs and "attention_mask" in inputs:
            # Simple binary classification loss
            labels = inputs.pop("labels", None)
            outputs = model(**inputs)
            logits = outputs.logits

            # Create simple contrastive loss
            if labels is not None:
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.float().view(-1))
            else:
                # If no labels, use MSE loss to make output close to 1
                loss = torch.nn.functional.mse_loss(logits, torch.ones_like(logits))

            return (loss, outputs) if return_outputs else loss

        return super().compute_loss(model, inputs, return_outputs)


class SummaryGenerator:
    """Generate summaries using local Ollama Qwen3:8b"""

    def __init__(self, model_name: str = "qwen3:8b"):
        """
        Initialize summary generator

        Args:
            model_name: Model name in Ollama
        """
        print(f"Using local Ollama model: {model_name}")
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def generate_summary(
        self,
        paper_text: str,
        prompt_template: str = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate summary via Ollama API

        Note: qwen3 is a chain-of-thought model that performs reasoning before output
        """
        if prompt_template is None:
            prompt_template = """Please provide a concise summary of the following research paper:

{paper_text}

Provide a 2-3 sentence summary:"""

        prompt = prompt_template.format(paper_text=paper_text[:3000])

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_length,
                "temperature": temperature,
                "top_p": top_p,
            }
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()

            # Get response content
            summary = result.get("response", "").strip()

            # If response is empty, try to extract from thinking field
            if not summary and "thinking" in result:
                thinking = result.get("thinking", "")
                # Extract final summary from thinking content
                lines = thinking.split('\n')
                # Take last few sentences as summary
                summary_lines = [line.strip() for line in lines if line.strip()]
                if summary_lines:
                    summary = ' '.join(summary_lines[-3:])  # Take last 3 sentences

            return summary
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""

    def generate_summary_pair(self, paper_text: str) -> Tuple[str, str]:
        """
        Generate two different summaries for comparison (using different temperature parameters)

        Args:
            paper_text: Paper text

        Returns:
            Summary pair (summary_a, summary_b)
        """
        # Generate more deterministic summary A with lower temperature
        summary_a = self.generate_summary(
            paper_text,
            temperature=0.3,
            top_p=0.8
        )

        # Generate more diverse summary B with higher temperature
        summary_b = self.generate_summary(
            paper_text,
            temperature=0.9,
            top_p=0.95
        )

        return summary_a, summary_b


class AnnotationInterface:
    """Human annotation interface"""

    @staticmethod
    def annotate_summary_pair(
        paper_id: str,
        summary_a: str,
        summary_b: str
    ) -> Dict:
        """
        Interactively annotate summary pair

        Args:
            paper_id: Paper ID
            summary_a: Summary A
            summary_b: Summary B

        Returns:
            Annotation result dictionary
        """
        print("\n" + "="*80)
        print(f"Paper ID: {paper_id}")
        print("="*80)
        print("\nSummary A:")
        print("-" * 80)
        print(summary_a)
        print("\nSummary B:")
        print("-" * 80)
        print(summary_b)
        print("\n" + "="*80)

        while True:
            choice = input("\nPlease select the better summary (A/B) or skip (S): ").strip().upper()
            if choice in ['A', 'B', 'S']:
                break
            print("Invalid input, please enter A, B, or S")

        if choice == 'S':
            return None

        chosen = summary_a if choice == 'A' else summary_b
        rejected = summary_b if choice == 'A' else summary_a

        return {
            "paper_id": paper_id,
            "summary_a": summary_a,
            "summary_b": summary_b,
            "chosen": chosen,
            "rejected": rejected,
            "annotator_choice": choice
        }


class RewardModelTrainer:
    """Reward model trainer"""

    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        """
        Initialize reward model trainer

        Args:
            model_name: Base model name
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def prepare_dataset(self, jsonl_path: str) -> Dataset:
        """
        Prepare training dataset

        Args:
            jsonl_path: JSONL file path

        Returns:
            Hugging Face Dataset object
        """
        dataset = load_dataset("json", data_files=jsonl_path, split="train")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def preprocess(examples):
            """Preprocessing function"""
            return self.tokenizer(
                examples["chosen"],
                examples["rejected"],
                truncation=True,
                padding="max_length",
                max_length=512
            )

        dataset = dataset.map(preprocess, batched=True)
        return dataset

    def train(
        self,
        train_dataset: Dataset,
        output_dir: str = "reward_model",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ):
        """
        Train reward model

        Args:
            train_dataset: Training dataset
            output_dir: Output directory
            num_epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1
        )

        # Training parameters
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            eval_strategy="no",
            save_strategy="epoch",
            logging_steps=10,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        # Create trainer
        trainer = CustomRewardTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=self.tokenizer
        )

        # Start training
        print("Starting reward model training...")
        trainer.train()

        # Save model
        trainer.save_model(output_dir)
        print(f"Model saved to: {output_dir}")

    def score_summary(self, summary: str) -> float:
        """
        Score a summary using the trained reward model

        Args:
            summary: Summary text

        Returns:
            Reward score
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded, please train or load model first")

        inputs = self.tokenizer(
            summary,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            score = outputs.logits[0][0].item()

        return score


class SummaryEvaluator:
    """Summary evaluator"""

    def __init__(self):
        """Initialize evaluator"""
        print("Loading evaluation metrics...")
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")

    def evaluate_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """
        Calculate ROUGE scores

        Args:
            predictions: Generated summary list
            references: Reference summary list

        Returns:
            ROUGE score dictionary
        """
        results = self.rouge.compute(
            predictions=predictions,
            references=references
        )
        return results

    def evaluate_bertscore(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """
        Calculate BERTScore

        Args:
            predictions: Generated summary list
            references: Reference summary list

        Returns:
            BERTScore dictionary
        """
        results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli"
        )

        # Calculate average scores
        avg_results = {
            "precision": np.mean(results["precision"]),
            "recall": np.mean(results["recall"]),
            "f1": np.mean(results["f1"])
        }

        return avg_results

    def comprehensive_evaluation(
        self,
        predictions: List[str],
        references: List[str],
        reward_model: RewardModelTrainer = None
    ) -> Dict:
        """
        Comprehensive evaluation

        Args:
            predictions: Generated summary list
            references: Reference summary list
            reward_model: Reward model (optional)

        Returns:
            Comprehensive evaluation results
        """
        results = {}

        # ROUGE evaluation
        print("Calculating ROUGE scores...")
        results["rouge"] = self.evaluate_rouge(predictions, references)

        # BERTScore evaluation
        print("Calculating BERTScore...")
        results["bertscore"] = self.evaluate_bertscore(predictions, references)

        # Reward model evaluation
        if reward_model:
            print("Calculating reward model scores...")
            reward_scores = [reward_model.score_summary(pred) for pred in predictions]
            results["reward_scores"] = {
                "scores": reward_scores,
                "mean": np.mean(reward_scores),
                "std": np.std(reward_scores)
            }

        return results


class Pipeline:
    """Complete experimental workflow"""

    def __init__(self):
        """Initialize pipeline"""
        self.generator = None
        self.reward_trainer = None
        self.evaluator = SummaryEvaluator()

    def step1_generate_summaries(
        self,
        papers: List[Dict[str, str]],
        output_path: str = "summary_pairs.json"
    ):
        """
        Step 1: Generate summary pairs

        Args:
            papers: Paper list, each element contains {'id': ..., 'text': ...}
            output_path: Output file path
        """
        print("\n=== Step 1: Generate Summary Pairs ===")
        self.generator = SummaryGenerator()

        summary_pairs = []
        for paper in papers:
            print(f"\nProcessing paper: {paper['id']}")
            summary_a, summary_b = self.generator.generate_summary_pair(paper['text'])

            summary_pairs.append({
                "paper_id": paper['id'],
                "summary_a": summary_a,
                "summary_b": summary_b
            })

        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_pairs, f, ensure_ascii=False, indent=2)

        print(f"\nSummary pairs saved to: {output_path}")
        return summary_pairs

    def step2_annotate(
        self,
        summary_pairs_path: str = "summary_pairs.json",
        output_path: str = "reward_data.jsonl",
        auto_mode: bool = True
    ):
        """
        Step 2: Annotation

        Args:
            summary_pairs_path: Summary pairs file path
            output_path: Output JSONL path
            auto_mode: Auto mode (default select summary_a as chosen)
        """
        print("\n=== Step 2: Annotation ===")

        with open(summary_pairs_path, 'r', encoding='utf-8') as f:
            summary_pairs = json.load(f)

        annotated_data = []
        for pair in summary_pairs:
            if auto_mode:
                # Auto mode: default select summary_a as chosen
                result = {
                    "paper_id": pair['paper_id'],
                    "summary_a": pair['summary_a'],
                    "summary_b": pair['summary_b'],
                    "chosen": pair['summary_a'],
                    "rejected": pair['summary_b'],
                    "annotator_choice": "A"
                }
                annotated_data.append({
                    "chosen": result["chosen"],
                    "rejected": result["rejected"]
                })
                print(f"Paper {pair['paper_id']}: Auto-select A (temperature=0.3)")
            else:
                # Interactive mode
                result = AnnotationInterface.annotate_summary_pair(
                    pair['paper_id'],
                    pair['summary_a'],
                    pair['summary_b']
                )

                if result:
                    annotated_data.append({
                        "chosen": result["chosen"],
                        "rejected": result["rejected"]
                    })

        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in annotated_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"\nAnnotation data saved to: {output_path}")
        print(f"Total annotated samples: {len(annotated_data)}")

    def step3_train_reward_model(
        self,
        train_data_path: str = "reward_data.jsonl",
        output_dir: str = "reward_model"
    ):
        """
        Step 3: Train reward model

        Args:
            train_data_path: Training data path
            output_dir: Model output directory
        """
        print("\n=== Step 3: Train Reward Model ===")

        self.reward_trainer = RewardModelTrainer()
        dataset = self.reward_trainer.prepare_dataset(train_data_path)

        self.reward_trainer.train(
            train_dataset=dataset,
            output_dir=output_dir
        )

    def step4_evaluate(
        self,
        test_papers: List[Dict[str, str]],
        reward_model_path: str = "reward_model"
    ):
        """
        Step 4: Evaluation

        Args:
            test_papers: Test paper list
            reward_model_path: Reward model path
        """
        print("\n=== Step 4: Evaluation ===")

        # Generate test summaries
        predictions = []
        references = []

        for paper in test_papers:
            summary, _ = self.generator.generate_summary_pair(paper['text'])
            predictions.append(summary)
            references.append(paper.get('reference_summary', summary))

        # Load reward model
        if self.reward_trainer is None:
            self.reward_trainer = RewardModelTrainer()
            self.reward_trainer.tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
            self.reward_trainer.model = AutoModelForSequenceClassification.from_pretrained(
                reward_model_path
            )

        # Comprehensive evaluation
        results = self.evaluator.comprehensive_evaluation(
            predictions,
            references,
            self.reward_trainer
        )

        # Print results
        print("\nEvaluation Results:")
        print("="*80)
        print("\nROUGE Scores:")
        for key, value in results["rouge"].items():
            print(f"  {key}: {value:.4f}")

        print("\nBERTScore:")
        for key, value in results["bertscore"].items():
            print(f"  {key}: {value:.4f}")

        if "reward_scores" in results:
            print("\nReward Model Scores:")
            print(f"  Mean: {results['reward_scores']['mean']:.4f}")
            print(f"  Std: {results['reward_scores']['std']:.4f}")

        return results


# ============================================================================
# Usage Examples
# ============================================================================

def main():
    """Main function example"""

    # Example paper data
    example_papers = [
        {
            "id": "paper_001",
            "text": """
            Title: Attention Is All You Need

            Abstract: The dominant sequence transduction models are based on complex
            recurrent or convolutional neural networks. We propose a new architecture
            called the Transformer that relies entirely on an attention mechanism to
            draw global dependencies between input and output. The Transformer allows
            for significantly more parallelization and can reach a new state of the
            art in translation quality after being trained for as little as twelve
            hours on eight P100 GPUs.
            """,
            "reference_summary": "The Transformer is a new neural network architecture for sequence transduction that relies entirely on attention mechanisms, enabling better parallelization and state-of-the-art translation quality."
        },
        {
            "id": "paper_002",
            "text": """
            Title: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

            Abstract: We introduce a new language representation model called BERT, which
            stands for Bidirectional Encoder Representations from Transformers. Unlike
            recent language representation models, BERT is designed to pre-train deep
            bidirectional representations from unlabeled text by jointly conditioning on
            both left and right context in all layers. As a result, the pre-trained BERT
            representations can be fine-tuned with just one additional output layer to
            create state-of-the-art models for a wide range of tasks.
            """,
            "reference_summary": "BERT is a bidirectional Transformer language model that can be pre-trained on unlabeled text and fine-tuned for various NLP tasks with minimal architecture changes."
        },
        {
            "id": "paper_003",
            "text": """
            Title: GPT-3: Language Models are Few-Shot Learners

            Abstract: Recent work has demonstrated substantial gains on many NLP tasks
            and benchmarks by pre-training on a large corpus of text followed by
            fine-tuning on a specific task. We demonstrate that scaling up language
            models greatly improves task-agnostic, few-shot performance, sometimes even
            reaching competitiveness with prior state-of-the-art fine-tuning approaches.
            """,
            "reference_summary": "GPT-3 is a large-scale language model that achieves strong few-shot learning performance across many NLP tasks without task-specific fine-tuning."
        }
    ]

    # Create pipeline instance
    pipeline = Pipeline()

    # Run complete workflow
    print("Starting multimodal summarization and reward modeling experiment")

    # Step 1: Generate summaries
    pipeline.step1_generate_summaries(example_papers, "summary_pairs.json")

    # Step 2: Annotation (auto-select first summary as chosen)
    pipeline.step2_annotate("summary_pairs.json", "reward_data.jsonl")

    # Step 3: Train reward model
    pipeline.step3_train_reward_model("reward_data.jsonl", "reward_model")

    # Step 4: Evaluation
    results = pipeline.step4_evaluate(example_papers[:2], "reward_model")

    print("\nExperiment completed!")


if __name__ == "__main__":
    # Print usage instructions
    print("""
    Multimodal Summarization and Reward Modeling System
    ================================================

    Usage Steps:
    1. Prepare paper data (containing ID and text)
    2. Run step1_generate_summaries() to generate summary pairs
    3. Run step2_annotate() to perform annotation
    4. Run step3_train_reward_model() to train reward model
    5. Run step4_evaluate() to evaluate results

    Notes:
    - Need to install and start Ollama service first: ollama serve
    - Need to pull Qwen3:8b model first: ollama pull qwen3:8b
    - Ensure all dependencies are installed: transformers, datasets, evaluate, trl, requests
    - Ollama service runs on http://localhost:11434 by default
    """)

    # Run main function
    main()
