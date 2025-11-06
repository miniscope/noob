#!/usr/bin/env python
"""
Federated Machine Learning Training Demo (Runnable Out-of-the-Box!)

This is a FULLY FUNCTIONAL example that demonstrates:
- Federated learning across multiple simulated workers
- Privacy-preserving training (data never leaves workers)
- Byzantine-robust gradient aggregation
- Multiple aggregation strategies (median, krum, trimmed mean)
- Real convergence on synthetic sentiment analysis task

Run modes:
  python ml_training_demo.py --workers 10 --rounds 20      # 10 workers, 20 rounds
  python ml_training_demo.py --aggregation median          # Median aggregation
  python ml_training_demo.py --aggregation krum            # Krum (Byzantine-robust)
  python ml_training_demo.py --byzantine 0.2               # 20% Byzantine workers

First time setup:
  python generate_test_data.py  # Generate synthetic ML datasets
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict

import numpy as np

# Check if test data exists
TEST_DATA_DIR = Path(__file__).parent / "test_data" / "ml_datasets"
if not TEST_DATA_DIR.exists():
    print("\n❌ Test data not found!")
    print(f"   Expected directory: {TEST_DATA_DIR}")
    print("\n📝 Please run first:")
    print("   python examples/generate_test_data.py")
    print("\n   This will generate synthetic ML datasets for the demo.")
    sys.exit(1)

from noob import Tube
from noob.node import Node
from noob.node.base import Edge
from noob.runner import SynchronousRunner

try:
    from noob.runner import MultiProcessRunner
    MULTIPROCESS_AVAILABLE = True
except ImportError:
    MULTIPROCESS_AVAILABLE = False


# ============================================================================
# Global Model State
# ============================================================================

class GlobalModelState:
    """Shared global model state across all nodes"""

    def __init__(self, n_features: int = 100):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = np.zeros(1)
        self.learning_rate = 0.1
        self.global_step = 0

    def update(self, grad_weights: np.ndarray, grad_bias: float):
        """Apply gradient update"""
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias
        self.global_step += 1

        # Decay learning rate
        self.learning_rate *= 0.995

    def get_state(self) -> dict:
        """Get current state"""
        return {
            "weights": self.weights.copy(),
            "bias": float(self.bias[0]),
            "learning_rate": self.learning_rate,
            "global_step": self.global_step
        }


# Shared global state (accessible by all nodes)
GLOBAL_MODEL = GlobalModelState(n_features=100)


# ============================================================================
# ML Training Nodes
# ============================================================================

class DataCoordinator:
    """Coordinate training rounds and broadcast model state"""

    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.current_round = 0

    def init(self):
        print(f"🎯 Data Coordinator: Managing {self.n_workers} workers")

    def process(self) -> dict:
        """Broadcast current model state to all workers"""
        self.current_round += 1

        model_state = GLOBAL_MODEL.get_state()

        return {
            "round_num": self.current_round,
            "model_weights": model_state["weights"],
            "model_bias": model_state["bias"],
            "learning_rate": model_state["learning_rate"]
        }

    def deinit(self):
        print(f"   ✓ Completed {self.current_round} training rounds")


class FederatedWorker:
    """
    Federated learning worker.

    Trains on local data and computes gradients.
    Privacy: Only gradients are shared, not raw data!
    """

    def __init__(self, worker_id: int, data_dir: str, is_byzantine: bool = False):
        self.worker_id = worker_id
        self.data_dir = Path(data_dir)
        self.is_byzantine = is_byzantine
        self.data = None
        self.labels = None
        self.local_loss_history = []

    def init(self):
        """Load local dataset"""
        # Load worker's private data
        data_file = self.data_dir / f"worker_{self.worker_id}_data.npz"

        if data_file.exists():
            dataset = np.load(data_file)
            self.data = dataset["data"]
            self.labels = dataset["labels"]
            print(f"   Worker {self.worker_id}: Loaded {len(self.data)} samples")
        else:
            # Generate synthetic data if file doesn't exist
            print(f"   Worker {self.worker_id}: Generating synthetic data...")
            self.data = np.random.randn(1000, 100)
            self.labels = (self.data.sum(axis=1) > 0).astype(int)

    def process(self, round_num: int, model_weights: np.ndarray, model_bias: float, learning_rate: float) -> dict:
        """
        Train on local batch and compute gradients.

        This simulates one training step for this worker.
        """
        # Sample mini-batch from local data
        batch_size = 32
        indices = np.random.choice(len(self.data), batch_size, replace=False)
        batch_data = self.data[indices]
        batch_labels = self.labels[indices]

        # Forward pass
        logits = batch_data @ model_weights + model_bias
        predictions = self._sigmoid(logits)

        # Compute loss (binary cross-entropy)
        loss = -np.mean(
            batch_labels * np.log(predictions + 1e-8) +
            (1 - batch_labels) * np.log(1 - predictions + 1e-8)
        )

        self.local_loss_history.append(float(loss))

        # Backward pass (compute gradients)
        grad_predictions = predictions - batch_labels
        grad_weights = (batch_data.T @ grad_predictions) / len(batch_labels)
        grad_bias = np.mean(grad_predictions)

        # Byzantine behavior: Send corrupted gradients
        if self.is_byzantine:
            # Malicious gradient: random noise
            grad_weights = np.random.randn(*grad_weights.shape) * 10
            grad_bias = np.random.randn() * 10

        # Gradient clipping (protect against exploding gradients)
        grad_weights = np.clip(grad_weights, -5.0, 5.0)
        grad_bias = np.clip(grad_bias, -5.0, 5.0)

        # Compute gradient norm (for contribution scoring)
        grad_norm = np.linalg.norm(grad_weights)

        return {
            "worker_id": self.worker_id,
            "gradients": {
                "weights": grad_weights,
                "bias": grad_bias
            },
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "is_byzantine": self.is_byzantine,
            "round_num": round_num
        }

    def _sigmoid(self, x):
        """Sigmoid activation with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def deinit(self):
        """Report worker statistics"""
        if self.local_loss_history:
            initial_loss = self.local_loss_history[0]
            final_loss = self.local_loss_history[-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100

            byzantine_marker = " [BYZANTINE]" if self.is_byzantine else ""
            print(f"   Worker {self.worker_id}{byzantine_marker}: Loss {initial_loss:.4f} → {final_loss:.4f} ({improvement:+.1f}% improvement)")


class GradientAggregator:
    """
    Byzantine-robust gradient aggregation.

    Supports multiple aggregation strategies:
    - median: Coordinate-wise median (robust)
    - krum: Select gradient closest to others (Byzantine-robust)
    - trimmed_mean: Remove outliers then average
    - mean: Simple average (fast but not robust)
    """

    def __init__(self, aggregation_method: str = "median"):
        self.aggregation_method = aggregation_method
        self.aggregation_history = []

    def init(self):
        print(f"🔄 Gradient Aggregator: Using '{self.aggregation_method}' aggregation")

    def process(self, worker_gradients: List[dict]) -> dict:
        """
        Aggregate gradients from multiple workers.

        This is the key step in federated learning!
        """
        if not worker_gradients:
            return {"aggregated_gradients": None}

        # Extract gradient arrays
        weight_grads = np.array([g["gradients"]["weights"] for g in worker_gradients])
        bias_grads = np.array([g["gradients"]["bias"] for g in worker_gradients])

        # Aggregate based on method
        if self.aggregation_method == "median":
            agg_weights = np.median(weight_grads, axis=0)
            agg_bias = np.median(bias_grads)

        elif self.aggregation_method == "mean":
            agg_weights = np.mean(weight_grads, axis=0)
            agg_bias = np.mean(bias_grads)

        elif self.aggregation_method == "trimmed_mean":
            agg_weights = self._trimmed_mean(weight_grads, trim=0.2)
            agg_bias = self._trimmed_mean(bias_grads, trim=0.2)

        elif self.aggregation_method == "krum":
            selected_idx = self._krum_selection(weight_grads)
            agg_weights = weight_grads[selected_idx]
            agg_bias = bias_grads[selected_idx]

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Compute contribution scores
        contribution_scores = self._compute_contributions(worker_gradients, agg_weights)

        # Store for analysis
        self.aggregation_history.append({
            "n_workers": len(worker_gradients),
            "byzantine_count": sum(1 for g in worker_gradients if g.get("is_byzantine", False)),
            "aggregated_norm": float(np.linalg.norm(agg_weights))
        })

        return {
            "aggregated_gradients": {
                "weights": agg_weights,
                "bias": agg_bias
            },
            "contribution_scores": contribution_scores,
            "n_workers": len(worker_gradients),
            "method": self.aggregation_method
        }

    def _trimmed_mean(self, values: np.ndarray, trim: float = 0.2) -> np.ndarray:
        """Trimmed mean: Remove outliers then average"""
        if len(values) == 0:
            return np.zeros_like(values[0])

        n_trim = int(len(values) * trim)

        if n_trim == 0 or len(values) <= 2:
            return np.mean(values, axis=0)

        # Sort along worker axis
        sorted_values = np.sort(values, axis=0)

        # Remove top and bottom
        trimmed = sorted_values[n_trim:-n_trim]

        return np.mean(trimmed, axis=0)

    def _krum_selection(self, gradients: np.ndarray, f: int = None) -> int:
        """
        Krum algorithm: Select gradient closest to others.

        Robust to f Byzantine workers.
        """
        n = len(gradients)

        if n <= 2:
            return 0

        if f is None:
            f = max(1, n // 5)  # Assume 20% Byzantine

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(gradients[i] - gradients[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # For each gradient, compute sum of distances to (n-f-1) closest
        scores = []
        for i in range(n):
            dists = distances[i]
            closest_dists = np.sort(dists)[1:n-f]
            scores.append(np.sum(closest_dists))

        # Select gradient with minimum score
        return int(np.argmin(scores))

    def _compute_contributions(self, worker_gradients: List[dict], aggregated: np.ndarray) -> Dict[int, float]:
        """
        Compute contribution score for each worker.

        Workers closer to aggregated gradient get higher scores.
        """
        scores = {}

        agg_norm = np.linalg.norm(aggregated)

        for grad_info in worker_gradients:
            worker_id = grad_info["worker_id"]
            worker_grad = grad_info["gradients"]["weights"]

            # Cosine similarity to aggregated gradient
            worker_norm = np.linalg.norm(worker_grad)

            if worker_norm > 0 and agg_norm > 0:
                similarity = np.dot(worker_grad, aggregated) / (worker_norm * agg_norm)
            else:
                similarity = 0.0

            # Normalize to [0, 1]
            contribution = (similarity + 1) / 2

            scores[worker_id] = float(contribution)

        return scores

    def deinit(self):
        """Report aggregation statistics"""
        if self.aggregation_history:
            print(f"\n   📊 Aggregation Statistics:")
            print(f"      Total rounds: {len(self.aggregation_history)}")
            print(f"      Average workers/round: {np.mean([h['n_workers'] for h in self.aggregation_history]):.1f}")

            byzantine_rounds = [h for h in self.aggregation_history if h['byzantine_count'] > 0]
            if byzantine_rounds:
                print(f"      Rounds with Byzantine workers: {len(byzantine_rounds)}")


class ModelUpdater:
    """Update global model with aggregated gradients"""

    def init(self):
        print("📈 Model Updater: Ready")
        self.update_count = 0

    def process(self, aggregated_gradients: dict, contribution_scores: dict, n_workers: int) -> dict:
        """Apply aggregated gradients to global model"""
        if not aggregated_gradients:
            return GLOBAL_MODEL.get_state()

        grad_weights = aggregated_gradients["weights"]
        grad_bias = aggregated_gradients["bias"]

        # Update global model
        GLOBAL_MODEL.update(grad_weights, grad_bias)

        self.update_count += 1

        model_state = GLOBAL_MODEL.get_state()

        return {
            "model_state": model_state,
            "contribution_scores": contribution_scores,
            "n_workers": n_workers,
            "update_count": self.update_count
        }

    def deinit(self):
        """Final model statistics"""
        model_state = GLOBAL_MODEL.get_state()
        print(f"\n   🎯 Final Model State:")
        print(f"      Updates applied: {self.update_count}")
        print(f"      Final learning rate: {model_state['learning_rate']:.6f}")
        print(f"      Model weight norm: {np.linalg.norm(model_state['weights']):.4f}")


# ============================================================================
# Pipeline Creation
# ============================================================================

def create_federated_pipeline(
    n_workers: int = 10,
    data_dir: str = None,
    byzantine_fraction: float = 0.0,
    aggregation_method: str = "median"
) -> Tube:
    """Create federated learning pipeline"""

    nodes = {}
    edges = []

    # Data coordinator
    nodes["coordinator"] = Node(
        id="coordinator",
        processor=DataCoordinator(n_workers=n_workers),
        signals=["round_num", "model_weights", "model_bias", "learning_rate"]
    )

    # Determine which workers are Byzantine
    n_byzantine = int(n_workers * byzantine_fraction)
    byzantine_workers = set(np.random.choice(n_workers, n_byzantine, replace=False))

    # Create worker nodes
    for i in range(n_workers):
        is_byzantine = i in byzantine_workers

        worker_node = Node(
            id=f"worker_{i}",
            processor=FederatedWorker(
                worker_id=i,
                data_dir=data_dir or str(TEST_DATA_DIR),
                is_byzantine=is_byzantine
            ),
            slots=["round_num", "model_weights", "model_bias", "learning_rate"],
            signals=["worker_gradients"]
        )
        nodes[f"worker_{i}"] = worker_node

        # Connect coordinator to worker
        for signal in ["round_num", "model_weights", "model_bias", "learning_rate"]:
            edges.append(Edge(
                source_node="coordinator",
                source_signal=signal,
                target_node=f"worker_{i}",
                target_slot=signal
            ))

    # Gradient aggregator
    nodes["aggregator"] = Node(
        id="aggregator",
        processor=GradientAggregator(aggregation_method=aggregation_method),
        slots=["worker_gradients"],
        signals=["aggregated_gradients", "contribution_scores", "n_workers"]
    )

    # Connect workers to aggregator
    for i in range(n_workers):
        edges.append(Edge(
            source_node=f"worker_{i}",
            source_signal="worker_gradients",
            target_node="aggregator",
            target_slot="worker_gradients"
        ))

    # Model updater
    nodes["updater"] = Node(
        id="updater",
        processor=ModelUpdater(),
        slots=["aggregated_gradients", "contribution_scores", "n_workers"],
        signals=["model_state"]
    )

    # Connect aggregator to updater
    for signal in ["aggregated_gradients", "contribution_scores", "n_workers"]:
        edges.append(Edge(
            source_node="aggregator",
            source_signal=signal,
            target_node="updater",
            target_slot=signal
        ))

    # Create tube
    tube = Tube(nodes=nodes, edges=edges)

    return tube


# ============================================================================
# Main Demo
# ============================================================================

def run_demo(
    n_workers: int = 10,
    n_rounds: int = 20,
    aggregation: str = "median",
    byzantine: float = 0.0
):
    """Run federated learning demo"""

    print("\n" + "🧠"*40)
    print("FEDERATED MACHINE LEARNING TRAINING DEMO")
    print("🧠"*40)

    print(f"""
    Workers: {n_workers}
    Training Rounds: {n_rounds}
    Aggregation: {aggregation}
    Byzantine Fraction: {byzantine * 100:.0f}%
    Dataset: {TEST_DATA_DIR}
    """)

    # Reset global model
    global GLOBAL_MODEL
    GLOBAL_MODEL = GlobalModelState(n_features=100)

    # Create pipeline
    tube = create_federated_pipeline(
        n_workers=n_workers,
        data_dir=str(TEST_DATA_DIR),
        byzantine_fraction=byzantine,
        aggregation_method=aggregation
    )

    # Create runner
    print("🚀 Initializing runner...")
    runner = SynchronousRunner(tube)

    # Run training
    print("\n🏋️  Starting federated training...")
    print("-"*80)

    runner.init()

    start_time = time.time()

    try:
        for round_num in range(n_rounds):
            result = runner.process()

            if result and "model_state" in result:
                print(f"   Round {round_num + 1}/{n_rounds}: "
                      f"LR={result['model_state']['learning_rate']:.5f}, "
                      f"Workers={result.get('n_workers', 0)}")

    finally:
        runner.deinit()

    elapsed = time.time() - start_time

    print("\n✨ Training Complete!")
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Time per round: {elapsed / n_rounds:.2f}s")


def main():
    """Parse arguments and run demo"""
    parser = argparse.ArgumentParser(
        description="Federated Machine Learning Training Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --workers 10 --rounds 20         # 10 workers, 20 training rounds
  %(prog)s --aggregation median             # Median aggregation (robust)
  %(prog)s --aggregation krum               # Krum (Byzantine-robust)
  %(prog)s --byzantine 0.2                  # 20%% Byzantine workers

First time setup:
  python generate_test_data.py             # Generate synthetic datasets
        """
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of federated workers (default: 10)"
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Number of training rounds (default: 20)"
    )

    parser.add_argument(
        "--aggregation",
        choices=["median", "mean", "krum", "trimmed_mean"],
        default="median",
        help="Gradient aggregation method (default: median)"
    )

    parser.add_argument(
        "--byzantine",
        type=float,
        default=0.0,
        help="Fraction of Byzantine workers (default: 0.0)"
    )

    args = parser.parse_args()

    run_demo(
        n_workers=args.workers,
        n_rounds=args.rounds,
        aggregation=args.aggregation,
        byzantine=args.byzantine
    )


if __name__ == "__main__":
    main()
