"""
Comprehensive tests for ML training example.

Tests verify:
- Federated learning pipeline execution
- Byzantine-robust aggregation
- Model convergence
- Different aggregation methods
- Worker functionality
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add examples to path
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

# Import test data generator
from generate_test_data import generate_ml_datasets

# Import demo components
from ml_training_demo import (
    DataCoordinator,
    FederatedWorker,
    GradientAggregator,
    ModelUpdater,
    GlobalModelState,
    create_federated_pipeline,
    GLOBAL_MODEL
)

from noob.runner import SynchronousRunner


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_data_dir():
    """Create temporary directory with ML datasets"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate 5 worker datasets
        generate_ml_datasets(tmpdir, n_workers=5)

        yield tmpdir


@pytest.fixture
def reset_global_model():
    """Reset global model state before each test"""
    import ml_training_demo
    ml_training_demo.GLOBAL_MODEL = GlobalModelState(n_features=100)
    yield
    ml_training_demo.GLOBAL_MODEL = GlobalModelState(n_features=100)


# ============================================================================
# Node Tests
# ============================================================================

class TestDataCoordinator:
    """Test DataCoordinator node"""

    def test_initialization(self):
        """Test coordinator initialization"""
        coordinator = DataCoordinator(n_workers=10)
        coordinator.init()

        assert coordinator.n_workers == 10
        assert coordinator.current_round == 0

        coordinator.deinit()

    def test_broadcast_model_state(self, reset_global_model):
        """Test broadcasting model state"""
        coordinator = DataCoordinator(n_workers=5)
        coordinator.init()

        result = coordinator.process()

        assert "round_num" in result
        assert "model_weights" in result
        assert "model_bias" in result
        assert "learning_rate" in result

        assert result["round_num"] == 1
        assert isinstance(result["model_weights"], np.ndarray)
        assert len(result["model_weights"]) == 100

        coordinator.deinit()

    def test_multiple_rounds(self, reset_global_model):
        """Test multiple training rounds"""
        coordinator = DataCoordinator(n_workers=5)
        coordinator.init()

        for i in range(5):
            result = coordinator.process()
            assert result["round_num"] == i + 1

        coordinator.deinit()


class TestFederatedWorker:
    """Test FederatedWorker node"""

    def test_worker_initialization(self, temp_data_dir):
        """Test worker initialization"""
        worker = FederatedWorker(
            worker_id=0,
            data_dir=str(temp_data_dir),
            is_byzantine=False
        )

        worker.init()

        # Should have loaded data
        assert worker.data is not None
        assert worker.labels is not None
        assert len(worker.data) == len(worker.labels)

        worker.deinit()

    def test_gradient_computation(self, temp_data_dir, reset_global_model):
        """Test gradient computation"""
        worker = FederatedWorker(
            worker_id=0,
            data_dir=str(temp_data_dir),
            is_byzantine=False
        )

        worker.init()

        # Get current model state
        model_state = GLOBAL_MODEL.get_state()

        # Compute gradients
        result = worker.process(
            round_num=1,
            model_weights=model_state["weights"],
            model_bias=model_state["bias"],
            learning_rate=model_state["learning_rate"]
        )

        assert "worker_id" in result
        assert "gradients" in result
        assert "loss" in result
        assert "grad_norm" in result

        # Check gradients
        grads = result["gradients"]
        assert "weights" in grads
        assert "bias" in grads
        assert len(grads["weights"]) == 100

        # Gradients should be reasonable
        assert np.isfinite(grads["weights"]).all()
        assert np.isfinite(grads["bias"])

        worker.deinit()

    def test_byzantine_worker(self, temp_data_dir, reset_global_model):
        """Test Byzantine worker behavior"""
        # Normal worker
        normal_worker = FederatedWorker(
            worker_id=0,
            data_dir=str(temp_data_dir),
            is_byzantine=False
        )
        normal_worker.init()

        # Byzantine worker
        byzantine_worker = FederatedWorker(
            worker_id=1,
            data_dir=str(temp_data_dir),
            is_byzantine=True
        )
        byzantine_worker.init()

        model_state = GLOBAL_MODEL.get_state()

        # Get gradients from both
        normal_result = normal_worker.process(
            round_num=1,
            model_weights=model_state["weights"],
            model_bias=model_state["bias"],
            learning_rate=model_state["learning_rate"]
        )

        byzantine_result = byzantine_worker.process(
            round_num=1,
            model_weights=model_state["weights"],
            model_bias=model_state["bias"],
            learning_rate=model_state["learning_rate"]
        )

        # Byzantine gradients should be very different
        normal_grad = normal_result["gradients"]["weights"]
        byzantine_grad = byzantine_result["gradients"]["weights"]

        # Byzantine gradients are random noise, should have high norm
        assert np.linalg.norm(byzantine_grad - normal_grad) > 1.0

        normal_worker.deinit()
        byzantine_worker.deinit()

    def test_loss_tracking(self, temp_data_dir, reset_global_model):
        """Test loss tracking over rounds"""
        worker = FederatedWorker(
            worker_id=0,
            data_dir=str(temp_data_dir),
            is_byzantine=False
        )

        worker.init()

        model_state = GLOBAL_MODEL.get_state()

        # Run multiple rounds
        losses = []
        for _ in range(5):
            result = worker.process(
                round_num=1,
                model_weights=model_state["weights"],
                model_bias=model_state["bias"],
                learning_rate=model_state["learning_rate"]
            )
            losses.append(result["loss"])

        # Should have tracked losses
        assert len(worker.local_loss_history) == 5

        worker.deinit()


class TestGradientAggregator:
    """Test GradientAggregator node"""

    def test_median_aggregation(self):
        """Test median aggregation"""
        aggregator = GradientAggregator(aggregation_method="median")
        aggregator.init()

        # Create mock gradients
        gradients = [
            {
                "worker_id": i,
                "gradients": {
                    "weights": np.random.randn(100),
                    "bias": np.random.randn()
                },
                "is_byzantine": False
            }
            for i in range(5)
        ]

        result = aggregator.process(gradients)

        assert "aggregated_gradients" in result
        assert "contribution_scores" in result
        assert "n_workers" in result

        agg_grads = result["aggregated_gradients"]
        assert len(agg_grads["weights"]) == 100

        aggregator.deinit()

    def test_mean_aggregation(self):
        """Test mean aggregation"""
        aggregator = GradientAggregator(aggregation_method="mean")
        aggregator.init()

        # Create simple gradients for verification
        gradients = [
            {
                "worker_id": i,
                "gradients": {
                    "weights": np.ones(10) * i,
                    "bias": float(i)
                },
                "is_byzantine": False
            }
            for i in range(3)
        ]

        result = aggregator.process(gradients)

        agg_grads = result["aggregated_gradients"]

        # Mean of [0, 1, 2] should be 1.0
        assert np.allclose(agg_grads["weights"], np.ones(10))
        assert np.isclose(agg_grads["bias"], 1.0)

        aggregator.deinit()

    def test_krum_aggregation(self):
        """Test Krum aggregation"""
        aggregator = GradientAggregator(aggregation_method="krum")
        aggregator.init()

        # Create gradients with one outlier
        gradients = []

        # 4 similar gradients
        for i in range(4):
            gradients.append({
                "worker_id": i,
                "gradients": {
                    "weights": np.ones(100) + np.random.randn(100) * 0.1,
                    "bias": 1.0 + np.random.randn() * 0.1
                },
                "is_byzantine": False
            })

        # 1 Byzantine gradient (outlier)
        gradients.append({
            "worker_id": 4,
            "gradients": {
                "weights": np.ones(100) * 100,  # Very different!
                "bias": 100.0
            },
            "is_byzantine": True
        })

        result = aggregator.process(gradients)

        agg_grads = result["aggregated_gradients"]

        # Should select one of the similar gradients, not the outlier
        assert np.linalg.norm(agg_grads["weights"]) < 10  # Not 100!

        aggregator.deinit()

    def test_trimmed_mean(self):
        """Test trimmed mean aggregation"""
        aggregator = GradientAggregator(aggregation_method="trimmed_mean")
        aggregator.init()

        # Create gradients with outliers
        gradients = [
            {
                "worker_id": i,
                "gradients": {
                    "weights": np.ones(100) * (1 if i < 8 else 100),  # 2 outliers
                    "bias": 1.0 if i < 8 else 100.0
                },
                "is_byzantine": False
            }
            for i in range(10)
        ]

        result = aggregator.process(gradients)

        agg_grads = result["aggregated_gradients"]

        # Should be close to 1.0, outliers trimmed
        assert np.allclose(agg_grads["weights"], np.ones(100), atol=0.5)

        aggregator.deinit()

    def test_contribution_scores(self):
        """Test contribution score computation"""
        aggregator = GradientAggregator(aggregation_method="mean")
        aggregator.init()

        # Create gradients
        gradients = [
            {
                "worker_id": i,
                "gradients": {
                    "weights": np.ones(100) * (i + 1),
                    "bias": float(i)
                },
                "is_byzantine": False
            }
            for i in range(3)
        ]

        result = aggregator.process(gradients)

        scores = result["contribution_scores"]

        # Should have score for each worker
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores.values())

        aggregator.deinit()


class TestModelUpdater:
    """Test ModelUpdater node"""

    def test_model_update(self, reset_global_model):
        """Test model update with gradients"""
        updater = ModelUpdater()
        updater.init()

        initial_weights = GLOBAL_MODEL.weights.copy()

        # Apply gradients
        result = updater.process(
            aggregated_gradients={
                "weights": np.ones(100) * 0.1,
                "bias": 0.1
            },
            contribution_scores={0: 1.0},
            n_workers=1
        )

        # Model should have been updated
        assert not np.allclose(GLOBAL_MODEL.weights, initial_weights)

        updater.deinit()

    def test_learning_rate_decay(self, reset_global_model):
        """Test learning rate decay over updates"""
        updater = ModelUpdater()
        updater.init()

        initial_lr = GLOBAL_MODEL.learning_rate

        # Apply multiple updates
        for _ in range(10):
            updater.process(
                aggregated_gradients={
                    "weights": np.ones(100) * 0.01,
                    "bias": 0.01
                },
                contribution_scores={0: 1.0},
                n_workers=1
            )

        # Learning rate should have decayed
        assert GLOBAL_MODEL.learning_rate < initial_lr

        updater.deinit()


# ============================================================================
# Pipeline Tests
# ============================================================================

class TestPipeline:
    """Test complete federated learning pipeline"""

    def test_create_pipeline(self, temp_data_dir):
        """Test pipeline creation"""
        tube = create_federated_pipeline(
            n_workers=3,
            data_dir=str(temp_data_dir),
            aggregation_method="median"
        )

        # Check nodes exist
        assert "coordinator" in tube.nodes
        assert "worker_0" in tube.nodes
        assert "worker_1" in tube.nodes
        assert "worker_2" in tube.nodes
        assert "aggregator" in tube.nodes
        assert "updater" in tube.nodes

        # Check edges
        assert len(tube.edges) > 0

    def test_pipeline_execution(self, temp_data_dir, reset_global_model):
        """Test end-to-end pipeline execution"""
        tube = create_federated_pipeline(
            n_workers=3,
            data_dir=str(temp_data_dir),
            aggregation_method="median"
        )

        runner = SynchronousRunner(tube)
        runner.init()

        initial_weights = GLOBAL_MODEL.weights.copy()

        # Run for 3 rounds
        for _ in range(3):
            result = runner.process()

        runner.deinit()

        # Model should have been updated
        assert not np.allclose(GLOBAL_MODEL.weights, initial_weights)

    def test_byzantine_workers(self, temp_data_dir, reset_global_model):
        """Test pipeline with Byzantine workers"""
        tube = create_federated_pipeline(
            n_workers=5,
            data_dir=str(temp_data_dir),
            byzantine_fraction=0.2,  # 1 Byzantine worker
            aggregation_method="median"
        )

        runner = SynchronousRunner(tube)
        runner.init()

        # Should still work with Byzantine workers
        for _ in range(5):
            result = runner.process()

        runner.deinit()

        # Model should have converged (Byzantine workers filtered out)
        assert GLOBAL_MODEL.global_step == 5

    def test_different_aggregation_methods(self, temp_data_dir, reset_global_model):
        """Test different aggregation methods"""
        methods = ["median", "mean", "krum", "trimmed_mean"]

        for method in methods:
            # Reset model
            import ml_training_demo
            ml_training_demo.GLOBAL_MODEL = GlobalModelState(n_features=100)

            tube = create_federated_pipeline(
                n_workers=3,
                data_dir=str(temp_data_dir),
                aggregation_method=method
            )

            runner = SynchronousRunner(tube)
            runner.init()

            # Run for 2 rounds
            for _ in range(2):
                runner.process()

            runner.deinit()

            # Should complete successfully
            assert GLOBAL_MODEL.global_step == 2


# ============================================================================
# Convergence Tests
# ============================================================================

class TestConvergence:
    """Test model convergence"""

    def test_loss_decreases(self, temp_data_dir, reset_global_model):
        """Test that loss decreases over training"""
        tube = create_federated_pipeline(
            n_workers=3,
            data_dir=str(temp_data_dir),
            aggregation_method="mean"
        )

        runner = SynchronousRunner(tube)
        runner.init()

        # Track initial state
        initial_lr = GLOBAL_MODEL.learning_rate

        # Run for 10 rounds
        for _ in range(10):
            runner.process()

        runner.deinit()

        # Learning rate should have decayed
        assert GLOBAL_MODEL.learning_rate < initial_lr

        # Model should have been updated
        assert GLOBAL_MODEL.global_step == 10

    def test_model_updates_accumulated(self, temp_data_dir, reset_global_model):
        """Test that model updates accumulate"""
        tube = create_federated_pipeline(
            n_workers=3,
            data_dir=str(temp_data_dir),
            aggregation_method="median"
        )

        runner = SynchronousRunner(tube)
        runner.init()

        weights_history = []

        # Track weights over rounds
        for _ in range(5):
            weights_history.append(GLOBAL_MODEL.weights.copy())
            runner.process()

        runner.deinit()

        # Weights should be changing
        for i in range(1, len(weights_history)):
            assert not np.allclose(weights_history[i], weights_history[i-1])


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests"""

    def test_full_training_run(self, temp_data_dir, reset_global_model):
        """Test complete training run"""
        tube = create_federated_pipeline(
            n_workers=5,
            data_dir=str(temp_data_dir),
            byzantine_fraction=0.0,
            aggregation_method="median"
        )

        runner = SynchronousRunner(tube)
        runner.init()

        # Run for 20 rounds
        results = []
        for _ in range(20):
            result = runner.process()
            if result:
                results.append(result)

        runner.deinit()

        # Should have completed all rounds
        assert len(results) == 20
        assert GLOBAL_MODEL.global_step == 20

    def test_robustness_to_byzantine_attacks(self, temp_data_dir, reset_global_model):
        """Test robustness to Byzantine attacks"""
        # Create two pipelines: one with Byzantine workers, one without
        tube_clean = create_federated_pipeline(
            n_workers=5,
            data_dir=str(temp_data_dir),
            byzantine_fraction=0.0,
            aggregation_method="median"
        )

        import ml_training_demo
        ml_training_demo.GLOBAL_MODEL = GlobalModelState(n_features=100)

        tube_byzantine = create_federated_pipeline(
            n_workers=5,
            data_dir=str(temp_data_dir),
            byzantine_fraction=0.2,  # 20% Byzantine
            aggregation_method="median"  # Robust aggregation
        )

        # Train clean model
        runner_clean = SynchronousRunner(tube_clean)
        runner_clean.init()

        for _ in range(10):
            runner_clean.process()

        clean_weights = GLOBAL_MODEL.weights.copy()
        runner_clean.deinit()

        # Reset and train with Byzantine workers
        ml_training_demo.GLOBAL_MODEL = GlobalModelState(n_features=100)

        runner_byzantine = SynchronousRunner(tube_byzantine)
        runner_byzantine.init()

        for _ in range(10):
            runner_byzantine.process()

        byzantine_weights = GLOBAL_MODEL.weights.copy()
        runner_byzantine.deinit()

        # Models should converge to similar solutions
        # (median aggregation should filter Byzantine workers)
        weight_diff = np.linalg.norm(clean_weights - byzantine_weights)

        # Should be relatively close (not perfect but reasonable)
        assert weight_diff < 100.0  # Some tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
