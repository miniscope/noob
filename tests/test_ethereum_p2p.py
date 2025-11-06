"""
Tests for Ethereum P2P Image Processing

Verifies end-to-end integration of:
- Rust P2P components (ContentStore, TaskRegistry, P2PNode)
- Ethereum smart contract simulation
- Distributed image processing
- Content-addressed storage with Blake3
"""

import pytest
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile

import noob_core
from examples.ethereum_p2p_processing import (
    EthereumContract,
    P2PImageProcessor
)


class TestRustP2PComponents:
    """Test Rust P2P components directly"""

    def test_content_id_creation(self):
        """Test Blake3-based content IDs"""
        data1 = b"hello world"
        data2 = b"hello world"
        data3 = b"different"

        cid1 = noob_core.ContentId(list(data1))
        cid2 = noob_core.ContentId(list(data2))
        cid3 = noob_core.ContentId(list(data3))

        # Same data = same hash
        assert cid1.hash == cid2.hash

        # Different data = different hash
        assert cid1.hash != cid3.hash

        # Hash is hex string
        assert len(cid1.hash) == 64  # Blake3 = 256 bits = 64 hex chars
        assert all(c in '0123456789abcdef' for c in cid1.hash)

    def test_content_store_operations(self):
        """Test content-addressed storage"""
        store = noob_core.ContentStore()

        # Store data
        data = b"test data"
        cid = store.put(list(data))

        # Retrieve data
        retrieved = store.get(cid.hash)
        assert retrieved is not None
        assert bytes(retrieved) == data

        # Check contains
        assert store.contains(cid.hash)
        assert not store.contains("nonexistent")

        # Check size
        assert store.size() == 1

        # Clear
        store.clear()
        assert store.size() == 0

    def test_task_metadata(self):
        """Test task metadata creation and serialization"""
        task = noob_core.TaskMetadata(
            task_id="task123",
            node_id="node456",
            worker_address="0xabc",
            gas_paid=1000
        )

        assert task.task_id == "task123"
        assert task.node_id == "node456"
        assert task.worker_address == "0xabc"
        assert task.gas_paid == 1000
        assert task.status == "pending"
        assert task.result_hash is None

        # JSON serialization
        json_str = task.to_json()
        assert "task123" in json_str
        assert "node456" in json_str

        # Deserialization
        task2 = noob_core.TaskMetadata.from_json(json_str)
        assert task2.task_id == task.task_id
        assert task2.node_id == task.node_id

    def test_task_registry_operations(self):
        """Test task registry"""
        registry = noob_core.TaskRegistry()

        # Register task
        task1 = noob_core.TaskMetadata(
            task_id="task1",
            node_id="node1",
            worker_address="0xaaa",
            gas_paid=500
        )

        task_id = registry.register_task(task1)
        assert task_id == "task1"

        # Get task
        retrieved = registry.get_task("task1")
        assert retrieved is not None
        assert retrieved.task_id == "task1"

        # Get by status
        pending = registry.get_by_status("pending")
        assert len(pending) == 1
        assert pending[0].task_id == "task1"

        # Update status
        registry.update_status("task1", "claimed")
        pending = registry.get_by_status("pending")
        assert len(pending) == 0

        claimed = registry.get_by_status("claimed")
        assert len(claimed) == 1

        # Set result
        registry.set_result("task1", "result_hash_123")
        task = registry.get_task("task1")
        assert task.result_hash == "result_hash_123"

        # Stats
        stats = registry.stats()
        assert stats["claimed"] == 1
        assert registry.total_tasks() == 1

    def test_task_registry_lifecycle(self):
        """Test full task lifecycle in registry"""
        registry = noob_core.TaskRegistry()

        # Create task
        task = noob_core.TaskMetadata(
            task_id="lifecycle_task",
            node_id="node1",
            worker_address="0x123",
            gas_paid=1000
        )

        # 1. Register (pending)
        registry.register_task(task)
        assert registry.count_by_status("pending") == 1

        # 2. Claim
        registry.update_status("lifecycle_task", "claimed")
        assert registry.count_by_status("pending") == 0
        assert registry.count_by_status("claimed") == 1

        # 3. Complete
        registry.set_result("lifecycle_task", "result_abc")
        registry.update_status("lifecycle_task", "completed")
        assert registry.count_by_status("completed") == 1

        # 4. Verify
        registry.update_status("lifecycle_task", "verified")
        assert registry.count_by_status("verified") == 1

        final_task = registry.get_task("lifecycle_task")
        assert final_task.status == "verified"
        assert final_task.result_hash == "result_abc"


class TestEthereumContract:
    """Test Ethereum smart contract simulation"""

    def test_contract_creation(self):
        """Test contract initialization"""
        contract = EthereumContract()

        assert contract.address.startswith("0x")
        assert contract.balance == 100_000
        assert len(contract.tasks) == 0

    def test_submit_task(self):
        """Test task submission"""
        contract = EthereumContract()

        task_id = contract.submit_task("task1", 500)
        assert task_id == "task1"
        assert contract.balance == 99_500

        # Check task recorded
        task = contract.get_task_status("task1")
        assert task is not None
        assert task["status"] == "pending"
        assert task["gas_paid"] == 500

    def test_claim_and_complete_task(self):
        """Test task claiming and completion"""
        contract = EthereumContract()

        contract.submit_task("task1", 500)

        # Claim
        claimed = contract.claim_task("task1", "worker1")
        assert claimed
        task = contract.get_task_status("task1")
        assert task["status"] == "claimed"
        assert task["worker"] == "worker1"

        # Can't claim twice
        claimed = contract.claim_task("task1", "worker2")
        assert not claimed

        # Submit result
        submitted = contract.submit_result("task1", "result_hash")
        assert submitted
        task = contract.get_task_status("task1")
        assert task["status"] == "completed"
        assert task["result_hash"] == "result_hash"

    def test_verify_result(self):
        """Test result verification and payment"""
        contract = EthereumContract()

        contract.submit_task("task1", 1000)
        contract.claim_task("task1", "worker1")
        contract.submit_result("task1", "correct_hash")

        # Verify correct hash
        verified = contract.verify_result("task1", "correct_hash")
        assert verified

        task = contract.get_task_status("task1")
        assert task["status"] == "verified"
        assert task["verified"]

        # Check payment recorded
        assert "task1" in contract.verifications
        verification = contract.verifications["task1"]
        assert verification["worker"] == "worker1"
        assert verification["payment"] == 1000

        # Verify wrong hash
        contract.submit_task("task2", 500)
        contract.claim_task("task2", "worker2")
        contract.submit_result("task2", "wrong_hash")

        verified = contract.verify_result("task2", "correct_hash")
        assert not verified

        task2 = contract.get_task_status("task2")
        assert task2["status"] == "disputed"


class TestP2PImageProcessor:
    """Test complete P2P image processor"""

    @pytest.fixture
    def temp_image(self):
        """Create temporary test image"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = Image.fromarray(
                np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            )
            img.save(f.name)
            yield f.name
            Path(f.name).unlink()

    @pytest.fixture
    def setup_network(self):
        """Setup P2P network with contract and nodes"""
        # Reset shared stores
        P2PImageProcessor._shared_content_store = None
        P2PImageProcessor._shared_task_registry = None

        contract = EthereumContract()
        coordinator = P2PImageProcessor("0x111", contract)
        worker = P2PImageProcessor("0x222", contract)

        return contract, coordinator, worker

    def test_submit_image_task(self, temp_image, setup_network):
        """Test submitting image processing task"""
        contract, coordinator, worker = setup_network

        task_id = coordinator.submit_image_task(
            temp_image,
            "blur",
            gas_amount=500
        )

        # Check task ID is Blake3 hash
        assert len(task_id) == 64
        assert all(c in '0123456789abcdef' for c in task_id)

        # Check blockchain recorded
        bc_task = contract.get_task_status(task_id)
        assert bc_task is not None
        assert bc_task["gas_paid"] == 500

        # Check P2P registry
        p2p_task = coordinator.task_registry.get_task(task_id)
        assert p2p_task is not None
        assert p2p_task.status == "pending"

    def test_worker_processes_tasks(self, temp_image, setup_network):
        """Test worker claiming and processing tasks"""
        contract, coordinator, worker = setup_network

        # Submit task
        task_id = coordinator.submit_image_task(temp_image, "blur", 500)

        # Worker processes
        processed = worker.process_pending_tasks()

        assert len(processed) == 1
        assert processed[0] == task_id

        # Check status updated
        bc_task = contract.get_task_status(task_id)
        assert bc_task["status"] == "completed"
        assert "result_hash" in bc_task

        p2p_task = worker.task_registry.get_task(task_id)
        assert p2p_task.status == "completed"
        assert p2p_task.result_hash is not None

    def test_coordinator_verifies_results(self, temp_image, setup_network):
        """Test coordinator verifying worker results"""
        contract, coordinator, worker = setup_network

        # Submit and process
        task_id = coordinator.submit_image_task(temp_image, "edge_detect", 500)
        worker.process_pending_tasks()

        # Verify
        verified = coordinator.verify_task_result(task_id)
        assert verified

        # Check payment made
        assert task_id in contract.verifications

    def test_get_result(self, temp_image, setup_network):
        """Test retrieving processed results"""
        contract, coordinator, worker = setup_network

        # Submit, process, verify
        task_id = coordinator.submit_image_task(temp_image, "fire_detect", 500)
        worker.process_pending_tasks()
        coordinator.verify_task_result(task_id)

        # Get result
        result = coordinator.get_result(task_id)

        assert result is not None
        assert result["operation"] == "fire_detect"
        assert "result" in result
        assert "processed_at" in result

    def test_multiple_tasks_multiple_workers(self, temp_image, setup_network):
        """Test multiple tasks across multiple workers"""
        contract, coordinator, worker1 = setup_network
        worker2 = P2PImageProcessor("0x333", contract)

        # Submit multiple tasks
        operations = ["blur", "edge_detect", "fire_detect"]
        task_ids = []

        for op in operations:
            task_id = coordinator.submit_image_task(temp_image, op, 500)
            task_ids.append(task_id)

        # Workers process in parallel (simulated)
        processed1 = worker1.process_pending_tasks()
        processed2 = worker2.process_pending_tasks()

        total_processed = len(processed1) + len(processed2)
        assert total_processed == 3

        # Verify all
        for task_id in task_ids:
            verified = coordinator.verify_task_result(task_id)
            assert verified

    def test_stats(self, temp_image, setup_network):
        """Test statistics collection"""
        contract, coordinator, worker = setup_network

        # Initial stats
        stats = coordinator.get_stats()
        assert stats["tasks_submitted"] == 0
        assert stats["content_items"] == 0

        # Submit and process
        coordinator.submit_image_task(temp_image, "blur", 500)
        worker.process_pending_tasks()

        # Updated stats
        stats = coordinator.get_stats()
        assert stats["tasks_submitted"] == 1
        assert stats["content_items"] == 2  # Task + result
        assert "pending" not in stats["p2p"] or stats["p2p"]["pending"] == 0


class TestIntegration:
    """End-to-end integration tests"""

    def test_full_workflow_with_real_images(self):
        """Test complete workflow with actual satellite images"""
        # Reset shared stores
        P2PImageProcessor._shared_content_store = None
        P2PImageProcessor._shared_task_registry = None

        contract = EthereumContract()
        coordinator = P2PImageProcessor("0xCoordinator", contract)
        worker = P2PImageProcessor("0xWorker", contract)

        # Get test images
        test_dir = Path("examples/test_data/satellite_images")
        if not test_dir.exists():
            pytest.skip("Test data not available")

        images = list(test_dir.glob("*.jpg"))[:2]
        if len(images) < 2:
            pytest.skip("Not enough test images")

        # Submit tasks
        task_ids = []
        for img_path in images:
            task_id = coordinator.submit_image_task(
                str(img_path),
                "fire_detect",
                gas_amount=750
            )
            task_ids.append(task_id)

        # Process
        processed = worker.process_pending_tasks()
        assert len(processed) == len(task_ids)

        # Verify
        verified_count = 0
        for task_id in task_ids:
            if coordinator.verify_task_result(task_id):
                verified_count += 1

                # Retrieve and check result
                result = coordinator.get_result(task_id)
                assert result is not None
                assert result["operation"] == "fire_detect"
                assert isinstance(result["result"], np.ndarray)

        assert verified_count == len(task_ids)

        # Final checks
        assert contract.balance < 100_000  # Gas spent
        assert len(contract.verifications) == len(task_ids)

        print(f"\n✅ Processed {len(task_ids)} images with P2P + Ethereum")
        print(f"   Gas spent: {100_000 - contract.balance} wei")
        print(f"   Verifications: {len(contract.verifications)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
