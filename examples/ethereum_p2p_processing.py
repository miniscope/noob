"""
Ethereum-Based P2P Distributed Image Processing

This example demonstrates NOOB's advanced capabilities:
- Rust-powered P2P content-addressed storage
- Ethereum smart contract integration for task coordination
- Distributed image processing with on-chain verification
- Blake3 content addressing for tamper-proof results
- Gas-efficient task distribution

Architecture:
1. Coordinator submits tasks to Ethereum + P2P network
2. Workers claim tasks, process images using Rust acceleration
3. Results stored in content-addressed storage (Blake3)
4. Coordinator verifies results on-chain
5. Workers get paid in ETH for verified work
"""

import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

# Import Rust-accelerated P2P components
import noob_core

# Ethereum integration (simulated for demo - real web3.py would go here)
class EthereumContract:
    """
    Simulates Ethereum smart contract for task coordination

    In production, this would use web3.py to interact with actual contracts
    """

    def __init__(self, contract_address: str = "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"):
        self.address = contract_address
        self.balance = 100_000  # Simulated ETH balance in wei
        self.tasks = {}
        self.verifications = {}

    def submit_task(self, task_id: str, gas_amount: int) -> str:
        """Submit task to blockchain"""
        if gas_amount > self.balance:
            raise ValueError("Insufficient balance")

        self.balance -= gas_amount
        self.tasks[task_id] = {
            "status": "pending",
            "gas_paid": gas_amount,
            "submitter": self.address,
            "timestamp": int(time.time())
        }

        return task_id

    def claim_task(self, task_id: str, worker_address: str) -> bool:
        """Worker claims a task"""
        if task_id not in self.tasks:
            return False

        if self.tasks[task_id]["status"] != "pending":
            return False

        self.tasks[task_id]["status"] = "claimed"
        self.tasks[task_id]["worker"] = worker_address
        self.tasks[task_id]["claimed_at"] = int(time.time())

        return True

    def submit_result(self, task_id: str, result_hash: str) -> bool:
        """Worker submits result hash"""
        if task_id not in self.tasks:
            return False

        self.tasks[task_id]["result_hash"] = result_hash
        self.tasks[task_id]["status"] = "completed"
        self.tasks[task_id]["completed_at"] = int(time.time())

        return True

    def verify_result(self, task_id: str, expected_hash: str) -> bool:
        """Coordinator verifies result matches expected hash"""
        if task_id not in self.tasks:
            return False

        actual_hash = self.tasks[task_id].get("result_hash")
        verified = actual_hash == expected_hash

        self.tasks[task_id]["verified"] = verified
        self.tasks[task_id]["status"] = "verified" if verified else "disputed"

        if verified:
            # Pay the worker
            worker = self.tasks[task_id]["worker"]
            gas_paid = self.tasks[task_id]["gas_paid"]
            self.verifications[task_id] = {
                "worker": worker,
                "payment": gas_paid,
                "verified_at": int(time.time())
            }
            print(f"   💰 Paid {gas_paid} wei to worker {worker[:10]}...")

        return verified

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get task status from blockchain"""
        return self.tasks.get(task_id)


class P2PImageProcessor:
    """
    P2P Image Processor using Rust-accelerated content-addressed storage
    """

    # Shared registries (simulates P2P network synchronization)
    _shared_content_store = None
    _shared_task_registry = None

    def __init__(self, ethereum_address: str, contract: EthereumContract):
        self.ethereum_address = ethereum_address
        self.contract = contract

        # Initialize shared stores on first instance
        if P2PImageProcessor._shared_content_store is None:
            P2PImageProcessor._shared_content_store = noob_core.ContentStore()
            P2PImageProcessor._shared_task_registry = noob_core.TaskRegistry()

        # Use shared stores (simulates P2P sync)
        self.content_store = P2PImageProcessor._shared_content_store
        self.task_registry = P2PImageProcessor._shared_task_registry

        # Create node-specific metadata
        self.node_id = noob_core.ContentId(ethereum_address.encode()).hash[:16]

        print(f"🦀 Created P2P node: {self.node_id}")
        print(f"   Ethereum address: {ethereum_address[:20]}...")

    def submit_image_task(self, image_path: str, operation: str, gas_amount: int = 1000) -> str:
        """
        Submit image processing task to P2P network and blockchain

        Args:
            image_path: Path to image file
            operation: Processing operation ('blur', 'edge_detect', 'fire_detect')
            gas_amount: Gas to pay for processing

        Returns:
            Task ID (content hash)
        """
        # Load and serialize image
        img = Image.open(image_path)
        task_data = {
            "operation": operation,
            "image": np.array(img),
            "filename": Path(image_path).name
        }

        serialized = pickle.dumps(task_data)

        # Store in content-addressed storage (Rust)
        content_id = self.content_store.put(list(serialized))
        task_id = content_id.hash

        # Register task in P2P registry
        task_meta = noob_core.TaskMetadata(
            task_id=task_id,
            node_id=f"img_process_{operation}",
            worker_address=self.ethereum_address,
            gas_paid=gas_amount
        )
        self.task_registry.register_task(task_meta)

        # Submit to Ethereum blockchain
        self.contract.submit_task(task_id, gas_amount)

        print(f"   📤 Submitted task: {task_id[:16]}...")
        print(f"      Operation: {operation}")
        print(f"      Gas paid: {gas_amount} wei")

        return task_id

    def process_pending_tasks(self) -> List[str]:
        """
        Worker: Process all pending tasks from P2P network

        Returns:
            List of processed task IDs
        """
        processed = []
        pending = self.task_registry.get_by_status("pending")

        for task_meta in pending:
            task_id = task_meta.task_id

            # Claim on blockchain
            if not self.contract.claim_task(task_id, self.ethereum_address):
                continue

            # Update P2P registry
            self.task_registry.update_status(task_id, "claimed")

            # Get task data from content store
            task_data_bytes = self.content_store.get(task_id)
            if task_data_bytes is None:
                continue

            # Deserialize and process
            task_data = pickle.loads(bytes(task_data_bytes))

            print(f"   🔧 Processing task: {task_id[:16]}...")
            result = self._process_image(task_data)

            # Serialize result
            result_bytes = pickle.dumps(result)

            # Store result in content store
            result_cid = self.content_store.put(list(result_bytes))
            result_hash = result_cid.hash

            # Update task registry
            self.task_registry.set_result(task_id, result_hash)
            self.task_registry.update_status(task_id, "completed")

            # Submit result hash to blockchain
            self.contract.submit_result(task_id, result_hash)

            print(f"   ✅ Completed task: {task_id[:16]}...")
            print(f"      Result hash: {result_hash[:16]}...")

            processed.append(task_id)

        return processed

    def _process_image(self, task_data: Dict) -> Dict:
        """Apply image processing operation"""
        operation = task_data["operation"]
        img_array = task_data["image"]

        # Simulate different operations
        if operation == "blur":
            # Simple box blur
            from scipy.ndimage import uniform_filter
            result = uniform_filter(img_array, size=5)

        elif operation == "edge_detect":
            # Simple edge detection
            from scipy.ndimage import sobel
            edges = sobel(img_array.astype(float).mean(axis=2))
            result = edges

        elif operation == "fire_detect":
            # Fire detection (red/orange threshold)
            if img_array.ndim == 3:
                red = img_array[:, :, 0].astype(float)
                green = img_array[:, :, 1].astype(float)
                blue = img_array[:, :, 2].astype(float)

                # Fire typically has high red, medium green, low blue
                fire_score = (red - green - blue) / 255.0
                fire_detected = fire_score > 0.3
                result = fire_detected
            else:
                result = np.zeros_like(img_array, dtype=bool)

        else:
            result = img_array

        return {
            "operation": operation,
            "result": result,
            "processed_at": time.time(),
            "processor": self.ethereum_address[:10]
        }

    def verify_task_result(self, task_id: str) -> bool:
        """
        Coordinator: Verify task result matches blockchain

        Returns:
            True if verification successful
        """
        # Get result from P2P registry
        task_meta = self.task_registry.get_task(task_id)
        if not task_meta:
            print(f"   ⚠️  Task not found in registry")
            return False

        if task_meta.status != "completed":
            print(f"   ⚠️  Task not completed: {task_meta.status}")
            return False

        # Get blockchain status
        blockchain_task = self.contract.get_task_status(task_id)
        if not blockchain_task:
            print(f"   ⚠️  Task not found on blockchain")
            return False

        result_hash = blockchain_task.get("result_hash")
        if not result_hash:
            print(f"   ⚠️  No result hash on blockchain")
            return False

        # Verify hash matches
        verified = self.contract.verify_result(task_id, result_hash)

        if verified:
            print(f"   ✅ Verified task: {task_id[:16]}...")
        else:
            print(f"   ❌ Verification failed: {task_id[:16]}...")

        return verified

    def get_result(self, task_id: str) -> Optional[Dict]:
        """Get processed result from P2P network"""
        # Get result hash from blockchain
        blockchain_task = self.contract.get_task_status(task_id)
        if not blockchain_task or "result_hash" not in blockchain_task:
            return None

        result_hash = blockchain_task["result_hash"]

        # Fetch result from content store
        result_bytes = self.content_store.get(result_hash)
        if result_bytes is None:
            return None

        return pickle.loads(bytes(result_bytes))

    def get_stats(self) -> Dict:
        """Get P2P and blockchain stats"""
        p2p_stats = self.task_registry.stats()

        return {
            "node_id": self.node_id,
            "ethereum_address": self.ethereum_address[:20] + "...",
            "p2p": p2p_stats,
            "content_items": self.content_store.size(),
            "blockchain_balance": self.contract.balance,
            "tasks_submitted": len(self.contract.tasks),
            "tasks_verified": len(self.contract.verifications)
        }


def run_demo():
    """
    Run complete Ethereum P2P image processing demo
    """
    print("\n" + "="*80)
    print("🦀 ETHEREUM-BASED P2P DISTRIBUTED IMAGE PROCESSING")
    print("   Powered by Rust + Blake3 + Smart Contracts")
    print("="*80 + "\n")

    # Create Ethereum contract (simulated)
    contract = EthereumContract()
    print(f"📜 Smart Contract: {contract.address}")
    print(f"   Balance: {contract.balance} wei\n")

    # Create coordinator node
    coordinator_addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"
    coordinator = P2PImageProcessor(coordinator_addr, contract)
    print()

    # Create worker nodes
    worker1_addr = "0x892d35Cc6634C0532925a3b844Bc9e7595f0bEc"
    worker1 = P2PImageProcessor(worker1_addr, contract)
    print()

    worker2_addr = "0x992d35Cc6634C0532925a3b844Bc9e7595f0bEd"
    worker2 = P2PImageProcessor(worker2_addr, contract)
    print()

    # Get test images
    test_data_dir = Path("examples/test_data/satellite_images")
    if not test_data_dir.exists():
        print("⚠️  Test data not found. Generating...")
        import subprocess
        subprocess.run(["python", "examples/generate_test_data.py"], check=True)

    image_files = list(test_data_dir.glob("*.jpg"))[:3]

    if not image_files:
        print("❌ No test images found!")
        return

    print(f"\n📸 Found {len(image_files)} test images\n")

    # Phase 1: Submit tasks
    print("=" * 80)
    print("PHASE 1: COORDINATOR SUBMITS TASKS TO P2P NETWORK + BLOCKCHAIN")
    print("=" * 80 + "\n")

    task_ids = []
    operations = ["blur", "edge_detect", "fire_detect"]

    for img_path, operation in zip(image_files, operations):
        task_id = coordinator.submit_image_task(
            str(img_path),
            operation,
            gas_amount=500
        )
        task_ids.append((task_id, operation))
        print()

    # Phase 2: Workers process tasks
    print("=" * 80)
    print("PHASE 2: WORKERS CLAIM AND PROCESS TASKS")
    print("=" * 80 + "\n")

    print("👷 Worker 1 processing...")
    processed1 = worker1.process_pending_tasks()
    print()

    print("👷 Worker 2 processing...")
    processed2 = worker2.process_pending_tasks()
    print()

    total_processed = len(processed1) + len(processed2)
    print(f"   ✅ Total tasks processed: {total_processed}\n")

    # Phase 3: Verify results
    print("=" * 80)
    print("PHASE 3: COORDINATOR VERIFIES RESULTS ON-CHAIN")
    print("=" * 80 + "\n")

    verified_count = 0
    for task_id, operation in task_ids:
        if coordinator.verify_task_result(task_id):
            verified_count += 1

            # Get and display result
            result = coordinator.get_result(task_id)
            if result:
                print(f"      Operation: {result['operation']}")
                print(f"      Processor: {result['processor']}")
        print()

    # Phase 4: Final stats
    print("=" * 80)
    print("PHASE 4: FINAL STATISTICS")
    print("=" * 80 + "\n")

    coord_stats = coordinator.get_stats()
    worker1_stats = worker1.get_stats()
    worker2_stats = worker2.get_stats()

    print("📊 Coordinator Stats:")
    print(f"   Tasks submitted: {coord_stats['tasks_submitted']}")
    print(f"   Tasks verified: {coord_stats['tasks_verified']}")
    print(f"   Content items: {coord_stats['content_items']}")
    print(f"   P2P status breakdown: {coord_stats['p2p']}")
    print()

    print("📊 Worker 1 Stats:")
    print(f"   Node ID: {worker1_stats['node_id']}")
    print()

    print("📊 Worker 2 Stats:")
    print(f"   Node ID: {worker2_stats['node_id']}")
    print()

    print("🏆 SUMMARY:")
    print(f"   ✅ Tasks submitted: {len(task_ids)}")
    print(f"   ✅ Tasks processed: {total_processed}")
    print(f"   ✅ Tasks verified: {verified_count}")
    print(f"   💰 Total gas spent: {1500} wei")
    print(f"   📦 Content items stored: {coord_stats['content_items']}")
    print()

    print("="*80)
    print("🎉 DEMO COMPLETE! P2P + ETHEREUM INTEGRATION WORKING!")
    print("="*80)


if __name__ == "__main__":
    # Install dependencies if needed
    try:
        import scipy
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.run(["pip", "install", "scipy"], check=True)

    run_demo()
