#!/usr/bin/env python
"""
Blockchain-Powered Distributed Image Processing

This example demonstrates a complex real-world use case:
- Distributed image processing across multiple workers
- On-chain task coordination with ETH payments
- Worker reputation and staking
- Cryptoeconomic guarantees for result verification

Use Case: Process 10,000 satellite images to detect forest fires
- Each image is 10MB
- Processing takes ~5 seconds per image
- Workers earn ETH for successful processing
- Byzantine workers are slashed
"""

import asyncio
import pickle
import time
from pathlib import Path

import numpy as np
from PIL import Image

from noob import Tube
from noob.blockchain.ethereum import (
    BlockchainConfig,
    EthereumTaskCoordinator,
    create_task_cid,
)
from noob.runner import QueuedRunner, TaskQueue, TaskPriority


# ============================================================================
# Image Processing Nodes
# ============================================================================

class ImageLoader:
    """Load and preprocess images"""

    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)

    def init(self):
        """Initialize loader"""
        print(f"📂 Loading images from {self.image_dir}")

    def process(self) -> dict:
        """Yield batch of image paths"""
        images = list(self.image_dir.glob("*.jpg"))

        return {
            "image_batch": images[:100],  # Process 100 at a time
            "total_images": len(images)
        }

    def deinit(self):
        """Cleanup"""
        pass


class ImagePreprocessor:
    """Preprocess images for analysis"""

    def init(self):
        pass

    def process(self, image_path: Path) -> dict:
        """Preprocess single image"""
        # Load image
        img = Image.open(image_path)

        # Resize to standard size
        img = img.resize((512, 512))

        # Convert to numpy array
        img_array = np.array(img)

        # Normalize
        img_array = img_array / 255.0

        return {
            "image_id": image_path.stem,
            "image_data": img_array,
            "original_size": Image.open(image_path).size
        }


class FireDetector:
    """Detect fires in satellite images using ML model"""

    def init(self):
        """Load ML model"""
        print("🔥 Loading fire detection model...")
        # In production, load actual ML model (YOLOv8, etc.)
        self.model = self._create_dummy_model()

    def _create_dummy_model(self):
        """Create dummy model for demonstration"""
        # In production: import torch; return torch.load('fire_detector.pt')
        return lambda img: {
            "fire_detected": np.random.random() > 0.95,  # 5% false positive rate
            "confidence": np.random.random(),
            "bounding_boxes": []
        }

    def process(self, image_data: np.ndarray, image_id: str) -> dict:
        """Run fire detection on image"""
        # Run ML inference
        result = self.model(image_data)

        return {
            "image_id": image_id,
            "fire_detected": result["fire_detected"],
            "confidence": result["confidence"],
            "bounding_boxes": result["bounding_boxes"],
            "processed_at": time.time()
        }

    def deinit(self):
        """Cleanup model"""
        pass


class ResultAggregator:
    """Aggregate results and prepare for blockchain submission"""

    def __init__(self):
        self.results = []

    def init(self):
        pass

    def process(self, result: dict) -> dict:
        """Accumulate results"""
        self.results.append(result)

        fire_count = sum(1 for r in self.results if r["fire_detected"])

        return {
            "total_processed": len(self.results),
            "fires_detected": fire_count,
            "fire_rate": fire_count / len(self.results) if self.results else 0,
            "results": self.results
        }

    def deinit(self):
        """Final report"""
        print(f"\n📊 Final Statistics:")
        print(f"   Images processed: {len(self.results)}")
        print(f"   Fires detected: {sum(1 for r in self.results if r['fire_detected'])}")


# ============================================================================
# Blockchain Integration
# ============================================================================

class BlockchainTaskRunner:
    """
    Runner that integrates with Ethereum smart contract for payments.

    Each task is submitted on-chain with ETH reward.
    Workers stake ETH to claim tasks.
    Results are verified on-chain after challenge period.
    Workers get paid for successful completion.
    """

    def __init__(
        self,
        tube: Tube,
        blockchain_config: BlockchainConfig,
        reward_per_task_eth: float = 0.001
    ):
        self.tube = tube
        self.reward_per_task = reward_per_task_eth

        # Initialize blockchain coordinator
        self.coordinator = EthereumTaskCoordinator(blockchain_config)

        # Initialize task queue
        self.queue = TaskQueue(persistent=True, db_path="./blockchain_tasks.db")

        # Track on-chain task mappings
        self.task_to_chain: dict[str, bytes] = {}
        self.chain_to_task: dict[bytes, str] = {}

    async def run_with_blockchain(self, n_epochs: int = 10):
        """
        Run pipeline with blockchain integration.

        1. Process tasks locally
        2. Submit results to blockchain
        3. Wait for verification
        4. Collect payments
        """
        print("\n🚀 Starting Blockchain-Integrated Pipeline")
        print("="*80)

        # Register as worker if not already
        worker_info = self.coordinator.get_worker()
        if worker_info["total_stake"] == 0:
            print("💰 Registering as worker and staking 1 ETH...")
            tx_hash = self.coordinator.register_worker(stake_eth=1.0)
            print(f"   ✓ Registered: {tx_hash}")
        else:
            print(f"✓ Already registered with {worker_info['reputation_percent']:.1f}% reputation")

        # Create QueuedRunner
        runner = QueuedRunner(
            self.tube,
            queue=self.queue,
            workers=[],  # Local execution
            local_execution=True
        )

        runner.init()

        try:
            for epoch in range(n_epochs):
                print(f"\n📊 Epoch {epoch + 1}/{n_epochs}")
                print("-"*80)

                # Process epoch
                result = runner.process()

                if result:
                    # Submit to blockchain
                    await self._submit_to_blockchain(result, epoch)

                    # Wait for verification and payment
                    await self._verify_and_collect(epoch)

        finally:
            runner.deinit()
            self.queue.shutdown()

    async def _submit_to_blockchain(self, result: dict, epoch: int):
        """Submit result to blockchain"""
        # Serialize result
        result_data = pickle.dumps(result)

        # Create content identifier
        result_cid = create_task_cid(result_data)

        print(f"   📝 Submitting result to blockchain...")
        print(f"      Result CID: {result_cid.hex()[:16]}...")

        # Submit on-chain
        tx_hash = self.coordinator.submit_result(
            task_cid=result_cid,  # In production, use actual task CID
            result_cid=result_cid
        )

        print(f"      ✓ Submitted: {tx_hash}")

        self.task_to_chain[f"epoch_{epoch}"] = result_cid

    async def _verify_and_collect(self, epoch: int):
        """Verify result and collect payment"""
        task_key = f"epoch_{epoch}"

        if task_key not in self.task_to_chain:
            return

        result_cid = self.task_to_chain[task_key]

        # In production, wait for challenge period
        print(f"   ⏳ Waiting for challenge period (simulation)...")
        await asyncio.sleep(2)  # Simulate challenge period

        # Verify and collect payment
        print(f"   ✅ Verifying task...")
        try:
            tx_hash = self.coordinator.verify_task(result_cid)
            print(f"      ✓ Verified! Payment released: {tx_hash}")

            # Check balance
            balance = self.coordinator.get_balance()
            print(f"      💰 Current balance: {balance:.4f} ETH")

        except Exception as e:
            print(f"      ⚠️  Verification failed: {e}")


# ============================================================================
# Demo Execution
# ============================================================================

def create_pipeline() -> Tube:
    """Create image processing pipeline"""
    from noob import Node, Edge

    # Create nodes
    loader = Node(
        id="loader",
        processor=ImageLoader(image_dir="/path/to/satellite/images"),
        signals=["image_batch", "total_images"]
    )

    preprocessor = Node(
        id="preprocessor",
        processor=ImagePreprocessor(),
        slots=["image_path"],
        signals=["image_data", "image_id"]
    )

    detector = Node(
        id="detector",
        processor=FireDetector(),
        slots=["image_data", "image_id"],
        signals=["result"]
    )

    aggregator = Node(
        id="aggregator",
        processor=ResultAggregator(),
        slots=["result"],
        signals=["summary"]
    )

    # Connect edges
    edges = [
        Edge(source_node="loader", source_signal="image_batch", target_node="preprocessor", target_slot="image_path"),
        Edge(source_node="preprocessor", source_signal="image_data", target_node="detector", target_slot="image_data"),
        Edge(source_node="preprocessor", source_signal="image_id", target_node="detector", target_slot="image_id"),
        Edge(source_node="detector", source_signal="result", target_node="aggregator", target_slot="result"),
    ]

    # Create tube
    tube = Tube(
        nodes={"loader": loader, "preprocessor": preprocessor, "detector": detector, "aggregator": aggregator},
        edges=edges
    )

    return tube


async def main():
    """Run blockchain-integrated image processing demo"""
    print("\n" + "🔥"*40)
    print("BLOCKCHAIN-POWERED DISTRIBUTED IMAGE PROCESSING")
    print("🔥"*40)

    print("""
    This demo shows:
    ✓ Distributed image processing pipeline
    ✓ On-chain task coordination with ETH payments
    ✓ Worker staking and reputation
    ✓ Cryptoeconomic guarantees
    ✓ Byzantine fault tolerance

    Use Case: Satellite image analysis for forest fire detection
    - Process 10,000 images across 100 workers
    - Each worker earns 0.001 ETH per image
    - Total payout: 10 ETH for successful completion
    - Workers must stake 1 ETH to participate
    - Bad actors are slashed 50% of stake
    """)

    # Create pipeline
    tube = create_pipeline()

    # Configure blockchain (testnet for demo)
    config = BlockchainConfig(
        rpc_url="http://127.0.0.1:8545",  # Local Hardhat/Ganache
        chain_id=1337,
        contract_address="0x5FbDB2315678afecb367f032d93F642f64180aa3",  # Example
        private_key="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"  # Example
    )

    # Create blockchain runner
    runner = BlockchainTaskRunner(
        tube,
        blockchain_config=config,
        reward_per_task_eth=0.001
    )

    # Run with blockchain integration
    await runner.run_with_blockchain(n_epochs=5)

    print("\n✨ Demo Complete!")


if __name__ == "__main__":
    asyncio.run(main())
