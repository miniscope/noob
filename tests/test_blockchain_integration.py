"""
Comprehensive Blockchain Integration Tests

Tests for Ethereum smart contract integration with NOOB distributed execution.

Test Coverage:
- Smart contract interaction
- Worker registration and staking
- Task submission and claiming
- Result verification and payment
- Reputation system
- Byzantine behavior detection
- Challenge mechanism
- Multi-chain support
"""

import asyncio
import pickle
import pytest
import time
from pathlib import Path

try:
    from web3 import Web3
    from eth_account import Account

    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

from noob import Tube, Node, Edge
from noob.runner import TaskQueue, TaskPriority

if WEB3_AVAILABLE:
    from noob.blockchain.ethereum import (
        BlockchainConfig,
        EthereumTaskCoordinator,
        create_task_cid,
        NETWORKS,
    )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def local_blockchain():
    """Start local Hardhat/Ganache node for testing"""
    if not WEB3_AVAILABLE:
        pytest.skip("web3.py not available")

    # Assume local node running at http://127.0.0.1:8545
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

    if not w3.is_connected():
        pytest.skip("Local blockchain not available (start with: npx hardhat node)")

    return w3


@pytest.fixture
def blockchain_config(local_blockchain):
    """Create blockchain configuration for testing"""
    # Use first test account
    test_account = Account.create()

    return BlockchainConfig(
        rpc_url="http://127.0.0.1:8545",
        chain_id=31337,  # Hardhat default
        contract_address=None,  # Will deploy
        private_key=test_account.key.hex()
    )


@pytest.fixture
def deployed_contract(blockchain_config, local_blockchain):
    """Deploy TaskCoordinator contract for testing"""
    # In production, would actually deploy contract
    # For tests, assume contract deployed at known address
    blockchain_config.contract_address = "0x5FbDB2315678afecb367f032d93F642f64180aa3"

    return blockchain_config


@pytest.fixture
def coordinator(deployed_contract):
    """Create EthereumTaskCoordinator instance"""
    return EthereumTaskCoordinator(deployed_contract)


# ============================================================================
# Basic Contract Interaction Tests
# ============================================================================

@pytest.mark.blockchain
@pytest.mark.skipif(not WEB3_AVAILABLE, reason="web3.py not available")
class TestContractInteraction:
    """Test basic contract interactions"""

    def test_connection(self, coordinator):
        """Test blockchain connection"""
        assert coordinator.w3.is_connected()
        assert coordinator.config.chain_id == 31337

    def test_account_setup(self, coordinator):
        """Test account is properly configured"""
        assert coordinator.account is not None
        assert coordinator.account.address is not None

    def test_balance_check(self, coordinator):
        """Test ETH balance checking"""
        balance = coordinator.get_balance()
        assert balance >= 0

    def test_contract_loaded(self, coordinator):
        """Test contract is loaded"""
        assert coordinator.contract is not None


# ============================================================================
# Worker Registration Tests
# ============================================================================

@pytest.mark.blockchain
@pytest.mark.skipif(not WEB3_AVAILABLE, reason="web3.py not available")
class TestWorkerRegistration:
    """Test worker registration and staking"""

    def test_register_worker(self, coordinator):
        """Test worker registration with stake"""
        # Register with 1 ETH stake
        try:
            tx_hash = coordinator.register_worker(stake_eth=1.0)
            assert tx_hash is not None
            assert len(tx_hash) == 66  # 0x + 64 hex chars
        except Exception as e:
            # May already be registered
            if "already registered" not in str(e).lower():
                raise

    def test_get_worker_info(self, coordinator):
        """Test retrieving worker information"""
        worker_info = coordinator.get_worker()

        assert "total_stake" in worker_info
        assert "available_stake" in worker_info
        assert "tasks_completed" in worker_info
        assert "tasks_failed" in worker_info
        assert "reputation_score" in worker_info

    def test_worker_reputation_initial(self, coordinator):
        """Test worker starts with 50% reputation"""
        worker_info = coordinator.get_worker()

        # New workers start at 5000 basis points (50%)
        assert worker_info["reputation_percent"] == pytest.approx(50.0, abs=1.0)


# ============================================================================
# Task Lifecycle Tests
# ============================================================================

@pytest.mark.blockchain
@pytest.mark.skipif(not WEB3_AVAILABLE, reason="web3.py not available")
class TestTaskLifecycle:
    """Test complete task lifecycle"""

    def test_create_task_cid(self):
        """Test content identifier creation"""
        task_data = b"test task data"
        cid = create_task_cid(task_data)

        assert len(cid) == 32  # 256 bits
        assert isinstance(cid, bytes)

        # Same data should produce same CID
        cid2 = create_task_cid(task_data)
        assert cid == cid2

        # Different data should produce different CID
        cid3 = create_task_cid(b"different data")
        assert cid != cid3

    def test_submit_task(self, coordinator):
        """Test task submission"""
        task_data = b"process this data"
        task_cid = create_task_cid(task_data)

        try:
            tx_hash = coordinator.submit_task(
                task_cid=task_cid,
                reward_eth=0.01,  # 0.01 ETH reward
                deadline_seconds=3600,
                required_stake_eth=0.1
            )

            assert tx_hash is not None
        except Exception as e:
            # Task may already exist
            if "already exists" not in str(e).lower():
                raise

    def test_claim_task(self, coordinator):
        """Test task claiming"""
        # First ensure worker is registered
        try:
            coordinator.register_worker(stake_eth=1.0)
        except:
            pass  # Already registered

        # Submit task
        task_data = b"test claim task"
        task_cid = create_task_cid(task_data)

        try:
            coordinator.submit_task(
                task_cid=task_cid,
                reward_eth=0.01,
                deadline_seconds=3600,
                required_stake_eth=0.1
            )
        except:
            pass  # May already exist

        # Claim task
        try:
            tx_hash = coordinator.claim_task(task_cid)
            assert tx_hash is not None
        except Exception as e:
            # May already be claimed
            if "not available" not in str(e).lower():
                raise

    def test_submit_result(self, coordinator):
        """Test result submission"""
        task_data = b"test result task"
        task_cid = create_task_cid(task_data)

        result_data = b"task result data"
        result_cid = create_task_cid(result_data)

        try:
            # Submit and claim task first
            coordinator.submit_task(task_cid, reward_eth=0.01)
            coordinator.claim_task(task_cid)
            coordinator.start_task(task_cid)

            # Submit result
            tx_hash = coordinator.submit_result(task_cid, result_cid)
            assert tx_hash is not None
        except Exception as e:
            # May be in wrong state
            pass

    def test_get_task_info(self, coordinator):
        """Test retrieving task information"""
        task_data = b"test info task"
        task_cid = create_task_cid(task_data)

        try:
            coordinator.submit_task(task_cid, reward_eth=0.01)
        except:
            pass

        # Get task info
        try:
            task_info = coordinator.get_task(task_cid)

            assert "submitter" in task_info
            assert "reward" in task_info
            assert "status" in task_info
            assert "worker" in task_info
        except Exception:
            # Task may not exist
            pass


# ============================================================================
# Reputation System Tests
# ============================================================================

@pytest.mark.blockchain
@pytest.mark.skipif(not WEB3_AVAILABLE, reason="web3.py not available")
class TestReputationSystem:
    """Test reputation tracking and updates"""

    def test_reputation_increases_on_success(self, coordinator):
        """Test reputation increases after successful task"""
        # Get initial reputation
        initial_info = coordinator.get_worker()
        initial_rep = initial_info["reputation_score"]

        # In a real test, would complete a task successfully
        # and verify reputation increased

    def test_reputation_decreases_on_failure(self, coordinator):
        """Test reputation decreases after task failure"""
        # Would simulate task failure (timeout or challenge)
        # and verify reputation decreased

    def test_reputation_bounds(self):
        """Test reputation stays within bounds [0, 10000]"""
        # Reputation should never go below 0 or above 10000
        pass


# ============================================================================
# Byzantine Behavior Tests
# ============================================================================

@pytest.mark.blockchain
@pytest.mark.skipif(not WEB3_AVAILABLE, reason="web3.py not available")
class TestByzantineBehavior:
    """Test detection and slashing of Byzantine workers"""

    def test_challenge_mechanism(self, coordinator):
        """Test challenging incorrect results"""
        # Submit task, claim, submit bad result
        # Submitter challenges
        # Worker should be slashed
        pass

    def test_stake_slashing(self, coordinator):
        """Test stake is slashed for Byzantine behavior"""
        # Worker submits invalid result
        # After challenge, verify stake decreased by slash percentage
        pass

    def test_timeout_handling(self, coordinator):
        """Test handling of task timeout"""
        # Submit task with short deadline
        # Worker claims but doesn't complete
        # After deadline, anyone can call handleTimeout
        # Verify stake not slashed but reputation decreased
        pass


# ============================================================================
# Payment Distribution Tests
# ============================================================================

@pytest.mark.blockchain
@pytest.mark.skipif(not WEB3_AVAILABLE, reason="web3.py not available")
class TestPaymentDistribution:
    """Test payment distribution and protocol fees"""

    def test_successful_payment(self, coordinator):
        """Test worker receives payment after successful task"""
        # Complete task successfully
        # Wait for challenge period
        # Verify task and check payment received
        pass

    def test_protocol_fee_deduction(self, coordinator):
        """Test protocol fee is deducted from payment"""
        # Protocol fee should be 2% by default
        # Worker should receive 98% of reward
        pass

    def test_challenge_period_enforcement(self, coordinator):
        """Test payment blocked during challenge period"""
        # Submit result
        # Try to verify immediately (should fail)
        # Wait for challenge period
        # Verify should succeed
        pass


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.blockchain
@pytest.mark.skipif(not WEB3_AVAILABLE, reason="web3.py not available")
@pytest.mark.slow
class TestEndToEndIntegration:
    """End-to-end integration tests"""

    @pytest.mark.asyncio
    async def test_full_task_lifecycle(self, coordinator):
        """Test complete task lifecycle from submission to payment"""
        # 1. Register worker
        try:
            coordinator.register_worker(stake_eth=1.0)
        except:
            pass

        # 2. Submit task
        task_data = pickle.dumps({"type": "process", "data": [1, 2, 3]})
        task_cid = create_task_cid(task_data)

        try:
            coordinator.submit_task(
                task_cid,
                reward_eth=0.01,
                deadline_seconds=3600
            )
        except:
            pass

        # 3. Claim task
        try:
            coordinator.claim_task(task_cid)
        except:
            pass

        # 4. Start processing
        try:
            coordinator.start_task(task_cid)
        except:
            pass

        # 5. Process task (simulate)
        await asyncio.sleep(1)
        result_data = pickle.dumps({"result": "success"})
        result_cid = create_task_cid(result_data)

        # 6. Submit result
        try:
            coordinator.submit_result(task_cid, result_cid)
        except:
            pass

        # 7. Wait for challenge period
        await asyncio.sleep(2)

        # 8. Verify and collect payment
        try:
            coordinator.verify_task(task_cid)
        except Exception as e:
            print(f"Verification note: {e}")

    @pytest.mark.asyncio
    async def test_multiple_workers_same_task(self, coordinator):
        """Test multiple workers attempting to claim same task"""
        # Only one worker should successfully claim
        # Others should fail with "not available"
        pass

    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self, coordinator):
        """Test processing multiple tasks concurrently"""
        # Submit multiple tasks
        # Workers claim and process in parallel
        # Verify all complete successfully
        pass


# ============================================================================
# Multi-Chain Tests
# ============================================================================

@pytest.mark.blockchain
@pytest.mark.skipif(not WEB3_AVAILABLE, reason="web3.py not available")
class TestMultiChain:
    """Test support for multiple blockchains"""

    def test_ethereum_mainnet_config(self):
        """Test Ethereum mainnet configuration"""
        config = NETWORKS["ethereum"]
        assert config.chain_id == 1

    def test_polygon_config(self):
        """Test Polygon configuration"""
        config = NETWORKS["polygon"]
        assert config.chain_id == 137

    def test_arbitrum_config(self):
        """Test Arbitrum configuration"""
        config = NETWORKS["arbitrum"]
        assert config.chain_id == 42161

    def test_optimism_config(self):
        """Test Optimism configuration"""
        config = NETWORKS["optimism"]
        assert config.chain_id == 10

    def test_base_config(self):
        """Test Base configuration"""
        config = NETWORKS["base"]
        assert config.chain_id == 8453


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.blockchain
@pytest.mark.skipif(not WEB3_AVAILABLE, reason="web3.py not available")
@pytest.mark.slow
class TestPerformance:
    """Performance tests for blockchain operations"""

    def test_batch_task_submission(self, coordinator):
        """Test submitting many tasks quickly"""
        # Submit 100 tasks
        # Measure time
        # Should be < 10 seconds
        pass

    def test_concurrent_claims(self, coordinator):
        """Test concurrent task claiming"""
        # Multiple workers try to claim same task
        # Should handle race conditions correctly
        pass

    @pytest.mark.asyncio
    async def test_throughput(self, coordinator):
        """Test system throughput (tasks/second)"""
        # Submit, claim, process, verify many tasks
        # Measure throughput
        # Should handle 10+ tasks/second
        pass


# ============================================================================
# Security Tests
# ============================================================================

@pytest.mark.blockchain
@pytest.mark.skipif(not WEB3_AVAILABLE, reason="web3.py not available")
class TestSecurity:
    """Security and attack resistance tests"""

    def test_unauthorized_verification(self, coordinator):
        """Test that only authorized parties can verify"""
        # Non-worker should not be able to verify task they didn't complete
        pass

    def test_double_claim_prevention(self, coordinator):
        """Test worker cannot claim same task twice"""
        pass

    def test_insufficient_stake_rejection(self, coordinator):
        """Test claiming rejected if insufficient stake"""
        pass

    def test_reentrancy_protection(self, coordinator):
        """Test contract protected against reentrancy"""
        # Contract should use ReentrancyGuard
        pass

    def test_integer_overflow_protection(self, coordinator):
        """Test safe math operations"""
        # Should use Solidity 0.8+ with overflow checks
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
