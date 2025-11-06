"""
Ethereum Smart Contract Integration

Connects NOOB distributed execution with on-chain verification, payments,
and cryptoeconomic guarantees.

Features:
- Task submission with ETH rewards
- Worker staking and slashing
- Reputation tracking
- Automatic payment distribution
- Challenge mechanism
- Multi-chain support (Ethereum, Polygon, Arbitrum, Optimism, Base)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from web3 import Web3
from web3.contract import Contract
from web3.middleware import geth_poa_middleware
from eth_account import Account
from eth_typing import ChecksumAddress


@dataclass
class BlockchainConfig:
    """Configuration for blockchain connection"""
    rpc_url: str
    chain_id: int
    contract_address: Optional[str] = None
    private_key: Optional[str] = None
    gas_price_gwei: Optional[float] = None


class EthereumTaskCoordinator:
    """
    Ethereum-backed task coordination with cryptoeconomic guarantees.

    Provides on-chain verification, staking, reputation, and payment distribution
    for distributed task execution.

    Example:
        >>> config = BlockchainConfig(
        ...     rpc_url="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY",
        ...     chain_id=1,
        ...     contract_address="0x123...",
        ...     private_key="0xabc..."
        ... )
        >>> coordinator = EthereumTaskCoordinator(config)
        >>>
        >>> # Submit task with 0.1 ETH reward
        >>> task_cid = content_hash(task_data)
        >>> tx_hash = coordinator.submit_task(task_cid, reward_eth=0.1)
        >>>
        >>> # Worker claims task
        >>> coordinator.claim_task(task_cid)
        >>>
        >>> # Submit result
        >>> result_cid = content_hash(result_data)
        >>> coordinator.submit_result(task_cid, result_cid)
        >>>
        >>> # After challenge period, verify and get paid
        >>> coordinator.verify_task(task_cid)
    """

    def __init__(self, config: BlockchainConfig):
        """Initialize Ethereum connection"""
        self.config = config

        # Connect to Web3
        self.w3 = Web3(Web3.HTTPProvider(config.rpc_url))

        # Add PoA middleware for chains like Polygon
        if config.chain_id in [137, 80001]:  # Polygon mainnet/testnet
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)

        # Check connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to {config.rpc_url}")

        # Load account if private key provided
        self.account = None
        if config.private_key:
            self.account = Account.from_key(config.private_key)

        # Load contract
        self.contract: Optional[Contract] = None
        if config.contract_address:
            self.contract = self._load_contract(config.contract_address)

    def _load_contract(self, address: str) -> Contract:
        """Load TaskCoordinator contract"""
        # Load ABI from compiled contract
        abi_path = Path(__file__).parent.parent.parent.parent / "contracts" / "TaskCoordinator.json"

        if abi_path.exists():
            with open(abi_path) as f:
                contract_json = json.load(f)
                abi = contract_json.get("abi", [])
        else:
            # Minimal ABI if full ABI not available
            abi = self._get_minimal_abi()

        checksum_address = Web3.to_checksum_address(address)
        return self.w3.eth.contract(address=checksum_address, abi=abi)

    def _get_minimal_abi(self) -> list:
        """Minimal ABI for TaskCoordinator contract"""
        return [
            {
                "inputs": [],
                "name": "registerWorker",
                "outputs": [],
                "stateMutability": "payable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "bytes32", "name": "_taskCID", "type": "bytes32"},
                    {"internalType": "uint256", "name": "_deadline", "type": "uint256"},
                    {"internalType": "uint256", "name": "_requiredStake", "type": "uint256"}
                ],
                "name": "submitTask",
                "outputs": [],
                "stateMutability": "payable",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "bytes32", "name": "_taskCID", "type": "bytes32"}],
                "name": "claimTask",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "bytes32", "name": "_taskCID", "type": "bytes32"}],
                "name": "startTask",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "bytes32", "name": "_taskCID", "type": "bytes32"},
                    {"internalType": "bytes32", "name": "_resultCID", "type": "bytes32"}
                ],
                "name": "submitResult",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "bytes32", "name": "_taskCID", "type": "bytes32"}],
                "name": "verifyTask",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "bytes32", "name": "_taskCID", "type": "bytes32"}],
                "name": "getTask",
                "outputs": [
                    {"internalType": "address", "name": "submitter", "type": "address"},
                    {"internalType": "uint256", "name": "reward", "type": "uint256"},
                    {"internalType": "uint8", "name": "status", "type": "uint8"},
                    {"internalType": "address", "name": "worker", "type": "address"},
                    {"internalType": "bytes32", "name": "resultCID", "type": "bytes32"},
                    {"internalType": "uint256", "name": "completedAt", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "address", "name": "_worker", "type": "address"}],
                "name": "getWorker",
                "outputs": [
                    {"internalType": "uint256", "name": "totalStake", "type": "uint256"},
                    {"internalType": "uint256", "name": "availableStake", "type": "uint256"},
                    {"internalType": "uint256", "name": "tasksCompleted", "type": "uint256"},
                    {"internalType": "uint256", "name": "tasksFailed", "type": "uint256"},
                    {"internalType": "uint256", "name": "reputationScore", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]

    def _send_transaction(self, tx: dict) -> str:
        """Send transaction and wait for receipt"""
        if not self.account:
            raise ValueError("No private key configured")

        # Build transaction
        tx["from"] = self.account.address
        tx["nonce"] = self.w3.eth.get_transaction_count(self.account.address)
        tx["chainId"] = self.config.chain_id

        # Estimate gas
        tx["gas"] = self.w3.eth.estimate_gas(tx)

        # Set gas price
        if self.config.gas_price_gwei:
            tx["gasPrice"] = self.w3.to_wei(self.config.gas_price_gwei, "gwei")
        else:
            tx["gasPrice"] = self.w3.eth.gas_price

        # Sign and send
        signed = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)

        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        if receipt["status"] != 1:
            raise Exception(f"Transaction failed: {tx_hash.hex()}")

        return tx_hash.hex()

    def register_worker(self, stake_eth: float) -> str:
        """
        Register as a worker and stake ETH.

        Args:
            stake_eth: Amount of ETH to stake

        Returns:
            Transaction hash
        """
        stake_wei = self.w3.to_wei(stake_eth, "ether")

        tx = self.contract.functions.registerWorker().build_transaction({
            "value": stake_wei
        })

        return self._send_transaction(tx)

    def submit_task(
        self,
        task_cid: bytes,
        reward_eth: float,
        deadline_seconds: int = 3600,
        required_stake_eth: float = 0.1
    ) -> str:
        """
        Submit a task to the blockchain with ETH reward.

        Args:
            task_cid: Content identifier (32 bytes)
            reward_eth: Reward in ETH
            deadline_seconds: Deadline in seconds from now
            required_stake_eth: Required worker stake in ETH

        Returns:
            Transaction hash
        """
        reward_wei = self.w3.to_wei(reward_eth, "ether")
        required_stake_wei = self.w3.to_wei(required_stake_eth, "ether")
        deadline = int(time.time()) + deadline_seconds

        tx = self.contract.functions.submitTask(
            task_cid,
            deadline,
            required_stake_wei
        ).build_transaction({
            "value": reward_wei
        })

        return self._send_transaction(tx)

    def claim_task(self, task_cid: bytes) -> str:
        """
        Claim a task as a worker.

        Args:
            task_cid: Content identifier

        Returns:
            Transaction hash
        """
        tx = self.contract.functions.claimTask(task_cid).build_transaction({})
        return self._send_transaction(tx)

    def start_task(self, task_cid: bytes) -> str:
        """
        Mark task as started.

        Args:
            task_cid: Content identifier

        Returns:
            Transaction hash
        """
        tx = self.contract.functions.startTask(task_cid).build_transaction({})
        return self._send_transaction(tx)

    def submit_result(self, task_cid: bytes, result_cid: bytes) -> str:
        """
        Submit task result.

        Args:
            task_cid: Task content identifier
            result_cid: Result content identifier

        Returns:
            Transaction hash
        """
        tx = self.contract.functions.submitResult(
            task_cid,
            result_cid
        ).build_transaction({})

        return self._send_transaction(tx)

    def verify_task(self, task_cid: bytes) -> str:
        """
        Verify task and release payment (after challenge period).

        Args:
            task_cid: Content identifier

        Returns:
            Transaction hash
        """
        tx = self.contract.functions.verifyTask(task_cid).build_transaction({})
        return self._send_transaction(tx)

    def get_task(self, task_cid: bytes) -> dict:
        """
        Get task details.

        Args:
            task_cid: Content identifier

        Returns:
            Task details dictionary
        """
        result = self.contract.functions.getTask(task_cid).call()

        return {
            "submitter": result[0],
            "reward": result[1],
            "status": result[2],
            "worker": result[3],
            "result_cid": result[4],
            "completed_at": result[5]
        }

    def get_worker(self, address: Optional[str] = None) -> dict:
        """
        Get worker details.

        Args:
            address: Worker address (defaults to current account)

        Returns:
            Worker details dictionary
        """
        if not address:
            if not self.account:
                raise ValueError("No address specified and no account configured")
            address = self.account.address

        checksum_address = Web3.to_checksum_address(address)
        result = self.contract.functions.getWorker(checksum_address).call()

        return {
            "total_stake": result[0],
            "available_stake": result[1],
            "tasks_completed": result[2],
            "tasks_failed": result[3],
            "reputation_score": result[4],
            "reputation_percent": result[4] / 100  # Convert from basis points
        }

    def get_balance(self, address: Optional[str] = None) -> float:
        """
        Get ETH balance.

        Args:
            address: Address to check (defaults to current account)

        Returns:
            Balance in ETH
        """
        if not address:
            if not self.account:
                raise ValueError("No address specified and no account configured")
            address = self.account.address

        checksum_address = Web3.to_checksum_address(address)
        balance_wei = self.w3.eth.get_balance(checksum_address)
        return self.w3.from_wei(balance_wei, "ether")


def create_task_cid(task_data: bytes) -> bytes:
    """
    Create content identifier for task data.

    Uses Keccak256 (Ethereum's hash function) for compatibility.

    Args:
        task_data: Task data bytes

    Returns:
        32-byte content identifier
    """
    from eth_utils import keccak
    return keccak(task_data)


# Network configurations
NETWORKS = {
    "ethereum": BlockchainConfig(
        rpc_url="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY",
        chain_id=1
    ),
    "goerli": BlockchainConfig(
        rpc_url="https://eth-goerli.g.alchemy.com/v2/YOUR_KEY",
        chain_id=5
    ),
    "sepolia": BlockchainConfig(
        rpc_url="https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY",
        chain_id=11155111
    ),
    "polygon": BlockchainConfig(
        rpc_url="https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY",
        chain_id=137
    ),
    "polygon_mumbai": BlockchainConfig(
        rpc_url="https://polygon-mumbai.g.alchemy.com/v2/YOUR_KEY",
        chain_id=80001
    ),
    "arbitrum": BlockchainConfig(
        rpc_url="https://arb-mainnet.g.alchemy.com/v2/YOUR_KEY",
        chain_id=42161
    ),
    "optimism": BlockchainConfig(
        rpc_url="https://opt-mainnet.g.alchemy.com/v2/YOUR_KEY",
        chain_id=10
    ),
    "base": BlockchainConfig(
        rpc_url="https://base-mainnet.g.alchemy.com/v2/YOUR_KEY",
        chain_id=8453
    ),
    "localhost": BlockchainConfig(
        rpc_url="http://127.0.0.1:8545",
        chain_id=1337
    )
}
