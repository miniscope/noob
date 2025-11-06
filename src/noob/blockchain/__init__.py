"""
NOOB Blockchain Integration

Ethereum smart contract integration for distributed task coordination with
cryptoeconomic guarantees.

Features:
- Worker staking and reputation
- On-chain task verification
- Automatic payment distribution
- Byzantine fault tolerance
- Multi-chain support

Quick Start:
    >>> from noob.blockchain import EthereumTaskCoordinator, BlockchainConfig
    >>>
    >>> config = BlockchainConfig(
    ...     rpc_url="https://polygon-mumbai.g.alchemy.com/v2/YOUR_KEY",
    ...     chain_id=80001,
    ...     contract_address="0xYourContractAddress",
    ...     private_key="0xYourPrivateKey"
    ... )
    >>>
    >>> coordinator = EthereumTaskCoordinator(config)
    >>>
    >>> # Register as worker with stake
    >>> coordinator.register_worker(stake_eth=1.0)
    >>>
    >>> # Submit task with reward
    >>> task_cid = create_task_cid(task_data)
    >>> coordinator.submit_task(task_cid, reward_eth=0.001)
    >>>
    >>> # Claim and process
    >>> coordinator.claim_task(task_cid)
    >>> # ... process task ...
    >>> coordinator.submit_result(task_cid, result_cid)
    >>>
    >>> # Verify and get paid (after challenge period)
    >>> coordinator.verify_task(task_cid)
"""

__all__ = []

try:
    from noob.blockchain.ethereum import (
        BlockchainConfig,
        EthereumTaskCoordinator,
        create_task_cid,
        NETWORKS,
    )

    __all__.extend([
        "BlockchainConfig",
        "EthereumTaskCoordinator",
        "create_task_cid",
        "NETWORKS",
    ])
except ImportError as e:
    # web3.py not installed
    BlockchainConfig = None
    EthereumTaskCoordinator = None
    create_task_cid = None
    NETWORKS = None
