# 🚀 Blockchain Deployment Guide

Complete guide to deploying NOOB with Ethereum integration for production.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Smart Contract Deployment](#smart-contract-deployment)
- [Worker Setup](#worker-setup)
- [Coordinator Setup](#coordinator-setup)
- [Production Configuration](#production-configuration)
- [Monitoring](#monitoring)
- [Security](#security)

---

## Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Ethereum Blockchain                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         TaskCoordinator Smart Contract               │   │
│  │  • Task Registry                                     │   │
│  │  • Worker Stakes                                     │   │
│  │  • Reputation Scores                                 │   │
│  │  • Payment Distribution                              │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────┬──────────────┬──────────────┬───────────────┘
               │              │              │
          Web3 │         Web3 │         Web3 │
               ▼              ▼              ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │ Coordinator  │  │   Worker 1   │  │   Worker N   │
   │  • Submit    │  │  • Claim     │  │  • Claim     │
   │  • Verify    │  │  • Process   │  │  • Process   │
   │  • Monitor   │  │  • Submit    │  │  • Submit    │
   └──────────────┘  └──────────────┘  └──────────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                    P2P Network (optional)
                 Content-Addressed Storage
```

---

## Prerequisites

### Required Software

1. **Python 3.10+**
   ```bash
   python --version  # Should be 3.10 or higher
   ```

2. **Node.js 18+** (for smart contract deployment)
   ```bash
   node --version
   npm --version
   ```

3. **Hardhat** (Ethereum development environment)
   ```bash
   npm install --global hardhat
   ```

4. **Foundry** (alternative, faster)
   ```bash
   curl -L https://foundry.paradigm.xyz | bash
   foundryup
   ```

### Python Dependencies

```bash
pip install noob[blockchain]

# Or install individually:
pip install web3 eth-account eth-utils
pip install fastapi uvicorn  # For worker server
pip install pytest pytest-asyncio  # For testing
```

### Blockchain Access

Choose your network:

1. **Local Development** (Hardhat/Ganache)
2. **Testnet** (Goerli, Sepolia, Mumbai)
3. **Mainnet** (Ethereum, Polygon, Arbitrum, Base)

**Get RPC URL** from:
- [Alchemy](https://www.alchemy.com/)
- [Infura](https://infura.io/)
- [QuickNode](https://www.quicknode.com/)

---

## Smart Contract Deployment

### Step 1: Setup Hardhat Project

```bash
cd contracts
npm init -y
npm install --save-dev hardhat @openzeppelin/contracts
npx hardhat init
```

### Step 2: Configure Hardhat

Edit `hardhat.config.js`:

```javascript
require("@nomiclabs/hardhat-ethers");

const PRIVATE_KEY = process.env.PRIVATE_KEY;
const ALCHEMY_API_KEY = process.env.ALCHEMY_API_KEY;

module.exports = {
  solidity: "0.8.20",
  networks: {
    // Local development
    localhost: {
      url: "http://127.0.0.1:8545"
    },

    // Ethereum Goerli Testnet
    goerli: {
      url: `https://eth-goerli.g.alchemy.com/v2/${ALCHEMY_API_KEY}`,
      accounts: [PRIVATE_KEY],
      chainId: 5
    },

    // Polygon Mumbai Testnet
    mumbai: {
      url: `https://polygon-mumbai.g.alchemy.com/v2/${ALCHEMY_API_KEY}`,
      accounts: [PRIVATE_KEY],
      chainId: 80001
    },

    // Ethereum Mainnet
    mainnet: {
      url: `https://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}`,
      accounts: [PRIVATE_KEY],
      chainId: 1
    },

    // Polygon Mainnet
    polygon: {
      url: `https://polygon-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}`,
      accounts: [PRIVATE_KEY],
      chainId: 137
    },

    // Arbitrum One
    arbitrum: {
      url: `https://arb-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}`,
      accounts: [PRIVATE_KEY],
      chainId: 42161
    },

    // Base
    base: {
      url: `https://base-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}`,
      accounts: [PRIVATE_KEY],
      chainId: 8453
    }
  }
};
```

### Step 3: Deploy Contract

Create `scripts/deploy.js`:

```javascript
async function main() {
  const [deployer] = await ethers.getSigners();

  console.log("Deploying contracts with account:", deployer.address);
  console.log("Account balance:", (await deployer.getBalance()).toString());

  const TaskCoordinator = await ethers.getContractFactory("TaskCoordinator");
  const coordinator = await TaskCoordinator.deploy();

  await coordinator.deployed();

  console.log("TaskCoordinator deployed to:", coordinator.address);

  // Save deployment info
  const fs = require('fs');
  const deployment = {
    address: coordinator.address,
    network: network.name,
    deployer: deployer.address,
    timestamp: new Date().toISOString()
  };

  fs.writeFileSync(
    `deployment-${network.name}.json`,
    JSON.stringify(deployment, null, 2)
  );
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
```

**Deploy to testnet:**

```bash
# Set environment variables
export PRIVATE_KEY="0xyour_private_key"
export ALCHEMY_API_KEY="your_alchemy_key"

# Deploy to Mumbai (Polygon testnet)
npx hardhat run scripts/deploy.js --network mumbai

# Output:
# TaskCoordinator deployed to: 0xYourContractAddress
```

### Step 4: Verify Contract (Etherscan/Polygonscan)

```bash
npx hardhat verify --network mumbai 0xYourContractAddress
```

---

## Worker Setup

### Configuration

Create `worker_config.yaml`:

```yaml
# Worker Configuration
worker_id: "worker-001"
blockchain:
  network: "mumbai"
  rpc_url: "https://polygon-mumbai.g.alchemy.com/v2/YOUR_KEY"
  chain_id: 80001
  contract_address: "0xYourContractAddress"
  private_key: "0xYourWorkerPrivateKey"

# Stake settings
initial_stake_eth: 1.0
minimum_stake_eth: 0.5

# Performance
max_concurrent_tasks: 10
task_timeout_seconds: 300

# Tube specification
tube_path: "./pipeline.yaml"
```

### Startup Script

Create `start_worker.sh`:

```bash
#!/bin/bash

# Load configuration
source .env

# Register worker if not already registered
python -c "
from noob.blockchain.ethereum import EthereumTaskCoordinator, BlockchainConfig
import yaml

with open('worker_config.yaml') as f:
    config = yaml.safe_load(f)

bc_config = BlockchainConfig(
    rpc_url=config['blockchain']['rpc_url'],
    chain_id=config['blockchain']['chain_id'],
    contract_address=config['blockchain']['contract_address'],
    private_key=config['blockchain']['private_key']
)

coordinator = EthereumTaskCoordinator(bc_config)

# Check if registered
worker_info = coordinator.get_worker()
if worker_info['total_stake'] == 0:
    print('Registering worker with stake...')
    coordinator.register_worker(stake_eth=config['initial_stake_eth'])
    print('Worker registered!')
else:
    print('Worker already registered')
    print(f'Reputation: {worker_info["reputation_percent"]:.1f}%')
"

# Start worker server
python -m noob.runner.worker_server \
    --host 0.0.0.0 \
    --port 8000 \
    --tube pipeline.yaml \
    --max-tasks 10 \
    --worker-id worker-001

```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Start worker
CMD ["python", "-m", "noob.runner.worker_server", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--tube", "/app/pipeline.yaml"]
```

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  worker:
    build: .
    ports:
      - "8000:8000"
    environment:
      - BLOCKCHAIN_RPC_URL=${BLOCKCHAIN_RPC_URL}
      - PRIVATE_KEY=${PRIVATE_KEY}
      - CONTRACT_ADDRESS=${CONTRACT_ADDRESS}
    volumes:
      - ./pipeline.yaml:/app/pipeline.yaml
      - ./data:/app/data
    restart: unless-stopped
```

**Deploy:**

```bash
docker-compose up -d
docker-compose logs -f worker
```

---

## Coordinator Setup

### Task Submission Script

Create `submit_tasks.py`:

```python
#!/usr/bin/env python
"""
Submit tasks to blockchain for distributed processing.
"""

import asyncio
import pickle
from pathlib import Path

from noob import Tube
from noob.blockchain.ethereum import (
    EthereumTaskCoordinator,
    BlockchainConfig,
    create_task_cid,
)

async def submit_batch_tasks(
    coordinator: EthereumTaskCoordinator,
    tube: Tube,
    n_tasks: int = 100,
    reward_per_task_eth: float = 0.001
):
    """Submit batch of tasks to blockchain"""
    print(f"Submitting {n_tasks} tasks to blockchain...")

    for i in range(n_tasks):
        # Create task data
        task_data = pickle.dumps({
            "tube_spec": tube.to_dict(),
            "epoch": i,
            "task_id": i
        })

        # Create content identifier
        task_cid = create_task_cid(task_data)

        # Submit to blockchain
        tx_hash = coordinator.submit_task(
            task_cid=task_cid,
            reward_eth=reward_per_task_eth,
            deadline_seconds=3600,  # 1 hour
            required_stake_eth=0.1
        )

        print(f"Task {i}: {tx_hash}")

        # Rate limit to avoid overwhelming RPC
        await asyncio.sleep(0.5)

    print(f"✓ All {n_tasks} tasks submitted!")


async def main():
    # Load tube
    tube = Tube.from_specification("pipeline.yaml")

    # Configure blockchain
    config = BlockchainConfig(
        rpc_url="https://polygon-mumbai.g.alchemy.com/v2/YOUR_KEY",
        chain_id=80001,
        contract_address="0xYourContractAddress",
        private_key="0xYourPrivateKey"
    )

    # Create coordinator
    coordinator = EthereumTaskCoordinator(config)

    # Submit tasks
    await submit_batch_tasks(
        coordinator,
        tube,
        n_tasks=100,
        reward_per_task_eth=0.001
    )

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Production Configuration

### Environment Variables

Create `.env`:

```bash
# Blockchain
BLOCKCHAIN_NETWORK=polygon
BLOCKCHAIN_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY
CONTRACT_ADDRESS=0xYourContractAddress
CHAIN_ID=137

# Coordinator
COORDINATOR_PRIVATE_KEY=0xYourKey

# Workers
WORKER_PRIVATE_KEYS=0xWorker1Key,0xWorker2Key,0xWorker3Key

# Payments
REWARD_PER_TASK_ETH=0.001
TOTAL_REWARD_POOL_ETH=100.0

# Performance
MAX_CONCURRENT_TASKS=100
TASK_TIMEOUT_SECONDS=300

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
```

### Systemd Service (Linux)

Create `/etc/systemd/system/noob-worker.service`:

```ini
[Unit]
Description=NOOB Blockchain Worker
After=network.target

[Service]
Type=simple
User=noob
WorkingDirectory=/opt/noob
EnvironmentFile=/opt/noob/.env
ExecStart=/usr/bin/python3 -m noob.runner.worker_server \
    --host 0.0.0.0 \
    --port 8000 \
    --tube /opt/noob/pipeline.yaml \
    --max-tasks 10
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**

```bash
sudo systemctl enable noob-worker
sudo systemctl start noob-worker
sudo systemctl status noob-worker
```

---

## Monitoring

### Prometheus Metrics

Add metrics endpoint to worker server:

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
tasks_processed = Counter('noob_tasks_processed_total', 'Total tasks processed')
tasks_failed = Counter('noob_tasks_failed_total', 'Total tasks failed')
task_duration = Histogram('noob_task_duration_seconds', 'Task duration')
reputation_score = Gauge('noob_worker_reputation', 'Worker reputation score')
stake_balance = Gauge('noob_worker_stake_eth', 'Worker stake balance')

# Start metrics server
start_http_server(9090)
```

### Grafana Dashboard

Import dashboard JSON:

```json
{
  "dashboard": {
    "title": "NOOB Blockchain Workers",
    "panels": [
      {
        "title": "Tasks Processed",
        "targets": [
          {"expr": "rate(noob_tasks_processed_total[5m])"}
        ]
      },
      {
        "title": "Worker Reputation",
        "targets": [
          {"expr": "noob_worker_reputation"}
        ]
      },
      {
        "title": "Stake Balance",
        "targets": [
          {"expr": "noob_worker_stake_eth"}
        ]
      }
    ]
  }
}
```

### Alerting

Create `alerts.yml`:

```yaml
groups:
  - name: noob_alerts
    rules:
      - alert: WorkerReputationLow
        expr: noob_worker_reputation < 30
        for: 10m
        annotations:
          summary: "Worker reputation below 30%"

      - alert: HighTaskFailureRate
        expr: rate(noob_tasks_failed_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "Task failure rate > 10%"

      - alert: StakeTooLow
        expr: noob_worker_stake_eth < 0.5
        for: 5m
        annotations:
          summary: "Worker stake below minimum"
```

---

## Security

### Best Practices

1. **Private Keys**
   - Never commit to git
   - Use hardware wallet for mainnet
   - Rotate regularly
   - Use key management service (AWS KMS, HashiCorp Vault)

2. **Smart Contract**
   - Audit before mainnet deployment
   - Use OpenZeppelin contracts
   - Enable ReentrancyGuard
   - Set reasonable gas limits

3. **Worker Security**
   - Run in isolated environment
   - Limit network access
   - Enable firewall
   - Use HTTPS for RPC

4. **Monitoring**
   - Alert on suspicious activity
   - Monitor gas prices
   - Track reputation changes
   - Log all transactions

### Incident Response

**If worker compromised:**

1. Stop worker immediately
2. Withdraw remaining stake
3. Rotate private keys
4. Review logs for unauthorized transactions
5. Report to protocol administrators

**If contract exploited:**

1. Pause contract (if pause function exists)
2. Notify all users
3. Work with security researchers
4. Prepare fix and migration plan

---

## Cost Estimation

### Gas Costs (Polygon)

| Operation | Gas | Cost @ 50 gwei |
|-----------|-----|----------------|
| Register Worker | 150,000 | $0.02 |
| Submit Task | 100,000 | $0.01 |
| Claim Task | 80,000 | $0.008 |
| Submit Result | 90,000 | $0.009 |
| Verify Task | 120,000 | $0.012 |

**Total per task: ~$0.05 on Polygon**

### Monthly Costs (10,000 tasks)

| Item | Cost |
|------|------|
| Gas fees | $500 |
| RPC (Alchemy) | $50 |
| Worker VPS (4x) | $80 |
| Monitoring | $20 |
| **Total** | **$650** |

---

## Troubleshooting

### Common Issues

**1. Transaction fails with "insufficient funds"**
```bash
# Check balance
python -c "from noob.blockchain.ethereum import *; \
           c = EthereumTaskCoordinator(config); \
           print(c.get_balance())"
```

**2. Worker can't claim tasks**
```python
# Check stake
worker_info = coordinator.get_worker()
print(f"Available stake: {worker_info['available_stake']} wei")
```

**3. Task verification fails**
```python
# Check challenge period
task_info = coordinator.get_task(task_cid)
if task_info['status'] == 4:  # Completed
    # Wait for challenge period
    print("Still in challenge period")
```

---

## Support

- **Documentation**: See `BLOCKCHAIN_EXAMPLES.md` for more examples
- **Issues**: https://github.com/miniscope/noob/issues
- **Discord**: [Community server]
- **Email**: support@noob.ai

---

**Deploy with confidence! 🚀⚡🔗**
