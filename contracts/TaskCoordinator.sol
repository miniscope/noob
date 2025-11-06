// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title TaskCoordinator
 * @dev Decentralized task coordination with cryptoeconomic guarantees
 *
 * Features:
 * - On-chain task registry with content-addressed verification
 * - Worker staking and slashing for Byzantine behavior
 * - Reputation system based on successful completions
 * - Automatic payment distribution
 * - Challenge mechanism for disputed results
 * - zkSNARK verification support (future)
 */

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract TaskCoordinator is Ownable, ReentrancyGuard {

    // Task structure
    struct Task {
        bytes32 taskCID;           // Content identifier (IPFS/IPLD)
        address submitter;         // Who submitted the task
        uint256 reward;            // Payment for completion
        uint256 stake;             // Required worker stake
        uint256 deadline;          // Task deadline timestamp
        TaskStatus status;         // Current status
        address worker;            // Claimed by worker
        bytes32 resultCID;         // Result content identifier
        uint256 completedAt;       // Completion timestamp
        uint256 challengeDeadline; // Challenge period end
        bool verified;             // Result verified
    }

    enum TaskStatus {
        Pending,        // Task available for claiming
        Claimed,        // Worker claimed but not started
        InProgress,     // Worker processing
        Completed,      // Result submitted, in challenge period
        Verified,       // Result verified, payment released
        Challenged,     // Result disputed
        Failed,         // Task failed (timeout or slashed)
        Cancelled       // Cancelled by submitter
    }

    // Worker reputation and stake
    struct Worker {
        uint256 totalStake;        // Total ETH staked
        uint256 lockedStake;       // Stake locked in active tasks
        uint256 tasksCompleted;    // Successfully completed tasks
        uint256 tasksFailed;       // Failed tasks (timeout/slash)
        uint256 reputationScore;   // Reputation (0-10000, basis points)
        bool registered;           // Is registered
        uint256 lastActiveTime;    // Last activity timestamp
    }

    // Task registry
    mapping(bytes32 => Task) public tasks;
    mapping(address => Worker) public workers;

    // Token for payments (if using ERC20, otherwise use ETH)
    IERC20 public paymentToken;
    bool public useNativeETH = true;

    // Protocol parameters
    uint256 public minStake = 0.1 ether;
    uint256 public challengePeriod = 1 hours;
    uint256 public slashPercentage = 50; // 50% of stake slashed for Byzantine behavior
    uint256 public protocolFee = 2; // 2% protocol fee

    // Events
    event TaskSubmitted(bytes32 indexed taskCID, address indexed submitter, uint256 reward);
    event TaskClaimed(bytes32 indexed taskCID, address indexed worker);
    event TaskStarted(bytes32 indexed taskCID, address indexed worker);
    event TaskCompleted(bytes32 indexed taskCID, bytes32 resultCID, address indexed worker);
    event TaskVerified(bytes32 indexed taskCID, address indexed worker);
    event TaskChallenged(bytes32 indexed taskCID, address indexed challenger);
    event TaskSlashed(bytes32 indexed taskCID, address indexed worker, uint256 slashAmount);
    event WorkerRegistered(address indexed worker, uint256 stake);
    event WorkerStakeAdded(address indexed worker, uint256 amount);
    event WorkerStakeWithdrawn(address indexed worker, uint256 amount);
    event ReputationUpdated(address indexed worker, uint256 newScore);

    constructor() {
        // Initialize with native ETH payments
        useNativeETH = true;
    }

    /**
     * @dev Set ERC20 token for payments (optional)
     */
    function setPaymentToken(address _tokenAddress) external onlyOwner {
        paymentToken = IERC20(_tokenAddress);
        useNativeETH = false;
    }

    /**
     * @dev Register as a worker and stake ETH
     */
    function registerWorker() external payable {
        require(!workers[msg.sender].registered, "Already registered");
        require(msg.value >= minStake, "Insufficient stake");

        workers[msg.sender] = Worker({
            totalStake: msg.value,
            lockedStake: 0,
            tasksCompleted: 0,
            tasksFailed: 0,
            reputationScore: 5000, // Start at 50% reputation
            registered: true,
            lastActiveTime: block.timestamp
        });

        emit WorkerRegistered(msg.sender, msg.value);
    }

    /**
     * @dev Add more stake
     */
    function addStake() external payable {
        require(workers[msg.sender].registered, "Not registered");
        require(msg.value > 0, "Must stake something");

        workers[msg.sender].totalStake += msg.value;

        emit WorkerStakeAdded(msg.sender, msg.value);
    }

    /**
     * @dev Withdraw unlocked stake
     */
    function withdrawStake(uint256 amount) external nonReentrant {
        Worker storage worker = workers[msg.sender];
        require(worker.registered, "Not registered");

        uint256 availableStake = worker.totalStake - worker.lockedStake;
        require(amount <= availableStake, "Insufficient unlocked stake");
        require(availableStake - amount >= minStake, "Must maintain minimum stake");

        worker.totalStake -= amount;

        payable(msg.sender).transfer(amount);

        emit WorkerStakeWithdrawn(msg.sender, amount);
    }

    /**
     * @dev Submit a task to the network
     */
    function submitTask(
        bytes32 _taskCID,
        uint256 _deadline,
        uint256 _requiredStake
    ) external payable {
        require(tasks[_taskCID].submitter == address(0), "Task already exists");
        require(_deadline > block.timestamp, "Deadline must be in future");
        require(msg.value > 0, "Must provide reward");

        tasks[_taskCID] = Task({
            taskCID: _taskCID,
            submitter: msg.sender,
            reward: msg.value,
            stake: _requiredStake,
            deadline: _deadline,
            status: TaskStatus.Pending,
            worker: address(0),
            resultCID: bytes32(0),
            completedAt: 0,
            challengeDeadline: 0,
            verified: false
        });

        emit TaskSubmitted(_taskCID, msg.sender, msg.value);
    }

    /**
     * @dev Claim a task (worker)
     */
    function claimTask(bytes32 _taskCID) external {
        Task storage task = tasks[_taskCID];
        Worker storage worker = workers[msg.sender];

        require(task.status == TaskStatus.Pending, "Task not available");
        require(worker.registered, "Not registered as worker");
        require(block.timestamp < task.deadline, "Task deadline passed");

        uint256 availableStake = worker.totalStake - worker.lockedStake;
        require(availableStake >= task.stake, "Insufficient stake");

        // Lock stake
        worker.lockedStake += task.stake;

        // Update task
        task.status = TaskStatus.Claimed;
        task.worker = msg.sender;

        emit TaskClaimed(_taskCID, msg.sender);
    }

    /**
     * @dev Start processing task (worker reports start)
     */
    function startTask(bytes32 _taskCID) external {
        Task storage task = tasks[_taskCID];

        require(task.worker == msg.sender, "Not task worker");
        require(task.status == TaskStatus.Claimed, "Task not claimed");
        require(block.timestamp < task.deadline, "Task deadline passed");

        task.status = TaskStatus.InProgress;
        workers[msg.sender].lastActiveTime = block.timestamp;

        emit TaskStarted(_taskCID, msg.sender);
    }

    /**
     * @dev Submit task result (worker)
     */
    function submitResult(bytes32 _taskCID, bytes32 _resultCID) external {
        Task storage task = tasks[_taskCID];

        require(task.worker == msg.sender, "Not task worker");
        require(
            task.status == TaskStatus.Claimed || task.status == TaskStatus.InProgress,
            "Invalid task status"
        );
        require(block.timestamp < task.deadline, "Task deadline passed");

        task.status = TaskStatus.Completed;
        task.resultCID = _resultCID;
        task.completedAt = block.timestamp;
        task.challengeDeadline = block.timestamp + challengePeriod;

        workers[msg.sender].lastActiveTime = block.timestamp;

        emit TaskCompleted(_taskCID, _resultCID, msg.sender);
    }

    /**
     * @dev Verify task (after challenge period) and release payment
     */
    function verifyTask(bytes32 _taskCID) external nonReentrant {
        Task storage task = tasks[_taskCID];
        Worker storage worker = workers[task.worker];

        require(task.status == TaskStatus.Completed, "Task not completed");
        require(block.timestamp >= task.challengeDeadline, "Challenge period active");

        // Release locked stake
        worker.lockedStake -= task.stake;

        // Update reputation (increase for successful completion)
        _updateReputation(task.worker, true);

        // Calculate payment
        uint256 fee = (task.reward * protocolFee) / 100;
        uint256 workerPayment = task.reward - fee;

        // Transfer payment
        payable(task.worker).transfer(workerPayment);

        // Update task
        task.status = TaskStatus.Verified;
        task.verified = true;

        // Update worker stats
        worker.tasksCompleted++;

        emit TaskVerified(_taskCID, task.worker);
    }

    /**
     * @dev Challenge a task result (if incorrect)
     */
    function challengeTask(bytes32 _taskCID, string calldata reason) external {
        Task storage task = tasks[_taskCID];

        require(task.status == TaskStatus.Completed, "Task not completed");
        require(block.timestamp < task.challengeDeadline, "Challenge period expired");
        require(msg.sender == task.submitter, "Only submitter can challenge");

        task.status = TaskStatus.Challenged;

        emit TaskChallenged(_taskCID, msg.sender);

        // In production, would trigger dispute resolution mechanism
        // For now, automatically slash (in production, use oracle/voting)
        _slashWorker(_taskCID);
    }

    /**
     * @dev Slash worker for Byzantine behavior
     */
    function _slashWorker(bytes32 _taskCID) internal {
        Task storage task = tasks[_taskCID];
        Worker storage worker = workers[task.worker];

        uint256 slashAmount = (task.stake * slashPercentage) / 100;

        // Slash stake
        worker.lockedStake -= task.stake;
        worker.totalStake -= slashAmount;

        // Update reputation (decrease for failure)
        _updateReputation(task.worker, false);

        // Refund submitter
        payable(task.submitter).transfer(task.reward + slashAmount);

        // Update task
        task.status = TaskStatus.Failed;

        // Update worker stats
        worker.tasksFailed++;

        emit TaskSlashed(_taskCID, task.worker, slashAmount);
    }

    /**
     * @dev Update worker reputation
     */
    function _updateReputation(address _worker, bool success) internal {
        Worker storage worker = workers[_worker];

        if (success) {
            // Increase reputation (max 10000)
            uint256 increase = 100; // +1% per success
            if (worker.reputationScore + increase <= 10000) {
                worker.reputationScore += increase;
            } else {
                worker.reputationScore = 10000;
            }
        } else {
            // Decrease reputation (min 0)
            uint256 decrease = 500; // -5% per failure
            if (worker.reputationScore >= decrease) {
                worker.reputationScore -= decrease;
            } else {
                worker.reputationScore = 0;
            }
        }

        emit ReputationUpdated(_worker, worker.reputationScore);
    }

    /**
     * @dev Handle task timeout (anyone can call)
     */
    function handleTimeout(bytes32 _taskCID) external {
        Task storage task = tasks[_taskCID];

        require(
            task.status == TaskStatus.Claimed || task.status == TaskStatus.InProgress,
            "Invalid status for timeout"
        );
        require(block.timestamp >= task.deadline, "Deadline not passed");

        Worker storage worker = workers[task.worker];

        // Unlock stake (no slash for timeout, just reputation hit)
        worker.lockedStake -= task.stake;

        // Reputation penalty
        _updateReputation(task.worker, false);

        // Refund submitter
        payable(task.submitter).transfer(task.reward);

        // Update task
        task.status = TaskStatus.Failed;

        // Update stats
        worker.tasksFailed++;
    }

    /**
     * @dev Get task details
     */
    function getTask(bytes32 _taskCID) external view returns (
        address submitter,
        uint256 reward,
        TaskStatus status,
        address worker,
        bytes32 resultCID,
        uint256 completedAt
    ) {
        Task storage task = tasks[_taskCID];
        return (
            task.submitter,
            task.reward,
            task.status,
            task.worker,
            task.resultCID,
            task.completedAt
        );
    }

    /**
     * @dev Get worker details
     */
    function getWorker(address _worker) external view returns (
        uint256 totalStake,
        uint256 availableStake,
        uint256 tasksCompleted,
        uint256 tasksFailed,
        uint256 reputationScore
    ) {
        Worker storage worker = workers[_worker];
        return (
            worker.totalStake,
            worker.totalStake - worker.lockedStake,
            worker.tasksCompleted,
            worker.tasksFailed,
            worker.reputationScore
        );
    }

    /**
     * @dev Withdraw protocol fees (owner)
     */
    function withdrawFees() external onlyOwner nonReentrant {
        uint256 balance = address(this).balance;
        payable(owner()).transfer(balance);
    }

    /**
     * @dev Update protocol parameters (owner)
     */
    function updateParameters(
        uint256 _minStake,
        uint256 _challengePeriod,
        uint256 _slashPercentage,
        uint256 _protocolFee
    ) external onlyOwner {
        require(_slashPercentage <= 100, "Slash percentage too high");
        require(_protocolFee <= 10, "Protocol fee too high");

        minStake = _minStake;
        challengePeriod = _challengePeriod;
        slashPercentage = _slashPercentage;
        protocolFee = _protocolFee;
    }
}
