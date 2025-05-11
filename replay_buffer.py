import random
from collections import namedtuple

import numpy as np
import torch


class SumTree:
    """
    A binary tree data structure where the value of a parent node is the sum of its children. This
    version also tracks the minimum and maximum priority values in the subtree rooted at each node.
    Allows for efficient O(log n) sampling and min/max priority-finding.
    """

    write_idx = 0

    def __init__(self, capacity):
        """
        Initializes the SumTree.

        Args:
            capacity (int): The desired maximum number of items (leaves) the tree
                           should represent (e.g., the buffer size).
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")

        # Calculate the next power of 2 about the requested capacity
        self.tree_capacity = 2 ** ((capacity - 1).bit_length())

        self.buffer_capacity = capacity  # Store the original requested capacity

        # Tree size is 2 * tree_capacity - 1
        tree_size = 2 * self.tree_capacity - 1
        self.sum_tree = np.zeros(tree_size)
        self.min_tree = np.full(tree_size, float("inf"))
        self._max_priority = 1.0

        self._size = 0

    def __len__(self):
        return self._size

    @property
    def size(self):
        return self._size

    @property
    def ptr(self):
        return self.write_idx

    @property
    def total(self):
        return self.sum_tree[0]

    @property
    def min_priority(self):
        # The minimum is tracked correctly even with padding,
        # as unused padded leaves remain 'inf'
        return self.min_tree[0] if self.size > 0 else 0.0

    @property
    def max_priority(self):
        # The maximum is tracked correctly even with padding,
        # as unused padded leaves remain '-inf'
        return self._max_priority if self.size > 0 else 0.0

    def _left(self, idx):
        """Returns the left child index of a given node."""
        return 2 * idx + 1

    def _right(self, idx):
        """Returns the right child index of a given node."""
        return 2 * idx + 2

    def _propagate_upwards(self, idx):
        """
        Propagates changes (sum/min/max) up the tree starting from a leaf node index.
        """
        while idx > 0:
            parent = (idx - 1) // 2
            left = self._left(parent)
            right = self._right(parent)

            self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
            self.min_tree[parent] = min(self.min_tree[left], self.min_tree[right])
            idx = parent

    def _retrieve(self, idx, s):
        """Finds the leaf index corresponding to a sample value 's'."""
        left = self._left(idx)
        right = self._right(idx)

        if left >= len(self.sum_tree):  # Reached a leaf node
            return idx

        return (
            self._retrieve(left, s)
            if s <= self.sum_tree[left]
            else self._retrieve(right, s - self.sum_tree[left])
        )

    def update(self, buffer_idx, p):
        """
        Updates the priority 'p' for a given *buffer* index.

        Args:
            buffer_idx (int): The index in the logical buffer (0 to buffer_capacity-1).
            p (float): The new priority value (must be non-negative).
        """
        if not (0 <= buffer_idx < self.buffer_capacity):
            raise IndexError(
                f"Buffer index {buffer_idx} out of range [0, {self.buffer_capacity - 1})."
            )
        if p < 0:
            raise ValueError("Priority must be non-negative.")

        # Convert buffer index to tree leaf index
        tree_idx = buffer_idx + self.tree_capacity - 1

        self.sum_tree[tree_idx] = p
        self.min_tree[tree_idx] = p
        self._max_priority = max(self._max_priority, p)
        self._propagate_upwards(tree_idx)

    def add(self, p):
        """
        Adds a new priority 'p' to the tree at the current write position.
        Uses the original buffer_capacity for cycling the write pointer.
        """
        # Get the buffer index for the current write position
        buffer_idx = self.write_idx

        # Update the tree at the corresponding leaf index
        self.update(buffer_idx, p)

        # Cycle the write pointer based on the *requested* buffer capacity
        self.write_idx = (self.write_idx + 1) % self.buffer_capacity
        self._size = min(self._size + 1, self.buffer_capacity)

    def get(self, s):
        """
        Gets the leaf index, priority, and corresponding *buffer* index
        for a given sample value 's'.
        """
        if self.total == 0:
            print("Warning: Sampling from SumTree with total priority zero.")
            # Need to decide how to handle this - raise error or return default?
            # Returning buffer index 0 as a placeholder, but might need adjustment.
            return (self.tree_capacity - 1, 0, 0)  # tree_idx, priority, buffer_idx

        tree_idx = self._retrieve(0, s)
        # Convert tree leaf index back to buffer index
        buffer_idx = tree_idx - self.tree_capacity + 1

        # Ensure the retrieved buffer index is valid (it might point to padded space
        # if total priority is low and sampling hits near the end). Usually, higher
        # level logic prevents sampling beyond n_entries, but this is a safeguard.
        if buffer_idx >= self.buffer_capacity:
            # This might indicate an issue or edge case, potentially sample again or clamp.
            # For simplicity, let's clamp to the last valid buffer index for now.
            print(
                f"Warning: Retrieved buffer index {buffer_idx} >= capacity {self.buffer_capacity}. Clamping."
            )
            buffer_idx = self.buffer_capacity - 1
            tree_idx = (
                buffer_idx + self.tree_capacity - 1
            )  # Recalculate tree_idx based on clamped buffer_idx

        return (self.sum_tree[tree_idx], buffer_idx)


class PrioritizedReplayBuffer:
    """
    Fixed-size buffer storing experience tuples, sampled based on priority using an updated SumTree
    """

    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(
        self,
        state_size,
        action_size,
        buffer_size,
        batch_size,
        seed,
        device,
        discrete_states=False,
        discrete_actions=True,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size  # This is the logical capacity
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device

        state_type = np.int64 if discrete_states else np.float32
        self.torch_state_t = torch.int64 if discrete_states else torch.float32
        action_type = np.int64 if discrete_actions else np.float32
        self.torch_act_t = torch.int64 if discrete_actions else torch.float32

        self.states = np.zeros((buffer_size, state_size), dtype=state_type)
        self.actions = np.zeros((buffer_size, 1), dtype=action_type)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_size), dtype=state_type)
        self.dones = np.zeros((buffer_size, 1), dtype=np.int8)

        self.tree = SumTree(buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    @property
    def size(self):
        return self.tree.size

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        # Assign high priority initially (e.g., 1.0)
        current_max_priority = max(1.0, self.tree.max_priority) ** self.alpha

        # Store experience in numpy arrays
        buffer_idx = self.tree.ptr  # Get current write index (buffer index)
        self.states[buffer_idx] = state
        self.actions[buffer_idx] = action
        self.rewards[buffer_idx] = reward
        self.next_states[buffer_idx] = next_state
        self.dones[buffer_idx] = done

        # Add priority to SumTree using the buffer index
        self.tree.add(current_max_priority)

    def sample(self):
        """Sample a batch of experiences from memory based on priorities."""
        if self.size == 0:
            raise BufferError("Cannot sample from an empty buffer.")

        current_batch_size = min(self.batch_size, self.size)

        idxs = np.empty((current_batch_size,), dtype=np.int32)  # Store buffer indices
        is_weights = np.empty((current_batch_size,), dtype=np.float32)
        max_weight = 1.0

        total_priority = self.tree.total
        if total_priority == 0 or self.alpha == 0:
            print("Warning: Total priority is zero. Sampling uniformly.")
            # Sample uniformly from valid *buffer* indices
            sampled_idxs = np.random.choice(
                self.size, current_batch_size, replace=(self.size < current_batch_size)
            )
            is_weights.fill(1.0)
            idxs = np.array(sampled_idxs, dtype=np.int32)
        else:
            priority_segment = total_priority / current_batch_size

            min_priority = self.tree.min_priority
            min_prob = min_priority / total_priority if total_priority > 0 else 0

            if self.size > 0 and min_prob > 0:
                max_weight = (self.size * min_prob) ** (-self.beta)
            else:
                max_weight = 1.0

            for i in range(current_batch_size):
                a = priority_segment * i
                b = priority_segment * (i + 1)
                s = random.uniform(a, b)

                priority, buffer_idx = self.tree.get(s)

                # Ensure buffer_idx is valid (safeguard, though `get` tries to handle it)
                buffer_idx = min(buffer_idx, self.size - 1)

                sampling_probability = priority / total_priority if total_priority > 0 else 0
                weight = (
                    (self.size * sampling_probability) ** (-self.beta)
                    if sampling_probability > 0
                    else 0
                )

                is_weights[i] = weight
                idxs[i] = buffer_idx  # Store the buffer index

            # Anneal beta towards 1.0
            self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

            if max_weight > 0:
                is_weights /= max_weight
            else:
                print(f"Warning: max_weight is {max_weight}, cannot normalize IS weights.")
                is_weights.fill(1.0)

        # Collect batch data using the sampled indices
        batch_states = self.states[idxs]
        batch_actions = self.actions[idxs]
        batch_rewards = self.rewards[idxs]
        batch_next_states = self.next_states[idxs]
        batch_dones = self.dones[idxs]

        # Convert batch to PyTorch tensors
        states_tensor = torch.from_numpy(batch_states).type(self.torch_state_t).to(self.device)
        actions_tensor = torch.from_numpy(batch_actions).type(self.torch_act_t).to(self.device)
        rewards_tensor = torch.from_numpy(batch_rewards).float().to(self.device)
        next_states_tensor = (
            torch.from_numpy(batch_next_states).type(self.torch_state_t).to(self.device)
        )
        dones_tensor = torch.from_numpy(batch_dones).float().to(self.device)
        is_weights_tensor = torch.from_numpy(is_weights).float().to(self.device)

        # Return buffer indices
        return (
            (
                states_tensor,
                actions_tensor,
                rewards_tensor,
                next_states_tensor,
                dones_tensor,
            ),
            is_weights_tensor,
            idxs,
        )

    def update_priorities(self, batch_indices, errors):
        """Updates priorities of sampled experiences using buffer indices."""
        if isinstance(errors, torch.Tensor):
            errors = errors.detach().cpu().numpy()

        errors = np.squeeze(errors)
        if errors.ndim == 0:
            errors = np.array([errors])

        if len(batch_indices) != len(errors):
            raise ValueError("Number of indices must match number of errors")

        for buffer_idx, error in zip(batch_indices, errors):
            # The user provides buffer indices (0 to buffer_size-1)
            if not (0 <= buffer_idx < self.size):
                print(
                    f"Warning: Attempting update for buffer index {buffer_idx} outside current size {self.size}. Skipping."
                )
                continue

            priority = self._get_priority(error)
            # Update using the buffer index; SumTree handles internal mapping
            self.tree.update(buffer_idx, priority)

    def __len__(self):
        return self.size


# --- Example Usage (remains the same, uses logical buffer_size) ---
if __name__ == "__main__":
    STATE_SIZE = 10
    BUFFER_SIZE = 1025  # Test with non-power-of-2
    BATCH_SIZE = 32
    ACTION_SIZE = 4
    SEED = 0
    DEVICE = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {DEVICE}")
    print(f"Requested Buffer Size: {BUFFER_SIZE}")

    buffer = PrioritizedReplayBuffer(STATE_SIZE, ACTION_SIZE, BUFFER_SIZE, BATCH_SIZE, SEED, DEVICE)
    print(f"Internal SumTree Capacity (Power of 2): {buffer.tree.tree_capacity}")
    print("Capacity: ", buffer.buffer_size, buffer.tree.buffer_capacity)

    def create_dummy_experience(state_size, action_size):
        state = np.random.randn(state_size).astype(np.float32)
        action = random.randint(0, action_size - 1)
        reward = random.uniform(-1, 1)
        next_state = np.random.randn(state_size).astype(np.float32)
        done = random.choice([True, False])
        return state, action, reward, next_state, done

    print("Buffer sizes: ", buffer.size, buffer.tree.size)
    print("write index: ", buffer.tree.ptr, buffer.tree.write_idx)

    print("Adding experiences...")
    num_to_add = BUFFER_SIZE // 2  # Fill partially
    for i in range(num_to_add):
        exp = create_dummy_experience(STATE_SIZE, ACTION_SIZE)
        buffer.add(*exp)

    print(f"Buffer size: {len(buffer)}")
    print("Buffer sizes: ", buffer.size, buffer.tree._size)
    print("write index: ", buffer.tree.ptr, buffer.tree.write_idx)
    print(f"SumTree total priority: {buffer.tree.total:.2f}")
    print(f"SumTree min priority (initial): {buffer.tree.min_priority:.6f}")
    print(f"SumTree max priority (initial): {buffer.tree.max_priority:.6f}")

    if len(buffer) >= BATCH_SIZE:
        print("\nSampling a batch...")
        experiences, is_weights, idxs = buffer.sample()
        print(f"Sampled buffer indices: {idxs[:5]}...")  # These are 0 to BUFFER_SIZE-1
        print(f"Sampled IS weights: {is_weights[:5].squeeze().cpu().numpy()}...")

        print("\nUpdating priorities with varying errors...")
        errors = np.concatenate(
            [
                np.random.rand(BATCH_SIZE // 2) * 0.1,
                np.random.rand(BATCH_SIZE - BATCH_SIZE // 2) * 10,
            ]
        )
        np.random.shuffle(errors)
        print(f"Dummy TD Errors (first 5): {errors[:5]}")

        # Pass buffer indices to update_priorities
        buffer.update_priorities(idxs, errors)
        print("Priorities updated.")
        print(f"SumTree total priority after update: {buffer.tree.total:.2f}")
        print(f"SumTree min priority after update: {buffer.tree.min_priority:.6f}")
        print(f"SumTree max priority after update: {buffer.tree.max_priority:.6f}")

        min_calc_p = (np.min(np.abs(errors)) + buffer.epsilon) ** buffer.alpha
        print(f"Expected min priority (approx):   {min_calc_p:.6f}")

        print(f"SumTree shape: {buffer.tree.sum_tree.shape}")

    else:
        print("Buffer not full enough to sample.")
