# DQNResNet — Architecture

The main job of the network is to take the current game state and estimate a Q-value for each possible action.

The game board is `17 × 17`, and the states are represented using 12 channels. The network then processes this information and finally gives me 6 posisble actions Q-values.

## Network Structure

The overall structure of the network is:

```text
Input
(B, 12, 17, 17)
       │
       ▼
Conv2d 3×3
12 → 64 channels
       │
       ▼
GroupNorm + ReLU
       │
       ▼
ResBlock 1
       │
       ▼
ResBlock 2
       │
       ▼
Flatten
       │
       ▼
Linear
18496 → 512
       │
       ▼
ReLU
       │
       ▼
Linear
512 → 6
       │
       ▼
Q-values
(B, 6)
```

The convolutional part keeps the board size at `17 × 17`. This means the network doesn't reduce the board size while extracting features.

---

## 1. Spatial Attention

A small `SpatialAttention` module is used  inside each residual block.

The reason for using it is that some positions on the board are more important than others. For example, the agent might need to pay more attention to a nearby coin, bomb, wall, or dangerous position.

The attention layer helps the network learn which parts of the board are more important.

### How it works

If the input has the shape:

```text
(B, 64, 17, 17)
```

the module first calculates the average value across all channels:


This gives:

```text
(B, 1, 17, 17)
```

It also calculates the maximum value across the channels:

which also gives:

```text
(B, 1, 17, 17)
```

These two outputs are combined:

```text
(B, 1, 17, 17)
        +
(B, 1, 17, 17)
        ↓
(B, 2, 17, 17)
```

Then I pass them through a `7 × 7` convolution and a sigmoid:


This produces an attention map:

```text
(B, 1, 17, 17)
```

Finally, the attention map is multiplied with the original feature map:


So the output still has the same shape:

```text
(B, 64, 17, 17)
```

Basically, the attention layer changes **which positions are emphasized**, but it doesn't change the size of the board or the number of channels.

---

## 2. Residual Block

The network has two residual blocks:

Each block has two convolutional layers, GroupNorm, ReLU, and the spatial attention module.

The structure is:

```text
Input
  │
  ▼
Conv 3×3
  │
  ▼
GroupNorm
  │
  ▼
ReLU
  │
  ▼
Conv 3×3
  │
  ▼
GroupNorm
  │
  ▼
Spatial Attention
  │
  ▼
Add original input
  │
  ▼
ReLU
  │
  ▼
Output
```

The important part is the skip connection.

At the beginning of the block I save the original input:

```python
residual = x
```

Then, after processing the input, I add it back:

```python
out = out + residual
```

So the block is basically learning some additional features while keeping the original information.

---

## 3. GroupNorm

I use `GroupNorm` instead of `BatchNorm`.

For example:

```python
nn.GroupNorm(4, 64)
```

means that the 64 channels are split into 4 groups:

```text
64 / 4 = 16 channels per group
```

One reason I use GroupNorm is that the agent can sometimes make a decision using just one state.

During training I might have a batch of many states, but during gameplay I may pass only one state through the network.

GroupNorm does not depend on the batch size in the same way as BatchNorm, so it works well for both situations.

---

## 4. DQNResNet

The main model is:

```python
DQNResNet(
    input_channels=12,
    num_actions=6
)
```

So the network expects:

```text
12 channels
17 × 17 board
```

and produces:

```text
6 Q-values
```

The 12 input channels come from the state representation used by the agent.

The six outputs correspond to the six actions used by the agent. For example:

```text
0 → UP
1 → RIGHT
2 → DOWN
3 → LEFT
4 → WAIT
5 → BOMB
```

The exact mapping depends on how the actions are defined in the agent.

---

## 5. The Stem

The first part of the network is called the stem:

```python
self.stem = nn.Sequential(

    nn.Conv2d(
        input_channels,
        64,
        kernel_size=3,
        padding=1
    ),

    nn.GroupNorm(
        4,
        64
    ),

    nn.ReLU()
)
```

The first convolution changes the number of channels from 12 to 64:

```text
(B, 12, 17, 17)
        ↓
(B, 64, 17, 17)
```

The board is still `17 × 17`.

The idea here is to take the original state representation and turn it into 64 learned feature maps that the rest of the network can work with.

---

## 6. Residual Blocks

After the stem, the data goes through the two residual blocks:

```python
out = self.res1(out)
out = self.res2(out)
```

The shape doesn't change:

```text
(B, 64, 17, 17)
        ↓
(B, 64, 17, 17)
        ↓
(B, 64, 17, 17)
```

Both blocks also contain the spatial attention mechanism described earlier.

So by the time the data reaches the head, the network has processed the board several times while keeping the original `17 × 17` spatial structure.

---

## 7. Flattening

After the two residual blocks, I flatten the feature maps:

```python
nn.Flatten()
```

Before flattening:

```text
(B, 64, 17, 17)
```

The total number of values for each state is:

```text
64 × 17 × 17 = 18,496
```

So the shape becomes:

```text
(B, 18,496)
```

---

## 8. Fully Connected Layers

The flattened features are passed into:

```python
nn.Linear(
    64 * 17 * 17,
    512
)
```

So:

```text
18,496 → 512
```

Then a ReLU is applied.

Finally:

```python
nn.Linear(
    512,
    num_actions
)
```

changes:

```text
512 → 6
```

These six values are the final Q-values.

---

## 9. Q-values

The output might look something like:

```text
[1.2, 0.8, 2.4, 0.5, 0.1, -0.7]
```

If the action mapping is:

```text
UP     = 1.2
RIGHT  = 0.8
DOWN   = 2.4
LEFT   = 0.5
WAIT   = 0.1
BOMB   = -0.7
```

then `DOWN` currently has the highest predicted Q-value.

The important thing is that these are **Q-values, not probabilities**.

That's why there is no softmax at the end of the network.

The values can be positive or negative.

The DQN training process uses these values together with the Bellman target to update the network.

---

## 10. Parameter Count

The network has about **9.63 million parameters**.

| Part                 |    Parameters |
| -------------------- | ------------: |
| Stem                 |         7,104 |
| ResBlock 1           |        74,210 |
| ResBlock 2           |        74,210 |
| Fully connected head |     9,473,542 |
| **Total**            | **9,629,066** |

Most of the parameters are in the fully connected part.

In particular, this layer:

```python
nn.Linear(18496, 512)
```

has around 9.47 million parameters.

This happens because I flatten the complete:

```text
64 × 17 × 17
```

feature map before sending it into the linear layer.

So although the convolutional part looks fairly large, most of the model's parameters actually come from the first fully connected layer.

---

## 11. Complete Data Flow

The complete flow through the network is:

```text
Game state
(B, 12, 17, 17)
        │
        ▼
Conv2d 3×3
12 → 64
        │
        ▼
GroupNorm + ReLU
        │
        ▼
ResBlock 1
        │
        ▼
ResBlock 2
        │
        ▼
(B, 64, 17, 17)
        │
        ▼
Flatten
        │
        ▼
18,496 features
        │
        ▼
Linear
18,496 → 512
        │
        ▼
ReLU
        │
        ▼
Linear
512 → 6
        │
        ▼
6 Q-values
```

---
## 12. Why I Used This Architecture

The main reason for this architecture is that the game state is a **grid**, so convolutional layers make sense for extracting information from nearby cells.

I then use residual blocks to process these features further without losing the original information.

The spatial attention is there to help the network focus on important parts of the board.

Finally, the fully connected layers convert all these features into the six Q-values needed by the DQN.

So the basic idea is:

```text
Understand the board
        ↓
Extract useful features
        ↓
Focus on important positions
        ↓
Estimate how good each action is
```

This network is only the **Q-network**. Other parts of the DQN, such as the replay buffer, epsilon-greedy exploration, target network, reward calculation, and training loop are handled separately.
