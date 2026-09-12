# Bomberman RL Agent — `Q_agent`

## Overview

This agent is a **tabular Q-learning agent** for the Bomberman environment.

The agent learns which action is best for a given situation by maintaining a **Q-table**. For every state, the Q-table stores a value for each possible action. These values represent the expected long-term usefulness of taking an action in that state.

The six possible actions are:

- `UP` — Move upward
- `RIGHT` — Move right
- `DOWN` — Move downward
- `LEFT` — Move left
- `WAIT` — Remain stationary
- `BOMB` — Place a bomb

The agent combines:

- Tabular Q-learning
- State abstraction
- BFS-based navigation features
- Time-aware danger detection
- Safety filtering
- Bomb-quality estimation
- Epsilon-greedy exploration

### Design Principle

> **BFS and safety logic provide information and constraints, while Q-learning remains responsible for the final strategic action selection.**

BFS does not directly choose the final action. The safety layer restricts dangerous actions, while Q-learning chooses among the remaining safe actions.

---

## Reinforcement Learning Formulation

The agent can be described as a **Markov Decision Process (MDP)**:

$$
(S, A, R, T)
$$

where:

- $S$ = set of possible states
- $A$ = set of possible actions
- $R$ = reward received after an action
- $T$ = transition to the next state

At each step, the agent:

1. Observes the current game state.
2. Converts it into a compact feature representation.
3. Looks up the corresponding state in the Q-table.
4. Determines which actions are physically valid.
5. Filters out unsafe actions.
6. Uses epsilon-greedy selection to choose the final action.
7. Receives a reward and transitions to the next state.
8. Updates the Q-value during training.

---

## Action Space

The agent has six possible actions:

| Action | Description |
|---|---|
| `UP` | Move upward |
| `RIGHT` | Move right |
| `DOWN` | Move downward |
| `LEFT` | Move left |
| `WAIT` | Remain stationary |
| `BOMB` | Place a bomb |

The Q-table stores one Q-value for each action.
The agent uses these values to estimate which action has the highest expected long-term value.

---

# State Representation

The agent converts the current game situation into a compact feature tuple:

```python
(
    danger,
    danger_direction,
    target_direction,
    bomb_quality,
    blocked_up,
    blocked_down,
    blocked_left,
    blocked_right,
    last_action,
)
```

This is the agent's **state abstraction**.

Instead of storing the complete board configuration as the Q-learning state, the agent extracts a small number of features describing the most relevant aspects of the current situation.

The nine features are:

1. `danger`
2. `danger_direction`
3. `target_direction`
4. `bomb_quality`
5. `blocked_up`
6. `blocked_down`
7. `blocked_left`
8. `blocked_right`
9. `last_action`

---

# State Features

## 1. Danger

The agent calculates a **time-aware danger map**.

Each board cell receives a value representing approximately how soon that cell can be affected by an explosion.

The time-to-explosion value (`tte`) is classified into four categories:

- `EXPLODING`
- `IMMINENT`
- `DELAYED`
- `SAFE`

The classification is:

```python
if tte == 0:
    return 'EXPLODING'

if tte <= 2:
    return 'IMMINENT'

if tte <= 3:
    return 'DELAYED'

return 'SAFE'
```

| Danger state | Meaning |
|---|---|
| `EXPLODING` | An explosion is occurring |
| `IMMINENT` | An explosion is expected very soon |
| `DELAYED` | An explosion is coming but not immediately |
| `SAFE` | There is no immediate explosion threat |

There are therefore **4 possible values** for the `danger` feature.

### Time-Aware Danger Detection

The danger system considers:

- Bomb position
- Bomb timer
- Blast radius
- Walls
- Chain reactions

The blast radius is:

```python
BLAST_RADIUS = 3
```

### Chain Reactions

The danger-map construction accounts for bombs triggering other bombs.

For example:

```text
Bomb A timer = 3
Bomb B timer = 5

If Bomb A can trigger Bomb B:

Bomb B timer becomes 3
```

This allows the agent to reason about future explosions caused by chain reactions.

---

## 2. Danger Direction

When the agent is in danger, it performs an escape search using a time-aware BFS.

The result can be:

```text
UP
RIGHT
DOWN
LEFT
NONE
```

For example:

```text
Agent is currently threatened.

Possible escape:
UP

danger_direction = UP
```

### Important

`danger_direction` does **not automatically force the agent to move in that direction**.

It is stored as a state feature. Q-learning still decides the final action.

---

## 3. Target Direction

`target_direction` represents the direction of the nearest useful navigation target.

The result can be:

```text
UP
RIGHT
DOWN
LEFT
NONE
```

The target hierarchy is:

```text
Coin
  ↓
Crate
  ↓
Opponent
  ↓
NONE
```

The agent therefore prioritizes:

1. Coin
2. Crate
3. Opponent
4. No target

If a coin is available, the agent first attempts to determine a direction toward a coin. If no usable coin target is available, it falls back to crates and then opponents.

### Coin Targeting

Coins are sorted according to **Manhattan distance**:

$$
d = |x_c - x| + |y_c - y|
$$

where:

- $(x,y)$ = current agent position
- $(x_c,y_c)$ = coin position

The coin with the smallest Manhattan distance is selected.

BFS is then used to determine the first movement direction toward that selected coin.

For example:

```text
Agent
  |
  |
  ↓
Coin
```

The resulting feature can be:

```python
target_direction = 'DOWN'
```

This direction is a navigation feature, not a forced action.

---

# BFS Navigation

**Breadth-First Search (BFS)** is used to determine navigation directions.

The BFS explores the board from the agent's current position and considers:

- Free tiles
- Tiles without bombs
- Tiles without opponents
- Safe tiles when danger avoidance is enabled

The BFS returns only the **first direction** of the path.

For example:

```text
Agent → → → Coin
```

BFS finds:

```text
RIGHT
RIGHT
RIGHT
```

The state feature becomes:

```python
target_direction = 'RIGHT'
```

### BFS Does Not Control the Final Action

Suppose BFS returns:

```text
target_direction = RIGHT
```

but the Q-table contains:

```text
RIGHT = 0.5
WAIT  = 1.2
BOMB  = 2.0
```

If `BOMB` is safe, Q-learning can select `BOMB`.

Therefore:

> **BFS provides navigation information; Q-learning makes the strategic decision.**

---

## 4. Bomb Quality

The agent estimates whether placing a bomb would be useful.

`bomb_quality` has four possible values:

```text
0
1
2
3
```

| Value | Meaning |
|---:|---|
| `0` | No useful/safe bomb opportunity |
| `1` | Bomb can destroy at least one crate |
| `2` | Bomb can destroy at least two crates |
| `3` | Bomb can hit an opponent |

`bomb_quality` is a **state feature** and does not directly force the agent to place a bomb.

Q-learning decides whether `BOMB` is the best safe action.

### Bomb Escape Calculation

Before considering a bomb useful, the agent checks whether it can escape the resulting explosion.

It simulates placing a bomb at the current position:

```python
simulated_bombs = list(bombs) + [
    ((x, y), BOMB_TIMER)
]
```

A new danger map is then constructed.

A time-aware BFS searches for an escape route.

If no escape route exists:

```text
bomb_quality = 0
```

This helps prevent suicidal bomb placements from being treated as useful opportunities.

---

## 5. Blocked Directions

The agent records whether each neighboring tile is blocked.

Four binary features are used:

```text
blocked_up
blocked_down
blocked_left
blocked_right
```

Each has two possible values:

```text
0 = not blocked
1 = blocked
```

For example:

```text
blocked_up    = 1
blocked_down  = 0
blocked_left  = 1
blocked_right = 0
```

This provides the Q-learning algorithm with information about its immediate movement possibilities.

A tile can be blocked by:

- Walls
- Crates
- Bombs
- Explosions
- Opponents
- Board boundaries

---

## 6. Last Action

The agent also stores the previous action.

There are six possible values:

```text
UP
RIGHT
DOWN
LEFT
WAIT
BOMB
```

This gives the state representation some short-term temporal context.

For example:

```text
Previous action = LEFT
```

is treated as a different state from:

```text
Previous action = RIGHT
```

This helps the agent distinguish between situations where it arrived at the current state through different recent actions.

It also supports the tie-breaking mechanism used to reduce immediate movement reversals.

---

# Complete State Space

The theoretical maximum number of state combinations can be calculated from the number of possible values of each feature.

| Feature | Possible values |
|---|---:|
| `danger` | 4 |
| `danger_direction` | 5 |
| `target_direction` | 5 |
| `bomb_quality` | 4 |
| `blocked_up` | 2 |
| `blocked_down` | 2 |
| `blocked_left` | 2 |
| `blocked_right` | 2 |
| `last_action` | 6 |

Therefore:

$$
4 	imes 5 	imes 5 	imes 4
	imes 2 	imes 2 	imes 2 	imes 2
	imes 6
$$

gives:

$$
oxed{38,400}
$$

possible feature states.

Each state has six possible actions.

Therefore, the theoretical maximum number of Q-values is:

$$
38,400 	imes 6
=
oxed{230,400}
$$

This is manageable for a tabular Q-learning implementation.

> **Note:** 38,400 is the theoretical maximum. The agent will not necessarily encounter all of these states because many combinations are impossible or rare in actual gameplay.

---

# Q-learning Algorithm

The fundamental Q-learning update is:

$$
Q(s,a) \leftarrow Q(s,a)
+
lpha
\left[
r + \gamma \max_{a'} Q(s',a')
-
Q(s,a)

ight]
$$

where:

- $s$ = current state
- $a$ = selected action
- $r$ = reward received
- $s'$ = next state
- $a'$ = possible next action
- $lpha$ = learning rate
- $\gamma$ = discount factor

The algorithm therefore learns:

> **"If I take action $a$ in state $s$, how good is the resulting future?"**

Over many training rounds, actions that lead to desirable outcomes should obtain higher Q-values.

---

# Exploration vs. Exploitation

The agent uses **epsilon-greedy exploration**.

With probability:

$$
\epsilon
$$

the agent explores by selecting a random safe action.

Otherwise, it exploits its learned knowledge by selecting the safe action with the highest Q-value.

This balances:

- **Exploration** — discovering potentially better actions
- **Exploitation** — using actions that the Q-table already considers valuable

Exploration is restricted to actions that pass the safety filter.

---

# Safety Layer

Before Q-learning selects the final action, the agent determines:

```text
valid_actions
```

and then:

```text
safe_actions
```


---

## Physical Validity

`get_valid_actions()` checks whether an action is physically possible.

For movement, it checks:

- Board boundaries
- Walls
- Crates
- Bombs
- Explosions
- Opponents

`WAIT` is normally valid.

`BOMB` is valid if the game state indicates that the agent can currently place a bomb.

Therefore, a physically valid action is an action permitted by the game environment.

---

## Safety Filtering

`get_safe_actions()` removes actions that are predicted to be dangerous.

### Movement

The agent checks:

> Can the agent reach the destination tile safely?

### Waiting

The agent checks:

> Is the current tile sufficiently safe to remain on?

### Bombing

The agent checks:

> Can the agent escape the bomb explosion?

Therefore:

$$
	ext{valid} 
eq 	ext{necessarily safe}
$$

An action can be physically valid but strategically unsafe.

For example, `BOMB` may be physically valid because the agent has a bomb available, but it may be removed by the safety layer if the agent cannot escape the resulting explosion.

---

# Emergency Fallback

There can be situations where no action passes the normal safety filter:

```python
safe_actions = []
```

Instead of returning no action, the agent uses an emergency fallback.

For available movement actions, it evaluates the danger-map value of the destination tiles and chooses the movement option with the highest safety value.

Conceptually:

```text
No completely safe action
          ↓
Check movement options
          ↓
Compare danger-map values
          ↓
Choose the least dangerous option
```

This provides a fallback when the agent is trapped or has very limited options.

---

# Tie-Breaking

When multiple safe actions have exactly the same highest Q-value, the agent uses a small tie-breaking rule.

It checks the previous action and tries to avoid immediately reversing direction.

For example:

```text
Previous action = RIGHT
Opposite action = LEFT
```

If `LEFT` and another action have the same Q-value, the non-reversing action is preferred.

This helps reduce simple oscillation patterns such as:

```text
RIGHT
LEFT
RIGHT
LEFT
RIGHT
LEFT
```

The tie-breaking mechanism does not override a clearly higher Q-value. It is only applied when actions have equal best values.

---


# Why State Abstraction Is Used

Representing the entire Bomberman board directly would create a very large state space.

Instead, `Q_agent` extracts a compact representation containing:

- Current danger
- Possible escape direction
- Navigation target
- Bomb usefulness
- Local obstacles
- Previous action

This reduces the number of possible states and makes a tabular approach practical.

The agent therefore focuses on information useful for decision-making rather than memorizing every possible board configuration.

---
 

 # Three agents 

 Tried building three agents by tuning the rewards

 - Agent : 
 ``` 
 game_rewards = {

        e.COIN_COLLECTED: 100.0,

        e.CRATE_DESTROYED: 40.0,

        e.KILLED_OPPONENT: 200.0,

        e.KILLED_SELF: -800.0,

        e.GOT_KILLED: -500.0,

        e.INVALID_ACTION: -10.0,

        ESCAPED_DANGER: 20.0,

        MOVED_TOWARDS_TARGET: 5.0,

        PLACED_KILL_BOMB: 20.0,

        PLACED_HIGH_VALUE_BOMB: 15.0,

        PLACED_LOW_VALUE_BOMB: -10.0,

        USELESS_WAIT: -2.0,

        OSCILLATION_PENALTY: -20.0,
    }
```

- Agent stats : 
```
"Q_agent": {
            "bombs": 35165,
            "coins": 616,
            "crates": 14686,
            "invalid": 813,
            "kills": 170,
            "moves": 154660,
            "rounds": 1000,
            "score": 1466,
            "steps": 268158,
            "suicides": 350,
            "time": 32.98667073249817
        }
```

- Agent 1 : 
```
 game_rewards = {

        e.COIN_COLLECTED: 100.0,

        e.CRATE_DESTROYED: 40.0,

        e.KILLED_OPPONENT: 200.0,

        e.KILLED_SELF: -800.0,

        e.GOT_KILLED: -500.0,

        e.INVALID_ACTION: -10.0,

        ESCAPED_DANGER: 20.0,

        MOVED_TOWARDS_TARGET: 3.0,

        PLACED_KILL_BOMB: 60.0,

        PLACED_HIGH_VALUE_BOMB: 15.0,

        PLACED_LOW_VALUE_BOMB: -10.0,

        USELESS_WAIT: -2.0,

        OSCILLATION_PENALTY: -10.0,
    }
```    

- Agent1 stats :
```
 "Q_agent": {
            "bombs": 36239,
            "coins": 533,
            "crates": 13235,
            "invalid": 1069,
            "kills": 193,
            "moves": 186360,
            "rounds": 1000,
            "score": 1498,
            "steps": 270191,
            "suicides": 289,
            "time": 32.91277194023132
        }
```        


- Agent2 : 
```
  game_rewards = {

    e.COIN_COLLECTED: 150.0,

    e.CRATE_DESTROYED: 30.0,

    e.KILLED_OPPONENT: 200.0,

    e.KILLED_SELF: -1000.0,

    e.GOT_KILLED: -600.0,

    e.INVALID_ACTION: -10.0,

    ESCAPED_DANGER: 20.0,

    MOVED_TOWARDS_TARGET: 5.0,

    PLACED_KILL_BOMB: 40.0,

    PLACED_HIGH_VALUE_BOMB: 10.0,

    PLACED_LOW_VALUE_BOMB: -10.0,

    USELESS_WAIT: -2.0,

    OSCILLATION_PENALTY: -10.0,
}
```
- Agent2 stats :
```
 "Q_agent": {
            "bombs": 34745,
            "coins": 680,
            "crates": 14140,
            "invalid": 914,
            "kills": 130,
            "moves": 176833,
            "rounds": 1000,
            "score": 1330,
            "steps": 268332,
            "suicides": 308,
            "time": 34.575828313827515
        }
```