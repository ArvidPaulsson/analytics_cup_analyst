# Alternative Game Representation Structures

Beyond SPADL, there are many creative ways to represent football games. Here are some innovative structures that leverage SkillCorner's rich tracking and contextual data.

## 1. Graph-Based Representation: Game as a Network

### Concept
Represent the game as a **dynamic graph** where:
- **Nodes** = Players, Ball, Zones, Phases
- **Edges** = Passes, Pressures, Runs, Spatial Relationships
- **Attributes** = Time, Location, Context

### Structure

```python
{
    "nodes": [
        {
            "id": "player_51649",
            "type": "player",
            "team": 1805,
            "position": "LCB",
            "attributes": {
                "x": -22.31,
                "y": 1.22,
                "speed": 5.34,
                "in_possession": True
            }
        },
        {
            "id": "zone_defensive_third",
            "type": "zone",
            "team": 1805,
            "attributes": {
                "phase_type": "build_up",
                "n_players": 4
            }
        }
    ],
    "edges": [
        {
            "source": "player_51649",
            "target": "player_735578",
            "type": "passing_option",
            "timestamp": 3.8,
            "attributes": {
                "xpass_completion": 0.9865,
                "xthreat": 0.0001,
                "distance": 15.1,
                "executed": True
            }
        },
        {
            "source": "player_50951",
            "target": "player_51649",
            "type": "pressing",
            "timestamp": 2.4,
            "attributes": {
                "pressing_chain_id": 442,
                "distance": 9.48,
                "successful": False
            }
        }
    ],
    "temporal_snapshots": [
        {
            "frame": 48,
            "timestamp": 3.8,
            "graph_state": {...}
        }
    ]
}
```

### Use Cases
- **Passing Networks**: Analyze team connectivity
- **Pressing Networks**: Study defensive coordination
- **Spatial Networks**: Understand zone relationships
- **Temporal Evolution**: Track network changes over time

---

## 2. State-Space Representation: Game as State Transitions

### Concept
Represent the game as a sequence of **states** and **transitions**:
- **States** = Game situations (possession, phase, score, location)
- **Transitions** = Actions that change state
- **Rewards** = Value of state changes (xThreat, xGoal)

### Structure

```python
{
    "states": [
        {
            "state_id": "s_001",
            "timestamp": 3.8,
            "frame": 48,
            "attributes": {
                "possession_team": 1805,
                "possession_player": 51649,
                "phase_type": "create",
                "opponent_phase": "medium_block",
                "location": {
                    "x": -22.31,
                    "y": 1.22,
                    "third": "defensive_third",
                    "channel": "center"
                },
                "score": {"home": 0, "away": 0},
                "game_state": "drawing",
                "n_passing_options": 2,
                "n_opponents_ahead": 10,
                "defensive_structure": {
                    "n_lines": 3,
                    "organised": True,
                    "last_line_x": 18.17
                }
            }
        }
    ],
    "transitions": [
        {
            "from_state": "s_001",
            "to_state": "s_002",
            "action": {
                "type": "pass",
                "player": 51649,
                "target": 735578,
                "timestamp": 4.8
            },
            "reward": {
                "xthreat_delta": 0.0001,
                "xloss_delta": -0.0,
                "phase_transition": False
            }
        }
    ]
}
```

### Use Cases
- **Reinforcement Learning**: Train agents to make decisions
- **Value Estimation**: Calculate state values
- **Tactical Analysis**: Study state transition patterns
- **Game Simulation**: Model game flow

---

## 3. Hierarchical Tree Structure: Game as Nested Sequences

### Concept
Represent the game as a **hierarchical tree**:
- **Root** = Match
- **Level 1** = Periods
- **Level 2** = Possessions/Phases
- **Level 3** = Actions/Events
- **Level 4** = Sub-actions (passing options, runs)

### Structure

```python
{
    "match_id": 1886347,
    "periods": [
        {
            "period": 1,
            "start_time": 0.0,
            "end_time": 2778.0,
            "possessions": [
                {
                    "possession_id": "p_001",
                    "team": 1805,
                    "start_time": 1.8,
                    "end_time": 7.9,
                    "phase_type": "create",
                    "actions": [
                        {
                            "action_id": "a_001",
                            "type": "player_possession",
                            "player": 51649,
                            "start_time": 3.8,
                            "end_time": 4.8,
                            "sub_actions": [
                                {
                                    "type": "passing_option",
                                    "target": 735578,
                                    "xpass": 0.9865,
                                    "executed": True
                                },
                                {
                                    "type": "passing_option",
                                    "target": 50978,
                                    "xpass": 0.9797,
                                    "executed": False
                                },
                                {
                                    "type": "off_ball_run",
                                    "player": 735573,
                                    "subtype": "pulling_wide"
                                }
                            ],
                            "context": {
                                "n_passing_options": 2,
                                "n_off_ball_runs": 1,
                                "pressing_chain": {
                                    "id": 442,
                                    "length": 3,
                                    "engagements": [
                                        {"player": 50951, "type": "pressing"}
                                    ]
                                }
                            }
                        }
                    ],
                    "outcome": {
                        "lead_to_shot": False,
                        "lead_to_goal": False,
                        "phase_transition": True
                    }
                }
            ]
        }
    ]
}
```

### Use Cases
- **Possession Analysis**: Study complete possession sequences
- **Phase Analysis**: Analyze phase transitions
- **Tactical Patterns**: Identify recurring patterns
- **Decision Trees**: Model decision-making processes

---

## 4. Multi-Dimensional Tensor: Game as Spatial-Temporal Grid

### Concept
Represent the game as a **4D tensor**:
- **Dimensions**: [Time, X, Y, Features]
- **Features**: Player positions, ball, pressure, passing options, etc.
- **Grid Resolution**: 1m x 1m cells, 0.1s time steps

### Structure

```python
{
    "tensor_shape": [59042, 105, 68, 50],  # [frames, x_cells, y_cells, features]
    "features": [
        "ball_x", "ball_y", "ball_z",
        "player_51649_x", "player_51649_y", "player_51649_speed",
        "player_735578_x", "player_735578_y", "player_735578_speed",
        # ... all players
        "pressure_intensity",  # Aggregated pressure at each cell
        "passing_option_density",  # Number of passing options
        "defensive_line_x",  # X position of defensive line
        "xthreat_map",  # Expected threat at each location
        "possession_probability"  # Probability of possession at each cell
    ],
    "metadata": {
        "pitch_length": 104,
        "pitch_width": 68,
        "cell_size": 1.0,  # meters
        "frame_rate": 10  # fps
    }
}
```

### Use Cases
- **Deep Learning**: Feed directly into CNNs/RNNs
- **Heatmaps**: Generate spatial heatmaps
- **Pattern Recognition**: Find spatial-temporal patterns
- **Simulation**: Predict future game states

---

## 5. Event Stream: Game as Continuous Events

### Concept
Represent the game as a **continuous stream of events** with rich context:
- **Events** = Any change in game state
- **Context** = Full game state at event time
- **Relationships** = Links between related events

### Structure

```python
{
    "events": [
        {
            "event_id": "e_001",
            "timestamp": 3.8,
            "frame": 48,
            "type": "possession_start",
            "player": 51649,
            "team": 1805,
            "context": {
                "location": {"x": -22.31, "y": 1.22},
                "phase": "create",
                "score": {"home": 0, "away": 0},
                "n_passing_options": 2,
                "n_opponents_ahead": 10
            },
            "related_events": [
                {"id": "e_002", "type": "passing_option_available"},
                {"id": "e_003", "type": "off_ball_run_started"},
                {"id": "e_004", "type": "pressing_started"}
            ]
        },
        {
            "event_id": "e_002",
            "timestamp": 3.8,
            "frame": 48,
            "type": "passing_option_available",
            "from_player": 51649,
            "to_player": 735578,
            "context": {
                "xpass_completion": 0.9865,
                "xthreat": 0.0001,
                "distance": 15.1,
                "dangerous": False
            },
            "related_events": [
                {"id": "e_001", "type": "possession_start"}
            ]
        },
        {
            "event_id": "e_005",
            "timestamp": 4.8,
            "frame": 58,
            "type": "pass_executed",
            "from_player": 51649,
            "to_player": 735578,
            "context": {
                "successful": True,
                "xthreat_delta": 0.0001,
                "phase_transition": False
            },
            "related_events": [
                {"id": "e_002", "type": "passing_option_available"},
                {"id": "e_006", "type": "possession_transfer"}
            ]
        }
    ],
    "event_types": [
        "possession_start", "possession_end",
        "pass_executed", "passing_option_available",
        "off_ball_run_started", "off_ball_run_ended",
        "pressing_started", "pressing_ended",
        "phase_transition", "shot_taken", "goal_scored"
    ]
}
```

### Use Cases
- **Real-time Analysis**: Process events as they happen
- **Event Correlation**: Find relationships between events
- **Pattern Mining**: Discover frequent event sequences
- **Causal Analysis**: Understand cause-effect relationships

---

## 6. Entity-Component System: Game as Entities and Components

### Concept
Represent the game using **Entity-Component-System** pattern:
- **Entities** = Players, Ball, Zones, Phases
- **Components** = Position, Velocity, Possession, Pressure, etc.
- **Systems** = Movement, Passing, Pressing, Phase Detection

### Structure

```python
{
    "entities": {
        "player_51649": {
            "components": {
                "position": {"x": -22.31, "y": 1.22, "z": 0},
                "velocity": {"x": 0.1, "y": 1.48, "speed": 5.34},
                "possession": {
                    "has_ball": True,
                    "possession_start": 3.8,
                    "n_passing_options": 2
                },
                "pressure": {
                    "under_pressure": True,
                    "pressers": ["player_50951"],
                    "pressing_chain_id": 442
                },
                "phase": {
                    "team_phase": "create",
                    "opponent_phase": "medium_block"
                },
                "metrics": {
                    "xthreat": 0.0001,
                    "xloss": 0.0,
                    "separation": 8.66
                }
            }
        },
        "ball": {
            "components": {
                "position": {"x": -22.21, "y": 2.7, "z": 0.1},
                "velocity": {"x": 0.1, "y": 1.48, "speed": 1.49},
                "possession": {
                    "owner": "player_51649",
                    "team": 1805
                }
            }
        },
        "zone_defensive_third_1805": {
            "components": {
                "position": {"x_min": -52, "x_max": -17.33, "y_min": -34, "y_max": 34},
                "occupancy": {
                    "team_1805": 4,
                    "team_4177": 1
                },
                "phase": {
                    "team_phase": "build_up",
                    "opponent_phase": "high_block"
                }
            }
        }
    },
    "systems": {
        "movement": {
            "updates": ["position", "velocity"],
            "frequency": 10  # fps
        },
        "possession": {
            "updates": ["possession", "passing_options"],
            "frequency": 1  # per event
        },
        "pressure": {
            "updates": ["pressure", "pressing_chains"],
            "frequency": 1  # per event
        }
    }
}
```

### Use Cases
- **Game Engines**: Build football simulation engines
- **Modular Analysis**: Analyze specific components independently
- **System Interactions**: Study how systems interact
- **Performance**: Efficient updates of specific components

---

## 7. Functional/Declarative: Game as Transformations

### Concept
Represent the game as a **series of transformations**:
- **Initial State** = Kickoff
- **Transformations** = Functions that modify state
- **Composition** = Chain transformations to build game

### Structure

```python
{
    "initial_state": {
        "timestamp": 0.0,
        "score": {"home": 0, "away": 0},
        "possession": None,
        "phase": None
    },
    "transformations": [
        {
            "type": "start_possession",
            "timestamp": 1.8,
            "function": "start_possession(player=51649, team=1805, location=(-22.31, 1.22))",
            "result": {
                "possession": {"player": 51649, "team": 1805},
                "phase": "create"
            }
        },
        {
            "type": "add_passing_option",
            "timestamp": 3.8,
            "function": "add_passing_option(from=51649, to=735578, xpass=0.9865)",
            "result": {
                "passing_options": [{"to": 735578, "xpass": 0.9865}]
            }
        },
        {
            "type": "add_pressing",
            "timestamp": 2.4,
            "function": "add_pressing(presser=50951, target=51649, chain_id=442)",
            "result": {
                "pressure": {"player": 51649, "pressers": [50951]}
            }
        },
        {
            "type": "execute_pass",
            "timestamp": 4.8,
            "function": "execute_pass(from=51649, to=735578, successful=True)",
            "result": {
                "possession": {"player": 735578, "team": 1805},
                "xthreat_delta": 0.0001
            }
        }
    ]
}
```

### Use Cases
- **Functional Programming**: Compose game from pure functions
- **Replay/Undo**: Easily replay or undo transformations
- **Testing**: Test individual transformations
- **Parallel Processing**: Process transformations in parallel

---

## 8. Probabilistic Graphical Model: Game as Bayesian Network

### Concept
Represent the game as a **probabilistic graphical model**:
- **Nodes** = Random variables (outcomes, states)
- **Edges** = Conditional dependencies
- **Probabilities** = Likelihood of outcomes

### Structure

```python
{
    "variables": {
        "pass_success": {
            "type": "bernoulli",
            "parents": ["pass_distance", "n_opponents_ahead", "xpass_completion"],
            "distribution": "P(success | distance, opponents, xpass)"
        },
        "xthreat": {
            "type": "continuous",
            "parents": ["location", "n_passing_options", "defensive_structure"],
            "distribution": "N(μ(location, options, defense), σ²)"
        },
        "phase_transition": {
            "type": "categorical",
            "parents": ["current_phase", "location", "n_passing_options"],
            "distribution": "P(transition | phase, location, options)"
        }
    },
    "observations": [
        {
            "variable": "pass_success",
            "value": True,
            "evidence": {
                "pass_distance": 15.1,
                "n_opponents_ahead": 10,
                "xpass_completion": 0.9865
            }
        }
    ],
    "inferences": [
        {
            "query": "P(goal | current_state)",
            "result": 0.0124
        },
        {
            "query": "P(best_pass_option | current_state)",
            "result": {"to_player": 735578, "probability": 0.8336}
        }
    ]
}
```

### Use Cases
- **Decision Making**: Make optimal decisions under uncertainty
- **Value Estimation**: Estimate action values probabilistically
- **Causal Inference**: Understand causal relationships
- **Prediction**: Predict future game states

---

## 9. Time-Series Representation: Game as Multivariate Time Series

### Concept
Represent the game as **multivariate time series**:
- **Time Index** = Frame/Timestamp
- **Variables** = All game features (positions, speeds, metrics)
- **Sampling Rate** = 10 fps

### Structure

```python
{
    "time_series": {
        "index": [0.0, 0.1, 0.2, ..., 5904.1],  # timestamps
        "variables": {
            "ball_x": [0.32, 0.54, 0.57, ...],
            "ball_y": [0.38, 0.08, -0.07, ...],
            "player_51649_x": [-39.63, -39.86, -40.06, ...],
            "player_51649_y": [1.22, 1.22, 1.22, ...],
            "player_51649_speed": [None, None, 5.34, ...],
            "possession_player": [None, None, 51649, ...],
            "n_passing_options": [0, 0, 2, ...],
            "xthreat": [0.0, 0.0, 0.0001, ...],
            "phase_type": [None, None, "create", ...],
            "pressure_intensity": [0.0, 0.0, 0.5, ...]
        }
    },
    "metadata": {
        "sampling_rate": 10,  # fps
        "duration": 5904.2,  # seconds
        "n_frames": 59042
    }
}
```

### Use Cases
- **Time-Series Analysis**: ARIMA, LSTM, Transformer models
- **Anomaly Detection**: Find unusual patterns
- **Forecasting**: Predict future game states
- **Signal Processing**: Apply DSP techniques

---

## 10. Hybrid Multi-View Representation

### Concept
Combine multiple representations in a **unified multi-view system**:
- **View 1**: Graph (for network analysis)
- **View 2**: State-Space (for decision-making)
- **View 3**: Tensor (for deep learning)
- **View 4**: Event Stream (for real-time processing)

### Structure

```python
{
    "views": {
        "graph": {...},  # Graph representation
        "state_space": {...},  # State-space representation
        "tensor": {...},  # Tensor representation
        "events": {...}  # Event stream representation
    },
    "mappings": {
        "graph_to_state": "function to convert graph to state",
        "state_to_tensor": "function to convert state to tensor",
        "tensor_to_events": "function to extract events from tensor"
    },
    "unified_query": {
        "interface": "Query any representation",
        "examples": [
            "Get passing network (graph view)",
            "Get optimal action (state-space view)",
            "Get heatmap (tensor view)",
            "Get event sequence (event view)"
        ]
    }
}
```

### Use Cases
- **Multi-Modal Analysis**: Use best representation for each task
- **Flexible Queries**: Query game from different perspectives
- **Integration**: Combine insights from different views
- **Scalability**: Use efficient representation for each use case

---

## Comparison Matrix

| Representation | Best For | Complexity | Scalability | Real-time |
|----------------|----------|------------|-------------|-----------|
| **Graph** | Network analysis | Medium | High | Medium |
| **State-Space** | Decision-making | High | Medium | Low |
| **Hierarchical** | Possession analysis | Medium | Medium | Medium |
| **Tensor** | Deep learning | High | Low | Low |
| **Event Stream** | Real-time processing | Low | High | High |
| **Entity-Component** | Simulation | Medium | Medium | High |
| **Functional** | Composition/Testing | Low | High | Medium |
| **Probabilistic** | Uncertainty/Decision | High | Low | Low |
| **Time-Series** | Forecasting | Medium | Medium | Medium |
| **Multi-View** | Comprehensive analysis | Very High | Medium | Medium |

---

## Recommendations

1. **Start Simple**: Event Stream or Hierarchical Tree
2. **For ML**: Tensor or Time-Series
3. **For Analysis**: Graph or State-Space
4. **For Real-time**: Event Stream or Entity-Component
5. **For Research**: Multi-View Hybrid

Each representation offers unique insights and is suited for different analytical tasks!

