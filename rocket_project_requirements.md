# Project Requirements

## Purpose
Create a Python Rocket Landing game and Deep Reinforcement Learning training tool with a desktop GUI, a self-implemented physics/game engine, and a PyTorch PPO training system.

## Functional Requirements

1. The project must launch from [run.py](C:\Users\TheunisvanNiekerk\Code\ptest\run.py).

2. `run.py` must support these entry modes:
   - `python run.py` for the desktop GUI
   - `python run.py gui`
   - `python run.py headless-train --generations N --games M`
   - `python run.py smoke-test`

3. All installable third-party dependencies must be listed in [requirements.txt](C:\Users\TheunisvanNiekerk\Code\ptest\requirements.txt), and `pip install -r requirements.txt` must work on Python 3.12.

4. Python 3.12 is the target runtime. PyTorch must be used for the PPO actor-critic model and training loop.

5. The GUI must use a Python-native desktop toolkit that runs reliably in the target Python 3.12 environment. Current implementation target: `tkinter`.

6. The codebase must remain modular:
   - game/environment logic separated from GUI
   - PPO/model/training logic separated from GUI
   - UI controls separated from UI drawing/rendering
   - config validation separated from GUI widgets

7. Core modules must be independently testable from the command line:
   - [environment.py](C:\Users\TheunisvanNiekerk\Code\ptest\rocket_lander\environment.py)
   - [ppo.py](C:\Users\TheunisvanNiekerk\Code\ptest\rocket_lander\ppo.py)
   - [training.py](C:\Users\TheunisvanNiekerk\Code\ptest\rocket_lander\training.py)

8. The rocket landing environment must be self-implemented and not depend on Gym or another prebuilt landing environment.

9. The rocket spawn must be randomized and “dramatic,” meaning spawns should often begin far from center, at varied height, angle, and velocity, while still remaining solvable.

10. The environment must expose a continuous action space with:
    - throttle
    - turn/gimbal

11. PPO must use an actor-critic design:
    - actor outputs continuous control actions
    - critic outputs scalar value estimates for training
    - critic output is not required in the GUI

12. The PPO trainer must collect data using batched multi-environment rollouts rather than only one episode at a time.

13. Observation normalization must be included in training and must also be reused during evaluation/playback.

14. Training stop behavior must be precise:
    - `Pause` finishes the current generation
    - `Stop` finishes currently active episodes, does not start new ones, and may produce a partial generation

15. Partial generations must not overwrite the saved “best brain so far.”

16. The app must track both:
    - current brain
    - best brain so far

17. “Best brain so far” must be determined primarily by landing rate and secondarily by mean score.

18. During pause or after training, the app must continuously evaluate the selected brain in the live game view.

19. The GUI layout must contain three major regions:
    - left scrollable configuration panel
    - center panel split vertically into game view and training graph
    - right resizable neural-network visualization panel

20. The left panel must scroll vertically only and contain:
    - session controls
    - brain controls
    - PPO hyperparameters
    - network layer configuration
    - physics and landing settings
    - reward and penalty settings

21. The left panel must allow editing hidden layer count, hidden layer size, and hidden layer activation per layer.

22. Default network settings must be:
    - hidden layer 1: `8`, `relu`
    - hidden layer 2: `40`, `relu`
    - hidden layer 3: `8`, `relu`
    - actor output activation: `tanh`

23. The left panel must allow editing at least these PPO values:
    - target generations
    - games per generation
    - learning rate
    - gamma
    - gae lambda
    - clip range
    - entropy coefficient
    - value coefficient
    - PPO epochs
    - minibatch size
    - max grad norm
    - action std
    - seed

24. The left panel must allow editing at least these physics values:
    - world width / height
    - dt
    - gravity
    - thrust
    - drag
    - wind
    - fuel capacity / burn rate
    - angular acceleration / damping
    - rocket dimensions
    - pad width
    - landing angle threshold
    - max landing velocity x
    - max landing velocity y
    - max steps
    - spawn position and velocity ranges

25. The physics section must include an `Apply Physics` action that updates the live evaluation environment without restarting the whole app.

26. The left panel must allow editing at least these rewards and penalties:
    - landing bonus
    - close to pad bonus
    - progress scale
    - alive bonus
    - center bonus
    - fuel bonus
    - upright bonus
    - delta x penalty
    - delta y penalty
    - step * delta x penalty
    - step * delta y penalty
    - crash penalty
    - offscreen penalty
    - timeout penalty
    - turn penalty
    - throttle penalty
    - step penalty
    - velocity penalty
    - spin penalty

27. The GUI must validate settings before training begins, including cross-field checks such as:
    - `spawn_y_min <= spawn_y_max`
    - positive fuel / thrust / gravity constraints
    - pad width smaller than world width
    - valid landing thresholds
    - supported activation names

28. Validation must distinguish between blocking errors and non-blocking warnings.

29. The game view must clearly visualize:
    - rocket body
    - flame/thrust
    - landing pad
    - trail/history
    - telemetry overlay
    - evaluation/training state

30. During training, the game animation must pause to reduce rendering overhead, while the graph and network panel update after each generation.

31. The graph view must show:
    - landing rate
    - generation best score
    - generation mean score

32. The right panel must render the actor network with all nodes shown.

33. Connections in the network view must visually emphasize stronger weights more than weaker weights.

34. The network view should be efficient enough to handle larger user-defined networks by caching layout and avoiding unnecessary redraw work.

35. The neural-network visualization must update when:
    - a new generation completes
    - the selected brain changes
    - the loaded checkpoint changes
    - the canvas size changes

36. The model input vector must include:
    - delta y from pad
    - delta x from pad
    - angle to pad center
    - sin(angle to pad center)
    - cos(angle to pad center)
    - fuel left
    - steps
    - velocity y
    - velocity x
    - facing angle
    - velocity magnitude
    - angular velocity
    - distance to pad

37. Save/load must persist:
    - model weights
    - config/settings from the left panel
    - training history
    - best metrics
    - observation normalization state
    - checkpoint metadata

38. Checkpoint metadata should include at least:
    - schema version
    - save timestamp
    - generation count
    - total episodes
    - total steps
    - best metrics
    - last generation summary
    - best generation summary

39. The project must include automated tests and smoke checks that verify:
    - environment observation shape and step behavior
    - trainer can complete a short run
    - checkpoint round-trip works
    - config validation catches bad input

40. Minimum validation commands for project completion:
    - `python run.py smoke-test`
    - `python -m unittest discover -s tests`
    - `python run.py headless-train --generations 1 --games 4`
