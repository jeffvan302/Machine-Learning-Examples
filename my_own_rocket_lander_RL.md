# Provide the following prompt to a coding LLM and it should give you your own game with a neural net to train.

```
Create a Python game with a Deep Reinforcement Learning capable tool for a self implemented Rocket Landing game by completing the following requirements.
1) The game, neural net and deep RL should be in a single python scripts file called run.py
2) Any required modules for the script to function should be added to a requirements.txt such that we can run pip install -r requirements.txt
3) Considerations should be taken that the code will be tested on Python 3.12 and preferable pytorch should be used for the Deep RL PPO learning and models.
4) Implement a full gui interface for the game and training functions.
5) There should be a left scrollable panel that will allow us to configure the game and neural net with a ppo trainer and ability to define the hidden layers and their activation functions.
6) The left panel should allow us to configure the number of games per generation. The Physics of the game.  The rewards and penalties for the model.
7) The rocket position should be random and in a dramatic way.
8) On the right panel their should be a graphical representation of the neural net and the "connections" between the nodes.  Make this look interesting and useful. 
9) The center section will have a top and bottom area.  Place the rocket game at the top and a graph area at the bottom.
10) The game area should have a nice presentation so it is clear as to what is happening.  If a rocket.png file is placed in the same folder is the run.py then use this image as the rocket image.
11) The graph should indicate the landing rate, the best score of the generation, and the mean score so we can track the progress.
12) To ensure faster training: While training stop the game display update and only update the graph and neural net display area after each generation.  Ensure that the buttons on the left panel works while training since we might want to trigger a stop during a generation training session.
13) Allow pausing a training (finish the generation that is training first), so we can do some evaluation.
14) During pauses or after training it would be helpful to have the system continually play the rocket game with the active (last) brain to see how well it does.
15) Since training can take many output results we should keep track of the "best" brain so far by using the landing rate and the mean score as a marker for "best" brain.  During pausing and/or after training allow us to toggle between best brain so far or current brain.
16)  Allow a save and load best brain so we can save our best brain so far.  With it save the settings on the left panel so it is easier to see what values were used to train the model.
17) It would be great to make the gui really sparkle to show off what can be done with python coding using your abilities!
```

## *If you are not familiar with Python then you can follow the next step.*
The LLM should provide you with two files.  One named run.py and the other is requirements.txt
The actual code will be in the run.py.  The requirements.txt has instructions as to the modules the run.py will need.  In this repository on GitHub you will find a launch.exe.  
Place the ***run.py*** the ***requirements.txt*** and the ***launch.exe*** in a folder on their own.  Then use the launch.exe to start the run.py.
**launch.exe**
It is a self contained executable that will examine the requirements.txt and then download python 3.12 in a standalone configuration.  It will set it up on the runtime folder inside the folder you placed it in.  It next will see which graphics card driver you are using and install the PyTorch library that best matches that graphics card driver.  After that it will install the requirements.txt modules.
Once ready the launch.exe will start the run.py script.
Everytime you use the launch.exe it will make sure the python standalone environment is in place and that the Pytorch is up to date.  It will take a moment but since it is just an update check on subsequent starts it won't take too long.