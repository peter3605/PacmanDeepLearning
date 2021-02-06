# PacmanDeepLearning
he bulk of where we will be working is in multiAgents.py, go check that out after reading this and read
the comments that I have included. The way the program works is you can execute the pacman.py file with
your specified arguments. Flags you can pass are under pacman.py readCommand(args) function. These
arguments will control things like, what level you use, number of ghosts, and what agent we use (though
we are only creating one). Some basic flags are -p (specify which agent to use), -l (select which level
to use), -k (number of ghosts, only works on default level with 0-2 ghosts). Underneath are some examples
of what you can run as commands:

'python3 pacman.py' # Lets you play pacman for yourself on the standard default level (pretty slow)

'python3 pacman.py -p ReinforcementAgent' # Run our agent on default level

'python3 pacman.py -p ReinforcementAgent -l testClassic' # Run our agent on a tiny level

'python3 pacman.py -p ReinforcementAgent -l testClassic -k 0' # Default level with 0 ghosts

'python3 pacman.py --frameTime 0 -p ReinforcementAgent' # Default level with faster frame rate

If you look into pacman.py you can see functionality to other commands but those should not be
necassary if we are using the same model structure that neal found on github.

*** Level names can be found in the layouts folder ***
*** We can create our own maps by using the same format as the provided layouts ***
*** I made a dummy layour to show how you can change the levels run with flag -l dummy ***
*** Be careful though! Ghosts are not allowed to 'Stop' so need a least 2 cells to move ***

The biggest thing in this is that the program runs in a way where all of the computation is done first
and then the path is executed. This is fine since the only differnce is instead of seeing pacman develop
in real time, we will see the actions he chose to make after everything was decided. And then from there
run the program again with the next iteration.
