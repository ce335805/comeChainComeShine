# comeChainComeShine
Repo hosting code for 'Quantum Floquet engineering with an exactly solvable tight-binding chain in a cavity' project.

On branch master current development is happening also beyond the scope of the associated paper.

The branch PaperReference is intended to make the results from the associated paper easily reproducible.
Here, the main file of the program is 1Dchain. 
To execute first create a folder
```
$mkdir savedPlots
```
and then execute the program via

```
$python3 1Dchain.py
```

The program will then produce all plots of the paper (this can take a while but should be doable in <1h on a laptop) and save them to the directory `savedPlots`.

To execute the program only standard (openly available) packages are required.
