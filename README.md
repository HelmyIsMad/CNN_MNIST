# Quick notes
1. The ```model.h5``` file is the model, the program will first check if the file exists in its directory, if nor its gonna train it and generate the model.h5 file
2. To be able to use it you have to uncomment one of 2 functions  ```line 95: test(index)``` or ```line 96: testScreen()```. test(index) will choose a random index from the testSet and test on it while testScreen() will capture a part of the screen and test on it (part of screen is configurable in line 121)
   
