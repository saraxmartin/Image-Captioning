# Image-Captioning
Image Captioning challenge

## Files structure
* Data

* results


* utils

    - **dataset.py**: creation of the dataset for both approaches

* config.py

    Specify the configuration of the model used

* main.py

    Main file, where all the project would be executed



## How To Use
To clone and run this project, you will need Git and Conda installed on your computer.
An alternative of conda is have installed the packages on the environment.yml file.

If you have Conda installed, from your command line write the following instructions to create an environment with all the dependencies and then activate it:

```bash
# Clone this repository
$ git clone https://github.com/saraxmartin/Image-Captioning

# Go into the repository
$ cd Image-Captioning

# Creating the environment with the provided environment.yml file
$ conda env create --file environment.yml

# Activating the environment
$ conda activate image_captioning

# For running the project in CPU
# running the main of the classification approach
py main.py

```
For running another file, you just need to change the file of above and put the one you want: i.e.: py test_c.py



### Problems activating environment
If you have trouble activating the environment and the error that shows up says something like:

-  Your shell has not been properly configured to use 'conda activate'

Do this instructions in the terminal:
```bash
# To know which type of shell you have 
$ echo $SHELL

# IF the output of the command above is for example: /bin/bash  

# To initialize the shell 
$ conda init bash

# To activate the environment
$ conda activate image_captioning
```

If you are in Visual Studio Code, you would have to select the interpreter manually. To select the interpreter of the environment, press 'Ctrl+Shift+P', search 'Python: Select interpreter' and click on the environment interpreter.
If you can't see the interpreter, in the terminal write
```
which python
```
Copy the path that output. Press again 'Ctrl+Shift+P', press in 'Enter interpreter path' and paste the path you have copied before.



In case that you can't see all the branches, run the command:
```bash
git fetch --all
```

## Contributors
Sara Martín Núñez -- Sara.MartinNu@autonoma.cat
Iván Martín  Campoy -- ivan.martinca@autonoma.cat
Lara Rodríguez Cuenca -- Lara.RodriguezC@autonoma.cat
Aina Navarro Rafols -- Aina.NavarroR@autonoma.cat

Vision and Learning
Degree in Artificial Intelligence
UAB, 2024-25

