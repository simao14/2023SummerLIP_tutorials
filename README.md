# 2023SummerLIP_tutorials

Welcome to this quick guide to help you get started on your summer internship at LIP. This guide will cover how to use the Pauli machines, how to use git and how to use python and uproot for machine, with sprinkles of ROOT and TMVA.

So let's start with the LIP machines, if you already have acess to them and have them set up simply do ssh username@pauli.ncg.ingrid.pt and you are in! I recommend going into your .ssh folder on your computer and creating a file called config (if none already exists) and configure the Pauli machines to avoid having to write so much everytime you need to enter.
If you do something like this then you only need to write ssh lip to access the machines, of course you can change this by changing the name after host. Dont worry about the last parameter that's just to avoid timeout's.

              Host lip
                  HostName pauli.ncg.ingrid.pt
                  User username
                  ServerAliveInterval 60
                                                                            
Secondly let's make sure you have all the packages you need on your machine, to do this we will create a conda environment to be able to install anything we might need. 
You can either check the documentation on how to do it (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or take my word for it and run:

                                                            conda create -c conda-forge -n myenv python root uproot
                                                            
This should give us all the packages we will need excepto pytorch. Conda is a bit stubborn so he will try and install an older version of pytorch if we try and load it from conda-forge so we will need to install it separatly (in general try not to do this, it's usually better to install all the packages of your environment when you create it to avoid future complications).

Now run conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia to install pytorch into your environment (make sure it is activated, if not run conda activate myenv).

Great now we have everything we need.
