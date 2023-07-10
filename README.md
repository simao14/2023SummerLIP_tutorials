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

Now run 

    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 

to install pytorch into your environment (make sure it is activated, if not run conda activate myenv).

Great now we have everything we need. If you have any doubts about how to work with your conda environment go check the official documentation, it should be straighforward.

If you haven't already go and fork this repository (it will create a copy but of your ownership that's still connected to this one). The option should be at the right of the repository's name.

Now you have a few options on how to proceed, you can either clone this repository (the forked one that's yours) to your LIP machine and work entirely there, or you can clone it on your personal computer to edit the codes there and then send them to Pauli. You can even clone it on both and push the changes from your computer and pull them on pauli.

Whatever option you choose you simply need to run git clone path, where path is the one that you get by clicking on the green button labeled code.

Now that we are all set up let's see how we can actually work. 

If you are working on the LIP machines solely then you can open your files with vim. Just type vim filename and you can go straight into editing. Vim is a very used and famous text editor, beloved by many people. Alas I am not one of those people. If you want to learn how to use Vim to its fullest potential I suggest you search for another guide. The only Vim knowledge I can impart unto you is press I to edit, esc to leave editing mode and type :q to exit and :wq to save and exit.

If you are like me and prefer to work on your own computer then you have two options, use git or scp. scp (or special copy paste) is a very simple function you will want to install on your computer if not already installed, and it's very easy to work with, simply write scp -r uploadpath downloadpath and you will upload the first file into the directory specified on the second argument, the -r argument is simply to be able to transfer or upload entire folders. If you want to download or upload anything to you pauli machines you simply do 
    
    scp -r username@pauli.ncg.ingrid.pt:uploadpath downloadpath #to download
    or
    scp -r uploadpath username@pauli.ncg.ingrid.pt:downloadpath #to upload

If there is any file in the receiving end with the same name as the one you are downloading then the existing one will be overwritten so be careful not to erase your own work by mistake.

Finally let's talk a bit more about git and how to upload files with it. Updating your repository is very simple, first make sure you are in the directory where you cloned a git repository (you can use git status to make sure you are in the right place and to see what files you have differing from the ones on git), then to push any changes you might have done to the files simly do the following commands:

    git add * #here * stands for every file that was changed, if you want to only push specific ones you can add them by specifying their paths
    git commit -m "write a message here about what this commit is changing. This will help you find a version of the code you need if you ever have to search through your history"
    git push

And there you go, your repository is now updated.
If you have some changes in git you want to put on your clone then simply run 

    git pull

Note that if you have something to push you might need to pull first if your cloned repository isn't up to date.
I could write a whole book here about useful git commands but it's much simpler to search for it yourself, git documentation is usually extremely detailed.
I will however leave you here with two commands you hopefully will never need, and realistically will find life savers (I for sure do at least).
When git pushes or pulls anything it tries to merge the two versions into a single one and doesn´t really like to overwritte anything, but this becomes a problem when for example two people are using the same repository and change the same lines of code. If one of them updates git first then the second one will find himself in a very big problem. Since git can't decide which version to maintain it will refuse to push or pull any changes ( and manually fixing it is not always a good solution). There is a very easy (and dangerous) solution to this, you simply force git to overwritte one of the versions.

You can do this with one of this commands:

    git push --force # will force your push and make the git repository the same as what you have. NEVER run this unless you are 110% sure you have the most correct and updated version of the code
                     # as it will erase any work other people have done that you weren't able to pull yet.
    
    git reset --hard origin/main #this one is a bit safer since the only thing at risk is your own work. It will simply force the pull and erase any changes you have.

If you and your colleague both have important changes in the files and you find this problem, well, one of you will have to bite the bullet and update onde of the codes by hand and force push it. You can also try and resolve the dependencies via github itself.

You can now go try and solve the tutorials on here!  You can start with either the root or uproot tutorial and after these two go to the pytorch one.

If you dont want to run the notebook on your computer here is a link to a google colab to run it - https://colab.research.google.com/drive/1G0_sq-lG4AcBMic7_olvzc0pE8U9_jU9?usp=sharing

For more information about this topics you can check out this sources:

-uproot intro: https://masonproffitt.github.io/uproot-tutorial/

-uproot bit more advanced but has less examples: https://uproot.readthedocs.io/en/latest/basic.html

-pytorch+TMVA main page: https://root.cern/blog/tmva-pytorch/

-pytorch+TMVA tutorial:https://anirudhdagar.ml/gsoc/tmva/pytorch/root/2020/08/21/TMVA-PyTorch-Tutorial.html

-pytorch + TMVA examples:https://github.com/root-project/root/tree/master/tutorials/tmva
