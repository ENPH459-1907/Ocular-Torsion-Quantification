# Ocular Torsion Quantification
Python software package to quantify ocular torsion from video recordings.

Documentation at: https://enph1759.github.io/ota/

## Upcoming Changes
* Executables for GUI Mac and Windows
* Better python packaging
* Improving 2D Correlation function
* Better Error Handling
* Documentation
	* Quickstart Guides
	* Examples

## Installing

Install git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

Open terminal (or git bash) and type *git clone https://github.com/enph1759/ota.git*

## Using Git

Checkout out the git cheat sheet for helpful commands: https://github.com/enph1759/ota/blob/master/git-cheat-sheet.txt


### Branches

__IMPORTANT__: please do not work out of the develop and master branch unless you know what you're doing. Create your own branch and then merge into master.

__master__ This is the main branch of the repo, code owners must accept changes to this branch. Anyone who uses the app for data analysis will use the version on the master branch.

__develop__ This branch is the where the "next release" of the code lives. By this we mean all experimental or new changes will go here. Once all the changes are verified as working, this branch will be merged into master.

__gh-pages__: this branch is for the documentation, basically don't touch it unless you updating the documentation.

### Creating Your Own Branch

Create your own branch if you want to work on new features or play around.

On the __develop__ branch, type *git checkout -B <your_branch_name>*. Keep branches to small features and keep names descriptive. ex. *hotfix/xcorr2d*.
