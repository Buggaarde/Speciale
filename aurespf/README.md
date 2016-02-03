aurespf
========

Central module for the power flow calculations classes and solvers.

As of May 19 this has been merged with Magnus' and Bo's branch:
- Support for 2 runs for capped flows.
- Support for randomised ensembles.

To add this submodule to your repository, just type

git submodule add git@github.com:AUESG/aurespf.git 

git submodule init

and keep it updated by 

git submodule foreach git pull origin master

in theory,

git submodule update

should do the trick, but I havent' been able to get it to work.

You are, of course, also welcome to contribute directly to it, if you find bugs, fixes, or want to try different sink/course allocation paradigms.

I have not included the data or settings folder, I assume each project will have different admitance matrices and datafiles.

This means that you should probably make a wrapper for whatever you are working in to send your parameters to the Nodes object (list of files, file prefix, location, NTC matrix, etc). We've been doing this so far, anyways.

I have taken the convention that numpy exclusive functions are prefaced with np.[].

If you find anything that doesn't work, its probably Gorm's fault. :P

rar(at)imf.au.dk
