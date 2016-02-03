regions is a module designed to cover data handling, in particular to be able to build 
Nodes() instances for different world regions automatically using config files (one for each
major region).

At the moment, to switch between different regions, it is necessary to edit the defaults
file (regions/defaults.py) manually. More elegant ideas are welcome.

The module is based on the classes.py, tools.py, and storage.py files in the aurespf module.
It is suggested that the entire data handling process will be moved from aurespf to regions,
such that aurespf really only deals with the power flow on the network and not with the 
reading and organization of timeseries and other input data into Nodes objects.

regions is written and maintained by Sarah (becker@fias.uni-frankfurt.de). Please do not merge
into master without conferring with the maintainer!
