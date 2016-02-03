import os

# DO NOT edit the root_dir_name or the configspec_file_name unless 
# you have a very good reason
rootdir = os.path.join(os.path.dirname(__file__), "RegionData/")
configspec = rootdir+"configspec.cfg"

# DO edit the regionspec_file_name to switch between regions.
# You may also need to edit the regional config files to match
# the data locations on your system.
#regionspec = rootdir+"USA.cfg"
#regionspec = rootdir+"Europe_without_Baltic.cfg"
regionspec = rootdir+"Europe_with_Baltic.cfg"
