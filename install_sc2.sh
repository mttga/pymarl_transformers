#!/bin/bash
# Install SC2 and add the custom maps

cd "$HOME"
export SC2PATH="$HOME/StarCraftII"
echo 'SC2PATH is set to '$SC2PATH

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
        unzip -P iagreetotheeula SC2.4.10.zip
        rm -rf SC2.4.10.zip$
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip

# if you want to add addtional maps, put them in the folder smac_maps and uncomment these two lines
# you would also need to add COPY ./smac_maps ./smac_maps in the DOckerfile
#smac_maps=$(pwd)/smac_maps
#cp -r $smac_maps/* ./SMAC_Maps 
mv SMAC_Maps $MAP_DIR
rm -rf SMAC_Maps.zip


echo 'StarCraft II and SMAC are installed.'