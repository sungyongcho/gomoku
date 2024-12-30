#!/bin/sh

# Check if $CONDA_PREFIX is empty
if [ -z "$CONDA_PREFIX" ]; then
	source ~/.zshrc
    conda activate "42AI-$USER"
fi

cd $CONDA_PREFIX/lib

mkdir backup
mv libstd* backup
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./
ln -s libstdc++.so.6 libstdc++.so

## want to use output of ls and get libstdc++.so.6.XX.XX and store it to the variable
libstdc_version=$(ls ./backup | grep -o 'libstdc++.so.6.[0-9]*.[0-9]*')

# Print the version to verify
echo "libstdc++ version: $libstdc_version"

ln -s "libstdc++.so.6" "libstdc++.so.$libstdc_version"
