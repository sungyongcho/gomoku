#!/bin/zsh

if [ -d "/goinfre" ]; then
  # Take action if $DIR exists. #
  echo "setting MYPATH for school\n"
  export MYPATH="/goinfre/$USER/miniconda3"
else
  echo "setting MYPATH for mac\n"
  export MYPATH="/Users/$USER/Documents/miniconda3/"
fi

if [[ "$(uname)" == "Darwin" ]]; then
	# For MAC
	curl -LO "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
	sh Miniconda3-latest-MacOSX-arm64.sh -b -p $MYPATH
elif [[ "$(uname)" == "Linux" ]]; then
	curl -LO "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
	sh Miniconda3-latest-Linux-x86_64.sh -b -p $MYPATH
fi

# For zsh
$MYPATH/bin/conda init zsh
$MYPATH/bin/conda config --set auto_activate_base false
source ~/.zshrc

# For bash
# $MYPATH/bin/conda init bash
# $MYPATH/bin/conda config --set auto_activate_base false
# source ~/.bash_profile


if [[ "$USER" == "sucho" ]]; then
	conda create --name gomoku python=3.13 -y
else
	conda create --name gomoku python=3.13 -y
fi
