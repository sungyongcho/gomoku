#!/bin/bash

# Determine whether to use sudo: if not root, then use sudo
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

# Check for dependencies.txt in current directory, fallback to /tmp/
if [ -f "./dependencies.txt" ]; then
    DEPENDENCIES_FILE="./dependencies.txt"
elif [ -f "/tmp/dependencies.txt" ]; then
    DEPENDENCIES_FILE="/tmp/dependencies.txt"
else
    echo "Error: dependencies.txt not found!"
    exit 1
fi

# Read dependencies from file
REQUIRED_PACKAGES=($(cat "$DEPENDENCIES_FILE"))

# Function to check if a package is installed
check_package_installed() {
    dpkg -s "$1" &> /dev/null
}

echo "Updating package list..."
$SUDO apt-get update

# Install make first if it's missing
if ! check_package_installed "make"; then
    echo "make is not installed. Installing make first..."
    $SUDO apt-get install -y make
else
    echo "make is already installed."
fi

echo "Checking and installing remaining packages..."
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if check_package_installed "$pkg"; then
        echo "$pkg is already installed."
    else
        echo "$pkg is not installed. Installing..."
        $SUDO apt-get install -y "$pkg"
    fi
done

echo "All dependencies are installed."
