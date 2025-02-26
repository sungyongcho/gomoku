#!/bin/bash

# List of required packages
REQUIRED_PACKAGES=(
    g++
    cmake
    make
    libwebsockets-dev
    rapidjson-dev
    entr
)

# Function to check if a package is installed
check_package_installed() {
    dpkg -s "$1" &> /dev/null
}

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install missing packages
echo "Checking and installing required packages..."
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if check_package_installed "$pkg"; then
        echo "$pkg is already installed."
    else
        echo "$pkg is not installed. Installing..."
        sudo apt-get install -y "$pkg"
    fi
done

echo "All dependencies are installed."
