conda create -n topoinfo python=3.12 --yes
conda activate topoinfo

# install packages
poetry install 
# Install the Python package
cd topotree/
poetry run python setup.py build_ext --inplace
cd ../benchmark

while true; do
    # Prompt the user to enter yes or no
    read -p "Do you want to run benchmarks ? (yes/no): " response
    
    # Convert the response to lowercase
    response=$(echo "$response" | tr '[:upper:]' '[:lower:]')

    # Check the user's response
    if [[ "$response" == "yes" || "$response" == "no" ]]; then
        break  # Exit loop if valid response
    else
        echo "Invalid input. Please enter 'yes' or 'no'."
    fi
done

# Perform action based on user's response
if [[ "$response" == "yes" ]]; then
    echo "Starting benchmarks [This will take a very long time]..."
    for file in *.py; do python "$file"; done

else
    echo "Install Done."
fi
