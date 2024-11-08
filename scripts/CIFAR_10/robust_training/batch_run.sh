current_script=$(basename "$0")

for script in ./*.sh; do
    if [[ $(basename "$script") != "$current_script" ]]; then
        echo "Running $script..."
        bash "$script"
    fi
done