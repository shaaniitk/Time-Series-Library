#!/bin/bash
# Script to load environment variables from .env file into VS Code

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    exit 1
fi

# Create settings.json directory if it doesn't exist
mkdir -p ~/.vscode

# Check if settings.json exists
SETTINGS_FILE=~/.vscode/settings.json
if [ ! -f "$SETTINGS_FILE" ]; then
    echo "{}" > "$SETTINGS_FILE"
fi

# Read .env file and add each variable to VS Code settings
echo "Loading environment variables into VS Code settings..."

# Start with an empty object for terminal.integrated.env.osx
ENV_JSON="{"

# Read each line from .env file
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^# ]]; then
        continue
    fi
    
    # Extract key and value
    key=$(echo "$line" | cut -d= -f1)
    value=$(echo "$line" | cut -d= -f2-)
    
    # Add to JSON
    ENV_JSON+="\"$key\": \"$value\","
done < .env

# Remove trailing comma and close object
ENV_JSON="${ENV_JSON%,}}"

# Update settings.json with jq if available, otherwise use a simple approach
if command -v jq &> /dev/null; then
    # Use jq to update settings.json
    TEMP_FILE=$(mktemp)
    jq --argjson env "$ENV_JSON" '.["terminal.integrated.env.osx"] = $env' "$SETTINGS_FILE" > "$TEMP_FILE"
    mv "$TEMP_FILE" "$SETTINGS_FILE"
else
    # Simple approach without jq
    SETTINGS_CONTENT=$(cat "$SETTINGS_FILE")
    if [[ "$SETTINGS_CONTENT" == "{}" ]]; then
        echo "{ \"terminal.integrated.env.osx\": $ENV_JSON }" > "$SETTINGS_FILE"
    else
        # This is a simplistic approach and might not work for complex settings.json files
        SETTINGS_CONTENT="${SETTINGS_CONTENT%\}}"
        if [[ "$SETTINGS_CONTENT" != *[,] ]]; then
            SETTINGS_CONTENT="$SETTINGS_CONTENT,"
        fi
        echo "$SETTINGS_CONTENT \"terminal.integrated.env.osx\": $ENV_JSON }" > "$SETTINGS_FILE"
    fi
fi

echo "Environment variables loaded into VS Code settings."
echo "Please restart VS Code for changes to take effect."