#!/usr/bin/env python3
import json
import os
import platform
import sys

def generate_mcp_config():
    """Generate platform-specific MCP configuration."""
    
    # Detect operating system
    is_windows = platform.system() == "Windows"
    
    # Base configuration
    config = {
        "servers": {
            "github": {
                "type": "http",
                "url": "https://api.githubcopilot.com/mcp/"
            },
            "desktopcomandeer": {
                "type": "stdio",
                "command": "npx",
                "args": [
                    "@wonderwhy-er/desktop-commander@latest"
                ]
            },
            "sequentialthinking": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-sequential-thinking"
                ],
                "type": "stdio"
            },
            "memory": {
                "type": "stdio",
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-memory@latest"
                ],
                "env": {},
                "autoApprove": [
                    "memory_bank_read",
                    "memory_bank_write",
                    "memory_bank_update",
                    "list_projects",
                    "list_project_files"
                ],
                "gallery": True
            },
            "memento": {
                "command": "npx",
                "args": [
                    "-y",
                    "@gannonh/memento-mcp"
                ],
                "env": {
                    "MEMORY_STORAGE_TYPE": "neo4j",
                    "NEO4J_URI": "bolt://127.0.0.1:7687",
                    "NEO4J_USERNAME": "neo4j",
                    "NEO4J_PASSWORD": "memento_password",
                    "NEO4J_DATABASE": "neo4j",
                    "NEO4J_VECTOR_INDEX": "entity_embeddings",
                    "NEO4J_VECTOR_DIMENSIONS": "1536",
                    "NEO4J_SIMILARITY_FUNCTION": "cosine",
                    "OPENAI_API_KEY": "your-openai-api-key",
                    "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small"
                }
            }
        },
        "inputs": []
    }
    
    # Platform-specific configurations
    if is_windows:
        # Windows paths and configurations
        base_path = "D:\\workspace"
        uv_command = "C:\\Users\\mishr\\.local\\bin\\uv.exe"
        
        config["servers"]["memory"]["env"]["MEMORY_FILE_PATH"] = f"{base_path}\\memory_banks"
        
        config["servers"]["serena"] = {
            "command": uv_command,
            "args": [
                "--directory",
                f"{base_path}\\serena",
                "run",
                "serena-mcp-server"
            ]
        }
        
        config["servers"]["zen"] = {
            "command": uv_command,
            "args": [
                "--directory",
                f"{base_path}\\zen-mcp-server",
                "run",
                "server.py"
            ],
            "env": {
                "GEMINI_API_KEY": "${env:GEMINI_API_KEY}"
            }
        }
    else:
        # macOS/Linux paths and configurations
        base_path = "/Users/shantanumisra/workspace/MCP"
        uv_command = "uv"
        
        config["servers"]["memory"]["env"]["MEMORY_FILE_PATH"] = f"{base_path}/serena/memory.json"
        
        config["servers"]["serena"] = {
            "command": uv_command,
            "args": [
                "--directory",
                f"{base_path}/serena",
                "run",
                "serena-mcp-server"
            ]
        }
        
        config["servers"]["zen"] = {
            "command": uv_command,
            "args": [
                "--directory",
                f"{base_path}/zen-mcp-server",
                "run",
                "server.py"
            ],
            "env": {
                "GEMINI_API_KEY": "${env:GEMINI_API_KEY}"
            }
        }
    
    # Ensure .vscode directory exists
    os.makedirs(".vscode", exist_ok=True)
    
    # Write the configuration to mcp.json
    with open(".vscode/mcp.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Generated platform-specific MCP configuration for {platform.system()}")

if __name__ == "__main__":
    generate_mcp_config()