#!/bin/bash

# Setup script for RAG Chatbot

echo "=================================="
echo "RAG Chatbot Setup"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python $(python3 --version) found"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create config file if it doesn't exist
if [ ! -f config.yaml ]; then
    echo ""
    echo "Creating config.yaml from template..."
    cp config.template.yaml config.yaml
    echo "✓ config.yaml created"
    echo ""
    echo "⚠️  IMPORTANT: Edit config.yaml and add your OpenAI API key!"
    echo ""
else
    echo ""
    echo "✓ config.yaml already exists"
fi

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Edit config.yaml and add your OpenAI API key"
echo ""
echo "3. Add documents to a 'data' directory (PDF or TXT files)"
echo ""
echo "4. Run the chatbot:"
echo "   python main.py"
echo ""
echo "Or index documents separately:"
echo "   python index_documents.py ./data"
echo ""
echo "For examples, run:"
echo "   python examples.py"
echo ""
