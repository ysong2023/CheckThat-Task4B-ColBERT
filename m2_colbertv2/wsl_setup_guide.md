# WSL2 Setup Guide for RAGatouille with ColBERT

This guide will help you set up Windows Subsystem for Linux (WSL2) with Ubuntu 22.04 and install RAGatouille for using ColBERT models. This setup has been successfully tested and verified working in CPU mode.

## 1. Install WSL2

First, you need to install WSL2 on your Windows system:

1. Open PowerShell as Administrator
2. Run the following command:
   ```powershell
   wsl --install
   ```
3. If you already have WSL1 installed, upgrade to WSL2:
   ```powershell
   wsl --set-version <DistroName> 2
   ```
4. To check your WSL installations and versions:
   ```powershell
   wsl -l -v
   ```

## 2. Install Ubuntu 22.04

If Ubuntu 22.04 isn't already installed:

1. Install Ubuntu 22.04 with:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
2. Set up your username and password when prompted
3. Verify installation with:
   ```powershell
   wsl -l -v
   ```

## 3. Set Up Python Environment

1. Update your Ubuntu system:
   ```bash
   sudo apt update && sudo apt install -y python3 python3-pip python3-dev
   ```

2. Create a virtual environment:
   ```bash
   pip3 install virtualenv && python3 -m virtualenv ragatouille_env
   ```

3. Install RAGatouille and dependencies:
   ```bash
   source ragatouille_env/bin/activate && pip install -r requirements.txt
   ```

## 4. Using RAGatouille (CPU Mode - Verified Working)

After extensive testing, we found that **CPU-only mode** is the most reliable approach for using RAGatouille in WSL2. While GPU mode is theoretically possible, it involves complex CUDA setup issues that are difficult to resolve in the WSL2 environment.

To run your code with CPU-only mode (verified working):

1. Create a Python script with GPU disabled:
   ```python
   if __name__ == "__main__":
       # Force CPU-only mode
       import os
       os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPU from PyTorch
       
       from ragatouille import RAGPretrainedModel
       from ragatouille.utils import get_wikipedia_page
       
       # Load the pretrained model
       RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
       
       # Get some test documents
       documents = [
           get_wikipedia_page("Hayao Miyazaki"),
           get_wikipedia_page("Studio Ghibli")
       ]
       
       # Create an index with CPU mode
       index_path = RAG.index(
           collection=documents,
           index_name="test_index",
           use_faiss=False  # Using CPU-only mode for stability in WSL
       )
       
       # Search the index
       results = RAG.search(
           query="Who directed Princess Mononoke?",
           k=3
       )
       
       # Print results
       for i, result in enumerate(results):
           print(f"{i+1}. Score: {result['score']}")
           print(f"   Content: {result['content'][:100]}...")
   ```

2. Run your script:
   ```bash
   source ragatouille_env/bin/activate && python3 your_script.py
   ```

## Notes

- The WSL file system is accessible at `\\wsl$\Ubuntu-22.04\` in File Explorer
- The requirements.txt file contains all necessary dependencies with specific versions
- RAGatouille indexing functions create files in the .ragatouille directory
- CPU-only mode works reliably but will be slower than GPU for large datasets
- For production workloads with large datasets, consider using a native Linux installation or a cloud solution with proper CUDA support 