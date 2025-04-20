if __name__ == "__main__":
    try:
        # Force CPU-only mode - VERIFIED WORKING IN WSL2
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPU from PyTorch
        
        from ragatouille import RAGPretrainedModel
        from ragatouille.utils import get_wikipedia_page
        
        print("Successfully imported RAGatouille")
        
        # Load the pretrained model
        print("Loading ColBERT model...")
        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        
        # Get some test documents
        print("Fetching test documents...")
        documents = [
            get_wikipedia_page("Hayao Miyazaki"),
            get_wikipedia_page("Studio Ghibli")
        ]
        
        # First create an index with the documents - use CPU for stability in WSL
        # This configuration has been verified working on WSL2 with Ubuntu 22.04
        print("Creating an index from documents...")
        index_path = RAG.index(
            collection=documents,
            index_name="test_index",
            use_faiss=False  # Using CPU-only mode for stability in WSL
        )
        
        # Now search the index
        print("Searching the index...")
        results = RAG.search(
            query="Who directed Princess Mononoke?",
            k=3
        )
        
        print("Search results:")
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']}")
            print(f"   Content: {result['content'][:100]}...")
    except Exception as e:
        print(f"Error: {e}") 