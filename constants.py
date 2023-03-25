import os


server_port = 7878

# Dataset keys
preprocessed_col_title = 'new_preprocessed_text'

# Directories
ui_dir = os.path.join(".","ui")
models_dir = "models"
dataset_dir = "dataset"

# Paths to files
processed_dataset_path = os.path.join(dataset_dir, "processed_dataset.csv")
ui = {
    "home":  "home.html",
    "train":  "train.html",
    "test":  "test.html",
    "preprocess":  "preprocess.html"
}

# Training Params
training_params = {
    "min_count":0,
    "embedding_model":"doc2vec",
    "ngram_vocab":True,
    "split_documents": True,
    "document_chunker": "sequential",
    "chunk_length": 5,
    "max_num_chunks": 2,
    "hdbscan_args": {
        "min_cluster_size": 3, 
        "metric": "euclidean", 
        "cluster_selection_method": "leaf"
    },
    "speed": "deep-learn",
    "workers": 80,
    "verbose": True
}


