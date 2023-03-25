import os
import copy
import pandas as pd
from io import StringIO
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from arabic_reshaper import reshape
from top2vec import Top2Vec

from constants import *

ar_stemmer = SnowballStemmer("arabic")
ar_stop_words = set(stopwords.words("arabic"))

def path_exists(p): return os.path.exists(p)

def check_dir (d):
    if not path_exists(d):
        os.makedirs(d)

def validate_req_dataset(req_files):
    if "dataset" not in req_files:
        return {"is_valid": False, "msg": "No dataset file uploaded", "file": None}
    dataset = req_files["dataset"]
    f_name = dataset.filename.lower()
    if not f_name:
        return {"is_valid": False, "msg": "No CSV file was Uploaded", "file": None}
    if not f_name.endswith(".csv"):
        return {"is_valid": False, "msg": "Uploaded file is not a CSV file", "file": None}
    return {"is_valid": True, "msg": "", "file": dataset}

def preprocess_single_text(text):
  try:
    # Tokenize the line into individual words
    words = word_tokenize(text)

    # Remove stop words and perform stemming on the words
    stemmed_words = [ar_stemmer.stem(word) for word in words if word not in ar_stop_words]

    # Normalize the stemmed words
    normalized_words = [word if not word.isdigit() else str(int(word)) for word in stemmed_words]

    # Join the normalized words back into a string
    normalized_text = " ".join(normalized_words)

    # Reshape the normalized text to fix any display issues
    reshaped_text = reshape(normalized_text)
    return reshaped_text

  except:
    print("Error during sigle text preprocessing:",normalized_text)
    return None

def preprocess_dataset(dataset, col_key):
    try:
        # Get contents of the uploaded file
        file_str = dataset.read().decode('utf-8')

        # Read dataset and remove N/A values
        df = pd.read_csv(StringIO(file_str))
        df = df.dropna()


        if col_key not in df.columns:
            print("Column not found in dataset: ", col_key)
            return None

        # Add a new column to the dataset with processed texts
        df[preprocessed_col_title] = df[col_key].apply(preprocess_single_text)

        # Save preprocessed dataset to a new file
        df.to_csv(processed_dataset_path, index=True)

        print("Dataset saved to file:", processed_dataset_path)

    except Exception as e:
        print('Error during dataset preprocessing:', e)

    return None

def train_new_model(model_name):
    try:
        if not path_exists(processed_dataset_path):
            print("Training didn't start !!!")
            print("Processed dataset doesn't exist. Please check")
            return
        df = pd.read_csv(processed_dataset_path)
        if preprocessed_col_title not in df.columns:
            print("Training didn't start !!!")
            print("Dataset is not processed or saved successfully. Please check")
            return None
        document_index = dict(enumerate(df[preprocessed_col_title].values.tolist()))
        documents = list(document_index.values())
        training_args = copy.deepcopy(training_params)
        training_args['documents'] = documents
        print("IMPORTANT!!!! Training process will start now, and it will run in the background. you can still perform request from this server BUT BE CAREFUL: DON\'T START A NEW TRAINING PROCESS!!!")
        model = Top2Vec(**training_args)
        print("Training is Finished Successfully")
        new_model_path = os.path.join(models_dir,model_name)
        model.save(new_model_path)
        print("New Model is saves Successfully in", new_model_path)
        return
    except Exception as e:
        print('Error during dataset preprocessing:', e)
        return


def test_model(data):
    res = {"message":"", "docs":[]}
    print(data)
    if not data["model_name"]:
        res["message"] = "Failed! Model name was not specified"
        return res
    if not data["user_query"]:
        res["message"] = "Failed! Query was not specified"
        return res
    if not path_exists(processed_dataset_path):
        res["message"] = "Failed! Original Document Dataset is Missing !!!"
        return res
    test_model_path = os.path.join(models_dir,data["model_name"])
    if not path_exists(test_model_path):
        res["message"] = "Failed! Testing model is Missing !!!"
        return res

    docs = []
    df = pd.read_csv(processed_dataset_path)
    document_index = dict(enumerate(df[preprocessed_col_title].values.tolist()))
    preprocessed_query = preprocess_single_text(data["user_query"])
    model = Top2Vec.load(test_model_path)
    documents, doc_scores, doc_indices = model.query_documents(query=str(preprocessed_query), ef=10000, num_docs=5)

    for idx, score in zip(doc_indices, doc_scores):
        index_title = [k for k, v in document_index.items() if v == model.documents[idx]][0]
        title = df.loc[index_title, 'title']
        details = df.loc[index_title, 'details']
        docs.append({"title":title, "score":f"{round(score * 100, 2)}%", "details":details})

    res["message"] = "Success"
    res["docs"] = docs
    return res


