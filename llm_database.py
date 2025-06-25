import os
import time
import chromadb
import csv
import pandas as pd

from embedding import SentenceTransformerEmbeddingFunction
from pdf_extraction import extract_text_with_page_numbers, extract_tables_and_captions_with_pdfminer

def clean_csv_lines(directory):
    """
    Function to scan through all CSV files in a directory, cleaning each line by removing
    unwanted newline characters (\n), carriage returns (\r), and other whitespace.

    Args:
        directory (str): Path to the directory containing CSV files.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    for file in files:
        try:
            file_path = os.path.join(directory, file)
            with open(file_path, 'r', newline='', encoding='utf-8') as infile:
                rows = list(csv.reader(infile))
            rows = [[cell.replace('\n', ' ').replace('\r', ' ') for cell in row] for row in rows]
            with open(file_path, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerows(rows)
        except Exception as e:
            print(e)

def remove_duplicate_or_unwanted_header(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    for file in files:
        try:
            file_path = os.path.join(directory, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                print(f"{file} has less than 2 lines. Skipping.")
                continue
            first_line = lines[0].strip()
            second_line = lines[1].strip()
            first_line_elements = first_line.split(",")
            if first_line == second_line or first_line == "0,1":
                print(f"Removing unwanted line from {file}.")
                with open(file_path, 'w') as f:
                    f.writelines(lines[1:])
            elif len(first_line_elements) > 1 and not first_line_elements[1]:
                print(f"Inserting an empty string in {file}.")
                first_line_elements[1] = "-"
                lines[0] = ",".join(first_line_elements) + "\n"
                with open(file_path, 'w') as f:
                    f.writelines(lines)
            else:
                print(f"No duplicate or unwanted header in {file}.")
        except Exception as e:
            print(e)

def save_dataframes(dataframes, starting_number, directory=None, file_prefix='df_'):
    if not isinstance(dataframes, list) or not all(isinstance(df, pd.DataFrame) for df in dataframes):
        raise ValueError("dataframes must be a list of pandas DataFrames")
    if not isinstance(starting_number, int):
        raise ValueError("starting_number must be an integer")
    if directory is None:
        directory = os.path.join(".", "tables")
    os.makedirs(directory, exist_ok=True)
    filepaths = []
    for i, df in enumerate(dataframes):
        filename = f"{file_prefix}{starting_number + i}.csv"
        filepath = os.path.join(directory, filename)
        df.to_csv(filepath, index=False)
        filepaths.append(filepath)
    return filepaths

def pdf_dir2database(pdf_dir, database_collection, config):
    start = time.time()
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            print('Adding document: {}'.format(file))
            try:
                used_ids = len(database_collection.get()['ids'])
            except:
                used_ids = 0
            pdf_path = os.path.join(pdf_dir, file)
            tables, dfs = extract_tables_and_captions_with_pdfminer(pdf_path, config)
            documents, page_nbs = extract_text_with_page_numbers(pdf_path, config)
            documents += tables
            table_paths = save_dataframes(dfs, used_ids+len(documents)-len(tables), directory=config["dataframe_loc"])
            meta_data_pages = [{"page": pages[0], "paragraph_type": "text", "origin":file} for pages in (page_nbs)]
            meta_data_tables = [{"paragraph_type": "table", "dataframe": table_paths[i], "origin":file} for i, table in enumerate(tables)]
            meta_data = meta_data_pages + meta_data_tables
            database_collection.add(documents=documents, ids=[f"id{i}" for i in range(used_ids, used_ids+len(documents))],
                               metadatas=meta_data)
    print("Done")
    end = time.time()
    diff = end-start
    print("Elapsed time:", diff)

def main():
    CHROMA_DATA_PATH = "UroBot_database"
    COLLECTION_NAME = "UroBot_v1.0"
    config = {"separator": " | ",
              "chunk_threshold": 800,
              "markdown_tables": True,
              "filter_toc_refs": False,
              "dataframe_loc": "tables"}
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    embedding_func = SentenceTransformerEmbeddingFunction()
    embedding_func.initialize_model()
    try:
        collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_func, metadata={"hnsw:space": "cosine"})
        print("New collection created!")
    except:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)
    pdf_dir = "resources"
    pdf_dir2database(pdf_dir, collection, config=config)
    remove_duplicate_or_unwanted_header(config["dataframe_loc"])
    clean_csv_lines(config["dataframe_loc"])

if __name__ == "__main__":
    main()
