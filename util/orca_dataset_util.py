# Some utilities to import datasets into orcadatabases

import numpy as np
from datasets.combine import concatenate_datasets
from datasets.load import load_dataset
from orcalib import IntT, OrcaClient, OrcaDatabase, TextT, VectorT
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from util.config import config
from util.torch_orca_utils import DictListToListDict


def initialize_database(db_name: str, index_name: str, embedding_size: int) -> OrcaDatabase:
    """Initialize a new database.

    :param name: The name of the database to initialize.
    """
    print("Initializing database...")
    db = OrcaDatabase(db_name)

    table_name = "llama_table"
    if table_name in OrcaClient.list_tables(db_name):
        print(f"Table {table_name} already exists, dropping it and recreating...")
        db.drop_table(table_name)
    table_handle = db.create_table(
        table_name,
        row_id=IntT.notnull,
        output_embedding=VectorT[embedding_size].notnull,
        text=TextT.notnull,
    )

    # TODO probably want this interface at some point
    # if index_name in db.list_indexes():
    #    raise ValueError(f"Index {index_name} already exists")

    db.create_vector_index(
        table_name=table_name,
        index_name="llama2_vector_index",
        column="output_embedding",
    )
    # db.create_text_index(table_name=table, index_name="llama2_text_index", column="text")
    return db, table_handle


def createDatabaseDataset(model, tokenizer, database_dataset, orca_table, batch_size, max_seq_length):
    print("Collecting embeddings for dataset...")
    rec = {}

    # TODO we might not need a hook for Llama at least; the output
    # has embeddings that may be the same, check it
    def get_final_encoding(module, input, output):
        rec["output_embedding"] = output[0].sum(dim=1).detach().cpu().numpy().tolist()

    hook_ref = model.model.register_forward_hook(get_final_encoding)

    # TODO this is so we eventually don't have to forward pass the trunk
    # when fine tuning the head; we can just use the trunk's output
    # that's been saved via the hook
    memory_dataset = []

    train_dataloader = DataLoader(database_dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(train_dataloader):
        rec = {}
        rec["row_id"] = np.random.randint(0, 100000000, len(batch["text"])).tolist()
        rec["text"] = batch["text"]

        data = tokenizer(
            batch["text"],
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        ).input_ids.to(model.device)

        _ = model(input_ids=data, labels=data)

        unbatched = DictListToListDict(rec)
        for row in unbatched:
            memory_dataset.append(row)
            orca_table.insert(row)

    hook_ref.remove()

    return memory_dataset


def orcaLayerPresidentDataset(hf_augmenting_dataset: str, num_extraneous_sequences: int):
    full_dataset = load_dataset(hf_augmenting_dataset, split="train", token=config.HF_ACCESS_TOKEN)

    ft_ds = concatenate_datasets(
        [
            full_dataset.filter(lambda example: example["document_title"] == "President of the United States"),  # type: ignore
            full_dataset.filter(lambda example: example["document_title"] == "Barack Obama"),  # type: ignore
            full_dataset.select(range(num_extraneous_sequences)),
        ]
    )

    ft_ds = ft_ds.map(
        lambda x: {"text": "Q: " + x["question"] + "? A: " + x["answer"]},
        remove_columns=["question", "answer"],
    )

    return ft_ds


def getDatasetandDatabase(
    hf_augmenting_dataset: str,
    num_extraneous_samples: int,
    embedding_size: int,
    model,
    tokenizer,
    batch_size: int,
    max_seq_length: int,
):
    """Create a database from a dataset.

    :param hf_augmenting_dataset: The name of the dataset to use.
    :param num_extraneous_samples: The number of extraneous samples to add to the dataset.
    """
    dataset = orcaLayerPresidentDataset(hf_augmenting_dataset, num_extraneous_samples)

    db_name = hf_augmenting_dataset + "_database"
    index_name = "llama2_vector_index"
    if True:  # db_name not in OrcaClient.list_databases():
        db, table_handle = initialize_database(db_name, index_name=index_name, embedding_size=embedding_size)
        _ = createDatabaseDataset(model, tokenizer, dataset, table_handle, batch_size, max_seq_length)
    else:
        print("Assuming existing db is correct...")
        db = OrcaDatabase(db_name)

    return dataset, db, index_name
