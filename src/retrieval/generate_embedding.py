import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE


def carregar_modelo_embedding():
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

    model.eval()

    return tokenizer, model


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * mask, 1) / torch.clamp(
        mask.sum(1),
        min=1e-9
    )


def gerar_embeddings(chunks, tokenizer, model):
    all_embeddings = []

    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
        textos = [chunk.page_content for chunk in batch_chunks]

        encoded_input = tokenizer(
            textos,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            model_output = model(**encoded_input)

        embeddings = mean_pooling(
            model_output,
            encoded_input["attention_mask"]
        )

        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())

        print(
            f"Embeddings gerados: "
            f"{min(i + EMBEDDING_BATCH_SIZE, len(chunks))}/{len(chunks)}"
        )

    embeddings_final = torch.cat(all_embeddings, dim=0)

    return embeddings_final.numpy().astype("float32")


def gerar_embedding_query(query, tokenizer, model):
    encoded_input = tokenizer(
        [query],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    embedding = mean_pooling(
        model_output,
        encoded_input["attention_mask"]
    )

    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.cpu().numpy().astype("float32")