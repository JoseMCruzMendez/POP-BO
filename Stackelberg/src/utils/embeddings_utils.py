from typing import List
from scipy import spatial
from tenacity import retry, stop_after_attempt, wait_random_exponential
import openai
import numpy as np
from sklearn.manifold import TSNE

# Copied from openai-cookbook
# https://github.com/openai/openai-cookbook/blob/3ed631d9f753d578c6a47d0aa6b02e32e3f3d75c/examples/utils/embeddings_utils.py#L174


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-similarity-davinci-001", **kwargs) -> List[float]:

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    response = openai.embeddings.create(input=[text], model=model, **kwargs)

    return response.data[0].embedding


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def aget_embedding(
    text: str, model="text-similarity-davinci-001", **kwargs
) -> List[float]:

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return (await openai.embeddings.create(input=[text], model=model, **kwargs))["data"][0][
        "embedding"
    ]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings(
    list_of_text: List[str], model="text-similarity-babbage-001", **kwargs
) -> List[List[float]]:
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = openai.embeddings.create(input=list_of_text, model=model, **kwargs).data
    return [d.embedding for d in data]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def aget_embeddings(
    list_of_text: List[str], model="text-similarity-babbage-001", **kwargs
) -> List[List[float]]:
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = (await openai.embeddings.create(input=list_of_text, model=model, **kwargs)).data
    return [d.embedding for d in data]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
    """Return a list of indices of nearest neighbors from a list of distances."""
    return np.argsort(distances)


def tsne_components_from_embeddings(
    embeddings: List[List[float]], n_components=2, **kwargs
) -> np.ndarray:
    """Returns t-SNE components of a list of embeddings."""
    # use better defaults if not specified
    if "init" not in kwargs.keys():
        kwargs["init"] = "pca"
    if "learning_rate" not in kwargs.keys():
        kwargs["learning_rate"] = "auto"
    tsne = TSNE(n_components=n_components, **kwargs)
    array_of_embeddings = np.array(embeddings)
    return tsne.fit_transform(array_of_embeddings)