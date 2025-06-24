from utils.embeddings_utils import get_embeddings
import pandas as pd
import numpy as np
import os


if __name__ == "__main__":
    city = "Philadelphia"
    min_review_count = 500
    min_review_per_user = 90
    aggregate_for_business = True
    api_key = "openai-api-key"  # Update to your OpenAI API key
    embedding_model = "text-embedding-3-large"
    model_arguments = {
        "dimensions": 32
    }

    os.environ["OPENAI_API_KEY"] = api_key

    df = pd.read_csv(
        f"../data/yelp_aggregates/reviews_{city}_min_review_{min_review_count}_min_reviews_per_user_{min_review_per_user}.csv",
        index_col=0,
    )
    print("Data loaded")
    df["text"] = df["text"].str.strip()
    if aggregate_for_business:
        print(df.head())
        print("Concatenate reviews for business")
        df = df.groupby("business_id")["text"].apply(lambda x: " ".join(x))
        # Cut long reviews
        long_review = df["qISf5ojuYbD9h71NumGUQA"].split("\n")
        long_review = "\n".join(long_review[:int(0.9*len(long_review))])
        df.loc["qISf5ojuYbD9h71NumGUQA"] = long_review
        df = df.reset_index()
        df.to_csv(
            f"../data/yelp_aggregates/reviews_{city}_min_review_{min_review_count}_min_reviews_per_user_{min_review_per_user}_aggregated.csv",
            index=False,
        )
    print("Data shape: ", df.shape)
    print(df.head())
    embeddings = []
    for i in range(0, len(df["text"]), 2000):
        print("Batch: ", i)
        batch = list(df["text"].iloc[i:i+2000].replace("\n", " "))
        batch_embeddings = get_embeddings(
            batch,
            model=embedding_model,
            **model_arguments,
        )
        embeddings.extend(batch_embeddings)
    embeddings = np.array(embeddings)
    print("Embedding created")
    print("Embeddings size: ", embeddings.shape)
    print(embeddings)
    # save embeddings
    output_path = f"../data/yelp_aggregates/reviews_{city}_min_review_{min_review_count}_min_reviews_per_user_{min_review_per_user}_embeddings_32.npy"
    np.save(output_path, embeddings)
