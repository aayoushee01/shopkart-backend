import numpy as np
import pandas as pd
from scipy import sparse

lmbda = 0.0002

def create_embeddings(n, k):
    return 11 * np.random.random((n, k)) / k

def create_sparse_matrix(df, rows, cols, column_name="Score"):
    return sparse.csc_matrix((df[column_name].values, (df['UserId'].values, df['ProductId'].values)), shape=(rows, cols))

def encode_column(column):
    keys = column.unique()
    key_to_id = {key: idx for idx, key in enumerate(keys)}
    return key_to_id, np.array([key_to_id[x] for x in column]), len(keys)

def encode_df(df):
    food_ids, df['ProductId'], num_foods = encode_column(df['ProductId'])
    user_ids, df['UserId'], num_users = encode_column(df['UserId'])
    return df, num_users, num_foods, user_ids, food_ids

def predict(df, emb_user, emb_food):
    df['prediction'] = np.sum(np.multiply(emb_food[df['ProductId']], emb_user[df['UserId']]), axis=1)
    return df

def cost(df, emb_user, emb_food):
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_food.shape[0])
    predicted = create_sparse_matrix(predict(df, emb_user, emb_food), emb_user.shape[0], emb_food.shape[0], 'prediction')
    return np.sum((Y - predicted).power(2)) / df.shape[0]

def gradient(df, emb_user, emb_food):
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_food.shape[0])
    predicted = create_sparse_matrix(predict(df, emb_user, emb_food), emb_user.shape[0], emb_food.shape[0], 'prediction')
    delta = (Y - predicted)
    grad_user = (-2 / df.shape[0]) * (delta * emb_food) + 2 * lmbda * emb_user
    grad_anime = (-2 / df.shape[0]) * (delta.T * emb_user) + 2 * lmbda * emb_food
    return grad_user, grad_anime

def gradient_descent(df, emb_user, emb_food, iterations=2000, learning_rate=0.01, df_val=None):
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_food.shape[0])
    beta = 0.9
    grad_user, grad_food = gradient(df, emb_user, emb_food)
    v_user = grad_user
    v_food = grad_food
    for i in range(iterations):
        grad_user, grad_food = gradient(df, emb_user, emb_food)
        v_user = beta * v_user + (1 - beta) * grad_user
        v_food = beta * v_food + (1 - beta) * grad_food
        emb_user = emb_user - learning_rate * v_user
        emb_food = emb_food - learning_rate * v_food
        if not (i + 1) % 50:
            print("\niteration", i + 1, ":")
            print("train mse:", cost(df, emb_user, emb_food))
            if df_val is not None:
                print("validation mse:", cost(df_val, emb_user, emb_food))
    return emb_user, emb_food

def preprocessing():
    userBoughtArray = pd.read_csv("user_product_buys.csv").to_numpy()[:, 1:-1]
    userLikedArray = pd.read_csv("user_product_likes.csv").to_numpy()[:, 1:-1]
    userClickedArray = pd.read_csv("user_product_clicks.csv").to_numpy()[:, 1:-1]
    numUsers = userBoughtArray.shape[0]
    numProducts = userBoughtArray.shape[1]

    def create_df_from_array(data_array):
        rows, cols = np.where(data_array != 0)
        scores = data_array[rows, cols]
        return pd.DataFrame({"ProductId": cols, "UserId": rows, "Score": scores})

    userBoughtDf = create_df_from_array(userBoughtArray)
    userLikedDf = create_df_from_array(userLikedArray)
    userClickedDf = create_df_from_array(userClickedArray)

    return userBoughtDf, userLikedDf, userClickedDf, numUsers, numProducts
