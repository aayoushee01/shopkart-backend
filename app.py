from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import csv
from util import *

app = Flask(__name__)
CORS(app) 

@app.route('/api/data', methods=['GET'])
def get_data():
    data = []

    with open('user_data.csv', 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)

    return jsonify(data)

@app.route('/api/products', methods=['GET'])
def get_product():
    data = []

    with open('products.csv', 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)

    return jsonify(data)

@app.route('/api/recomended-products', methods=['GET'])
def get_recomended_products():
    user_id = request.args.get("user_id", default=-1)

    if user_id == -1:
        return jsonify({"error": "Invalid user ID"}), 400

    num_recommendations = 10 

    recommended_indices = get_recommandation(user_id, num_recommendations)
    
    # Assuming you have a product mapping or lookup to get product details
    recommended_products = [get_product_details(idx) for idx in recommended_indices]

    return jsonify(recommended_products)

product_data = pd.read_csv("products.csv")

def get_product_details(product_id):
    product = product_data[product_data["id"] == product_id]
    
    if not product.empty:
        product_details = {
            "id": int(product["id"].values[0]),
            "title": product["title"].values[0],
            "description": product["description"].values[0],
            "price": float(product["price"].values[0]),  # Convert to float if needed
            "discount": int(product["discountPercentage"].values[0]),
            "rating": float(product["rating"].values[0]),  # Convert to float if needed
            "stock": int(product["stock"].values[0]),
            "brand": product["brand"].values[0],
            "category": product["category"].values[0],
            "thumbnail": product["thumbnail"].values[0]
        }
        return product_details
    else:
        return None

@app.route('/api//update_interaction', methods=['POST'])
def update_interactions():
    data = request.json
    user_id = data['user_id']
    product_id = data['product_id']
    interaction_type = data['interaction_type']

    fieldnames = []
    rows = []
    with open(f'user_product_{interaction_type}s.csv', "r") as csv_file:
      reader = csv.DictReader(csv_file)
      fieldnames = reader.fieldnames
      for row in reader:
        rows.append(row)
    
    # # Update data
    for row in rows:
      r_user_id = row["User ID"]
      if r_user_id == user_id:
        row[product_id] = 1 + int(row[product_id])
    
    with open(f'user_product_{interaction_type}s.csv', "w", newline="") as csv_file:
      # fieldnames = ["User ID", "1", "2", "3", ...]  # List of column headers
      writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(rows)

    return jsonify({'message': 'updated'})

def train_model():
  userBoughtDf, userLikedDf, userClickedDf, numUsers, numProducts = preprocessing()
  emb_user_bought = create_embeddings(numUsers, 5)
  emb_food_bought = create_embeddings(numProducts, 5)
  print("iteration", 0, ":")
  print("train mse:",  cost(userBoughtDf, emb_user_bought, emb_food_bought))
  emb_user_bought, emb_food_bought = gradient_descent(userBoughtDf, emb_user_bought, emb_food_bought, iterations=200, learning_rate=0.1)

  emb_user_liked = create_embeddings(numUsers, 5)
  emb_food_liked = create_embeddings(numProducts, 5)
  print("iteration", 0, ":")
  print("train mse:",  cost(userLikedDf, emb_user_liked, emb_food_liked))
  emb_user_liked, emb_food_liked = gradient_descent(userBoughtDf, emb_user_liked, emb_food_liked, iterations=200, learning_rate=0.1)

  emb_user_clicked = create_embeddings(numUsers, 5)
  emb_food_clicked = create_embeddings(numProducts, 5)
  print("iteration", 0, ":")
  print("train mse:",  cost(userClickedDf, emb_user_clicked, emb_food_clicked))
  emb_user_clicked, emb_food_clicked = gradient_descent(userBoughtDf, emb_user_clicked, emb_food_clicked, iterations=200, learning_rate=0.1)
  return emb_food_bought, emb_food_bought, emb_user_liked, emb_food_liked, emb_user_clicked, emb_food_clicked


def get_recommandation(user_id, num_recommadations):
  emb_user_bought, emb_food_bought, emb_user_liked, emb_food_liked, emb_user_clicked, emb_food_clicked = train_model()
  user_ids = pd.read_csv("user_data.csv").to_numpy()[:,0]
  print(user_ids)
  print(type(user_ids))
  print(type(user_ids[0]))
  print(type(user_id))
  luserids = list(user_ids)
  print(type(luserids))
  print(type(luserids[0]))
  user_id = int(user_id)
  idx = list(user_ids).index(user_id)
  ratings_predicted_bought = np.matmul(np.reshape(emb_user_bought[idx], (1, 5)), np.transpose(emb_food_bought))
  ratings_predicted_liked = np.matmul(np.reshape(emb_user_liked[idx], (1, 5)), np.transpose(emb_food_liked))
  ratings_predicted_clicked = np.matmul(np.reshape(emb_user_clicked[idx], (1, 5)), np.transpose(emb_food_clicked))
  
  ratings_predicted = np.ones_like(ratings_predicted_bought)
  for i in range(len(ratings_predicted_bought)):
    ratings_predicted[i] = 0.6 * ratings_predicted_bought + 0.3 * ratings_predicted_liked + 0.1 * ratings_predicted_clicked 
  recommanded_index = ratings_predicted[0].argsort()[-num_recommadations:][::-1]
  print(recommanded_index)
  for i in range(num_recommadations):
    recommanded_index[i] += 1
  return recommanded_index 

if __name__ == '__main__':
    app.run(debug=True)