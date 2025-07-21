import pymongo
import bcrypt
import ast
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LoginManager:

    def __init__(self) -> None:
        # MongoDB connection
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["hw3"]
        self.collection = self.db["users"]
        self.salt = b"$2b$12$ezgTynDsK3pzF8SStLuAPO"  # TODO: if not working, generate a new salt

    def register_user(self, username: str, password: str) -> None:
        if not username or not password:
            raise ValueError("Username and password are required.")
        if len(username) < 3 or len(password) < 3:
            raise ValueError("Username and password must be at least 3 characters.")
        if self.collection.find_one({"username": username}):
            raise ValueError(f"User already exists: {username}.")

        hashed_password = bcrypt.hashpw(password.encode(), self.salt)
        self.collection.insert_one({"username": username, "password": hashed_password})

    def login_user(self, username: str, password: str) -> object:
        hashed_password = bcrypt.hashpw(password.encode(), self.salt)
        user = self.collection.find_one({"username": username, "password": hashed_password})
        if user:
            print(f"Logged in successfully as: {username}")
            return user
        raise ValueError("Invalid username or password")


class DBManager:

    def __init__(self) -> None:
        # MongoDB connection
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["hw3"]
        self.user_collection = self.db["users"]
        self.game_collection = self.db["games"]

    def load_csv(self) -> None:
        df = pd.read_csv("NintendoGames.csv")
        df["genres"] = df["genres"].apply(ast.literal_eval)
        df["is_rented"] = False
        for _, row in df.iterrows():
            if not self.game_collection.find_one({"title": row["title"]}):
                self.game_collection.insert_one(row.to_dict())

    def rent_game(self, user: dict, game_title: str) -> str:
        game = self.game_collection.find_one({"title": game_title})
        if not game:
            return f"{game_title} not found"
        if game["is_rented"]:
            return f"{game_title} is already rented"
        self.game_collection.update_one({"_id": game["_id"]}, {"$set": {"is_rented": True}})
        self.user_collection.update_one({"_id": user["_id"]}, {"$push": {"rented_game_ids": game["_id"]}})
        return f"{game_title} rented successfully"

    def return_game(self, user: dict, game_title: str) -> str:
        game = self.game_collection.find_one({"title": game_title})
        if not game or game["_id"] not in user.get("rented_games", []):
            return f"{game_title} was not rented by you"
        self.game_collection.update_one({"_id": game["_id"]}, {"$set": {"is_rented": False}})
        self.user_collection.update_one({"_id": user["_id"]}, {"$pull": {"rented_game_ids": game["_id"]}})
        return f"{game_title} returned successfully"

    def recommend_games_by_genre(self, user: dict) -> list:
        rented_game_ids = user.get("rented_game_ids", [])
        if not rented_game_ids:
            return ["No games rented"]

        rented_games = list(self.game_collection.find({"_id": {"$in": rented_game_ids}}))
        genres = [genre for game in rented_games for genre in game["genres"]]
        genre = random.choices(genres, k=1)[0]

        recommendations = list(
            self.game_collection.aggregate([
                {"$match": {"genres": genre, "_id": {"$nin": rented_game_ids}}},
                {"$sample": {"size": 5}}
            ])
        )
        return [game["title"] for game in recommendations]

    def recommend_games_by_name(self, user: dict) -> list:
        rented_game_ids = user.get("rented_game_ids", [])
        if not rented_game_ids:
            return ["No games rented"]

        rented_games = list(self.game_collection.find({"_id": {"$in": rented_game_ids}}))
        random_game = random.choice(rented_games)
        all_titles = [game["title"] for game in self.game_collection.find()]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_titles)
        chosen_title_index = all_titles.index(random_game["title"])
        cosine_sim = cosine_similarity(tfidf_matrix[chosen_title_index], tfidf_matrix).flatten()
        sorted_indices = cosine_sim.argsort()[::-1]
        similar_indices = [
                              i for i in sorted_indices
                              if all_titles[i] not in [game["title"] for game in rented_games]
                          ][:5]
        return [all_titles[i] for i in similar_indices]

    def find_top_rated_games(self, min_score) -> list:
        games = list(
            self.game_collection.find(
                {"user_score": {"$gte": min_score}},
                {"title": 1, "user_score": 1}
            ).sort("user_score", -1)
        )
        return [{"title": game["title"], "user_score": game["user_score"]} for game in games]

    def decrement_scores(self, platform_name) -> None:
        self.game_collection.update_many(
            {"platform": platform_name}, {"$inc": {"user_score": -1}}
        )

    def get_average_score_per_platform(self) -> dict:
        pipeline = [
            {"$group": {"_id": "$platform", "average_score": {"$avg": "$user_score"}}}
        ]
        result = list(self.game_collection.aggregate(pipeline))
        return {doc["_id"]: round(doc["average_score"], 3) for doc in result}

    def get_genres_distribution(self) -> dict:
        pipeline = [
            {"$unwind": "$genres"},
            {"$group": {"_id": "$genres", "count": {"$sum": 1}}}
        ]
        result = list(self.game_collection.aggregate(pipeline))
        return {doc["_id"]: doc["count"] for doc in result}
