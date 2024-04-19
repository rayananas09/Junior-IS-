import pickle
import json
import numpy as np

class HomePricePredictor:
    def __init__(self):
        self.__locations = None
        self.__data_columns = None
        self.__model = None

    def get_estimated_price(self, location, sqft, bhk, bath):
        loc_index = self.__data_columns.index(location.lower()) if location.lower() in self.__data_columns else -1
        x = np.zeros(len(self.__data_columns))
        x[0] = sqft
        x[1] = bhk
        x[2] = bath
        if loc_index >= 0:
            x[loc_index] = 1
        return round(self.__model.predict([x])[0], 2)

    def load_saved_artifacts(self):
        print("Loading saved artifacts...")
        with open("./artifacts/columns.json", "r") as f:
            self.__data_columns = json.load(f)['data_columns']
            self.__locations = self.__data_columns[3:]

        if self.__model is None:
            with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
                self.__model = pickle.load(f)

    def get_location_names(self):
        return self.__locations

    def get_data_columns(self):
        return self.__data_columns
