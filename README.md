# -
スマートモビリティは、エッジコンピューティングと機械学習を活用して、自動運転車の安全性と効率性を向上させ、交通渋滞の緩和と事故防止に貢献します。
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Simulate sensor data (e.g., distance to the next car, speed, and road conditions)
# For a real project, this data would come from the vehicle's sensors
def simulate_sensor_data():
    # Random data for demonstration
    distance_to_next_car = np.random.randint(5, 100)  # in meters
    speed = np.random.randint(20, 130)  # in km/h
    road_condition = np.random.choice(['wet', 'dry', 'icy'])  # example road conditions
    return distance_to_next_car, speed, road_condition

# Encode road conditions as numerical data
def encode_road_condition(condition):
    return {'dry': 0, 'wet': 1, 'icy': 2}[condition]

# A simple machine learning model to decide whether to accelerate, maintain speed, or decelerate
# Based on the current speed, distance to the next car, and road conditions
class AutonomousVehicleModel:
    def __init__(self):
        self.classifier = RandomForestClassifier()
        # Example training data
        X = np.array([
            [50, 70, 0],  # distance, speed, road_condition
            [30, 50, 1],
            [10, 30, 2],
        ])
        y = np.array(['decelerate', 'maintain', 'accelerate'])  # actions
        self.classifier.fit(X, y)
    
    def predict_action(self, distance, speed, road_condition):
        encoded_condition = encode_road_condition(road_condition)
        return self.classifier.predict([[distance, speed, encoded_condition]])[0]

# Simulate edge computing by processing data locally on the device (vehicle) instead of sending it to a central server
def edge_compute_decision(distance, speed, road_condition):
    model = AutonomousVehicleModel()
    action = model.predict_action(distance, speed, road_condition)
    return action

def main():
    for _ in range(5):  # Simulate 5 instances
        distance, speed, road_condition = simulate_sensor_data()
        print(f"Sensor Data: Distance={distance}m, Speed={speed}km/h, Road Condition={road_condition}")
        action = edge_compute_decision(distance, speed, road_condition)
        print(f"Recommended Action: {action}\n")

if __name__ == "__main__":
    main()
