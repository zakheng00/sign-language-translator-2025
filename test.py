import numpy as np
import requests
     
X_train = np.load('X_train.npy')
frames = X_train[0].tolist()  # 取第一個樣本
response = requests.post('http://localhost:5000/predict', json={'frames': frames})
print(response.json())