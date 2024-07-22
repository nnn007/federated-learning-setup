import matplotlib.pyplot as plt
import pandas as pd
import requests

# Fetch performance data from server
response = requests.get("http://127.0.0.1:8000/performance/")
performance_data = response.json()

df = pd.DataFrame(performance_data)
df.to_csv('performance.csv', index=False)

plt.plot(df['round'], df['accuracy'])
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Federated Learning Performance')
plt.savefig('performance.png')
plt.show()
