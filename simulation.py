import threading
from edge_device import download_model, train_loader, train_local, upload_weights


# Simulate edge device training and communication
def edge_device_simulation(device_id):
    print(f"Device {device_id} started simulation")

    print(f"Device {device_id} started download model")
    global_model = download_model()
    print(f"Device {device_id} finished download model")

    print(f"Device {device_id} started local training")
    local_weights = train_local(global_model, train_loader)
    print(f"Device {device_id} finished local training")

    print(f"Device {device_id} started upload weights")
    upload_weights(local_weights)
    print(f"Device {device_id} finished upload weights")

    print(f"Device {device_id} finished simulation")


# # Multithreaded simulation of edge devices
# threads = []
# n_devices = 5
# for i in range(n_devices):
#     t = threading.Thread(target=edge_device_simulation, args=(i,))
#     threads.append(t)
#     t.start()
#
# for t in threads:
#     t.join()

# Simulate edge device training and communication
# def edge_device_simulation(device_id):
#     print(f"Device {device_id} started training")
#     global_model = download_model()
#     local_weights = train_local(global_model, train_loader)
#     upload_weights(local_weights)
#     print(f"Device {device_id} finished training")
#
#
# Multithreaded simulation of edge devices
threads = []
n_devices = 5
for i in range(n_devices):
    t = threading.Thread(target=edge_device_simulation, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# single device
# edge_device_simulation(device_id=0)
