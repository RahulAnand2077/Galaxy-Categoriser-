import tensorflow as tf

# List all GPUs
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus)

# Enable memory growth so TF doesn’t pre‑allocate all GPU RAM
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# (Optional) Log device placement for ops
tf.debugging.set_log_device_placement(True)
