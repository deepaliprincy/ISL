from tensorflow import keras

# Load your existing .h5 model
model = keras.models.load_model("model.h5")

# Export to TensorFlow SavedModel format (folder)
model.export("saved_model_dir")
print("Model exported to SavedModel format successfully!")
