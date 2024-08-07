import cv2
import numpy as np
import tensorflow as tf  # Use tensorflow directly
import tensorflow_hub as hub

# ... (rest of your code)
def build_model():
    # ... (rest of the function)
    pretrained_model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
    pretrained_model = hub.KerasLayer(pretrained_model_url, input_shape=(224, 224, 3), trainable=False)
  
    model = tf.keras.Sequential()  # Use tf.keras for consistency
    model.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3))) # Add an Input layer to specify the input shape
    model.add(tf.keras.layers.Lambda(lambda x: pretrained_model(x))) # Wrap the KerasLayer in a Lambda layer 
    model.add(tf.keras.layers.Dense(128, activation='relu'))  # Now using tf.keras.layers
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



# Preprocess a single frame
def preprocess_frame(frame):
    # Resize frame to 224x224 for the pretrained model
    resized_frame = cv2.resize(frame, (224, 224))
    resized_frame = resized_frame / 255.0  # Normalize pixel values
    resized_frame = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
    return resized_frame


model = build_model()

# Process video
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)
        
        # Predict gender using the pretrained model
        features = model.predict(preprocessed_frame)
        predictions.append(features[0][0])
        
        # Display the frame with prediction
        gender = 'Female' if features[0][0] > 0.5 else 'Male'
        cv2.putText(frame, f'Gender: {gender}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Gender Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Aggregate predictions
    average_prediction = np.mean(predictions)
    final_gender = 'Female' if average_prediction > 0.5 else 'Male'
    print(f'Final Predicted Gender: {final_gender}')

# Path to the input video
video_path = 0
# Main function to run the video processing and gender detection
def main():
    process_video(video_path, model)

if __name__ == "__main__":
    main()
