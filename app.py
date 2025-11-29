import os
import cv2
import pickle
import time
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Initialize Streamlit app
st.title("Hand Gesture Recognition")
st.sidebar.title("Options")

# Sidebar options
mode = st.sidebar.selectbox("Choose Mode", ["Collect Data", "Train Model", "Live Prediction"])

SAVE_DIR = './dataset'

# Mode 1: Data Collection
if mode == "Collect Data":
    st.subheader("Data Collection")
    class_count = st.sidebar.number_input("Number of Classes", min_value=1, max_value=100, value=36)
    images_per_class = st.sidebar.number_input("Images per Class", min_value=1, max_value=500, value=100)

    start_collection = st.button("Start Data Collection")

    if start_collection:
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        camera = cv2.VideoCapture(0)
        capturing = False  # Variable to track whether image capture is active

        for class_index in range(class_count):
            class_path = os.path.join(SAVE_DIR, str(class_index))
            if not os.path.exists(class_path):
                os.makedirs(class_path)

            st.write(f"Ready to collect data for class {class_index}.")

            image_index = 0

            while True:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to capture image. Check camera connection.")
                    break

                frame = cv2.flip(frame, 1)  # Flip the frame for mirror effect

                if capturing:
                    # Add collection progress text
                    cv2.putText(frame, f"Collecting images for class {class_index}: {image_index}/{images_per_class}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Save the current frame
                    image_path = os.path.join(class_path, f'{image_index}.jpg')
                    cv2.imwrite(image_path, frame)
                    image_index += 1

                    # Stop collecting images once the limit is reached
                    if image_index >= images_per_class:
                        capturing = False
                        st.write(f"Images for class {class_index} collected.")
                        break

                else:
                    # Add instructions text
                    cv2.putText(frame, "Press 's' to start collecting. Press 'q' to quit.",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('Data Collection', frame)

                key = cv2.waitKey(1) & 0xFF  # Wait for 1 millisecond
                if key == ord('s') and not capturing:
                    capturing = True
                elif key == ord('q'):
                    camera.release()
                    cv2.destroyAllWindows()
                    st.write("Data collection stopped by user.")
                    exit()

        camera.release()
        cv2.destroyAllWindows()
        st.success("Data Collection Complete")


# Mode 2: Model Training
elif mode == "Train Model":
    st.subheader("Train Model")
    start_training = st.button("Start Training")

    if start_training:
        IMAGE_DIR = './dataset'
        mp_hands = mp.solutions.hands
        hand_detector = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        features = []
        class_labels = []

        for class_folder in os.listdir(IMAGE_DIR):
            for image_file in os.listdir(os.path.join(IMAGE_DIR, class_folder)):
                feature_data = []
                x_coordinates = []
                y_coordinates = []

                image = cv2.imread(os.path.join(IMAGE_DIR, class_folder, image_file))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                results = hand_detector.process(image_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            x_coordinates.append(x)
                            y_coordinates.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            feature_data.append(x - min(x_coordinates))
                            feature_data.append(y - min(y_coordinates))

                    features.append(feature_data)
                    class_labels.append(class_folder)

        with open('processed_data.pickle', 'wb') as file:
            pickle.dump({'features': features, 'labels': class_labels}, file)

        st.success("Features extracted and saved to 'processed_data.pickle'. Training the model now...")

        data_dict = pickle.load(open('./processed_data.pickle', 'rb'))
        data = np.asarray(data_dict['features'])
        labels = np.asarray(data_dict['labels'])

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        score = accuracy_score(y_predict, y_test)
        st.success(f"Model trained with an accuracy of {score * 100:.2f}%")

        with open('model.p', 'wb') as f:
            pickle.dump({'model': model}, f)

# Mode 3: Live Prediction
elif mode == "Live Prediction":
    start_prediction = st.button("Start Prediction")

    if start_prediction:
        # Load the trained model only when the button is pressed
        model_dict = pickle.load(open('./model.p', 'rb'))
        model = model_dict['model']  # Load the model here inside the button click

        cap = cv2.VideoCapture(0)

        # Initialize MediaPipe hands and drawing tools
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

        # Labels for the classes
        labels_dict = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
            20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
            26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'
        }

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  
            if not ret:
                print("Failed to capture image. Check camera connection.")
                break

            # Convert the frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            data_aux = []
            x_ = []
            y_ = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Collect landmark points
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    # Normalize landmarks and append to data
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    # Predict character and get probabilities if data_aux is complete
                    if len(data_aux) == len(hand_landmarks.landmark) * 2:
                        prediction = model.predict([np.asarray(data_aux)])
                        probabilities = model.predict_proba([np.asarray(data_aux)])[0]  # Probabilities for each class
                        predicted_class = int(prediction[0])
                        predicted_character = labels_dict[predicted_class]
                        predicted_probability = 91 - (probabilities[predicted_class] * 100)  # Convert to percentage

                        # Define bounding box for displaying the prediction
                        x1 = int(min(x_) * frame.shape[1]) - 10
                        y1 = int(min(y_) * frame.shape[0]) - 10
                        x2 = int(max(x_) * frame.shape[1]) + 10
                        y2 = int(max(y_) * frame.shape[0]) + 10

                        # Draw bounding box and prediction text on frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

                        # Display the probability of the predicted class
                        text = f"{predicted_character}: {predicted_probability:.2f}%"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        text_x = frame.shape[1] - text_size[0] - 10
                        text_y = frame.shape[0] - 10
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the frame
            cv2.imshow('Hand Gesture Recognition', frame)

            # Press 'Q' to quit the main loop
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
