**Pose Detection Project Overview**
For my pose detection project, I began by conducting comprehensive research on how to approach the task. After gaining a solid understanding of the underlying concepts, I proceeded to write the Python code using **Visual Studio Code (VS Code)**. I ensured that I had installed the necessary extensions for **Jupyter Notebook** and **Python** in VS Code, and then used the command prompt to download the essential Python libraries, including **numpy**, **pandas**, **opencv-python**, **tensorflow**, and **keras**.

### **Dataset Creation and Model Development**
To build the dataset, I downloaded 12 videos from **YouTube**, representing 3 distinct gestures: **run**, **pushup**, and **cry** (with 4 videos per gesture: 3 for training and 1 for testing). After downloading, I converted these videos into **AVI** format for easier processing. From these videos, I generated a dataset, normalized it, split the data for training and testing, and then trained the model. The final model achieved **99.71% accuracy** during testing.

### **Project Workflow**

#### 1. **Capturing Frames and Extracting Pose Landmarks:**
Using **OpenCV** to open the video files and **MediaPipe Pose** to detect key body landmarks, I processed each video frame. As each frame was processed, the detected pose landmarks (such as joints and key body parts) were saved to a **CSV** file, along with the frame number and a label indicating the gesture (e.g., "run," "pushup," or "cry"). **MediaPipe Pose** detects **33 landmarks** (e.g., elbows, knees), and for each landmark, I captured the **x, y, z coordinates** and **visibility**.

cap = cv2.VideoCapture('run1.avi')
csv_writer.writerow(headers)

2. **Normalizing the Data:**
Once the raw landmark data was collected, I normalized the dataset by referencing the position of the **left hip** (landmark 23) to ensure that the model would focus on relative body movements rather than the absolute position of the individual.

for i in range(33):  df[f'x_{i}'] -= df['x_23']  # Normalize by left hip x-coordinate

3. **Splitting the Dataset:**
The normalized data was then split into **training** and **testing** sets using the `train_test_split()` method. This approach allowed the model to learn from one subset of the data and test its accuracy on a different subset.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

4. **Training the Model:**
For model development, I used **TensorFlow/Keras** to build a neural network. The network consists of multiple layers: two **dense layers** with 64 and 32 neurons respectively (both using the **ReLU** activation function), and an **output layer** that uses **softmax** activation to predict one of the three gestures (run, pushup, or cry). After compiling the model, I trained it on the landmark data, achieving **100% accuracy** on the test data.

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(set(y)), activation='softmax')  # Output layer])

5. **Saving the Model:**
After training the model, I saved it to a file for future use, enabling the system to easily load and use the trained model for further testing or deployment.
model.save(model_path)

6. **Testing the Model:**
For the final testing phase, I loaded the trained model and used it to predict gestures from a new test video. For each frame of the video, the pose landmarks were detected, prepared for the model, and used to predict the gesture (run, pushup, or cry). The predicted gesture was displayed in real-time on the video feed.

predicted_label = np.argmax(prediction)
label_text = label_map[predicted_label]
cv2.putText(image, f"Pose: {label_text}", (10, 40), ...)

### **Terminating the Program:**
The program displayed the test video with real-time pose labels. The video can be terminated by pressing the **Q** key.

### **Output:**
- **For pushup:**
- ![Screenshot 2024-11-14 060410](https://github.com/user-attachments/assets/2ed7ad4e-8299-4303-81bc-d1fdcdf289d8)

- **For crying:**
  ![Screenshot 2024-11-14 050456](https://github.com/user-attachments/assets/4397a3fa-f09a-49eb-9a10-af0856e30a71)


The following is a link to the project’s **GitHub repository**:  


 **Future Scope and Enhancements:**

The future potential of this pose detection project offers exciting opportunities for expansion and refinement. Below are several key areas where the project can be enhanced:

1. **Adding More Gestures and Activities:**
   - The current model recognizes three gestures: **run**, **pushup**, and **cry**. To improve the system's versatility, I plan to add more complex activities such as **jumping**, **sitting**, **standing up**, and even **yoga poses** or **dance movements**. This expansion would broaden the scope of the system for use in fields such as **sports coaching**, **fitness tracking**, and **rehabilitation therapy**.

2. **Real-time Pose Feedback and Correction:**
   - Beyond recognizing gestures, the system could be enhanced to provide **real-time feedback** on the correctness of the user's posture. In areas like fitness training or physical therapy, the system could assess whether an exercise is being performed correctly and provide corrective suggestions. This would not only help users improve their form but also reduce the risk of injuries.

3. **Integration with Wearable Devices:**
   - By integrating the pose detection system with **wearable devices** such as **smartwatches** or **AR glasses**, the model could provide even more granular data on physical activity. Additional sensors on wearables (e.g., heart rate, motion sensors, etc.) could be combined with the visual pose data to give a more complete picture of a user's performance, tracking metrics like fatigue or overall health.

4. **Enhanced Model Accuracy with 3D Pose Estimation:**
   - A future version of this system could incorporate **3D pose estimation** using depth sensors or stereo cameras, which would enable more accurate tracking of movements in three dimensions. This would be particularly useful for activities like **jumping** or **bending**, where depth information is critical for precise pose analysis.

5. **Applications in Augmented Reality (AR) and Virtual Reality (VR):**
   - The pose detection system could be adapted for use in **AR** and **VR** applications, enabling users to control avatars or interact with virtual environments based on body movements. For example, the system could be used in **virtual fitness classes** to track and correct users’ poses in real time or in **gaming** to create more immersive and interactive experiences.

6. **Multi-person Pose Detection and Interaction:**
   - The current model focuses on detecting the pose of a single individual. Future iterations could expand to support **multi-person pose detection**, allowing the system to track multiple users simultaneously. This would be beneficial in applications such as **team sports analysis**, **group fitness classes**, or even **crowd behavior analysis** in security or public safety contexts.

### **Conclusion:**

This pose detection system, while initially designed for gesture recognition, has the potential to evolve into a comprehensive platform with a broad range of applications. From sports and fitness to healthcare, gaming, and entertainment, the future possibilities are vast. As advancements continue in AI, sensor technologies, and real-time processing, the system's capabilities can be enhanced to provide even more sophisticated insights and user experiences.


Here are the **confusion matrices** and **F1 scores** for the trained model:
![Screenshot 2024-11-14 060101](https://github.com/user-attachments/assets/88bf055e-cc1c-49df-8685-c3d6d884b311)



