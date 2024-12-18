import cv2
import streamlit as st
import numpy as np
from rembg import remove
import mediapipe as mp
import base64

# Function to set the background image
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            font-family: 'Times New Roman', Times, serif;
        }}
        .title {{
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            color: navy;
        }}
        .subtitle {{
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            color: navy;
            margin-bottom: 30px;
        }}
        .uploader-label {{
            color: navy;
            font-weight: bold;
            font-size: 18px;
            margin-top: 20px;
            text-align: center;
        }}
        .team-section {{
            text-align: center;
            color: navy;
            margin-top: 50px;
            border-top: 2px solid navy;
            padding-top: 20px;
        }}
        .team-header {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .team-member {{
            font-size: 18px;
            font-weight: bold;
            margin: 5px 0;
        }}
        .stButton > button {{
            background-color: #ff4d4d;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            padding: 8px 20px;
        }}
        .stButton > button:hover {{
            background-color: #ff6666;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


class PoseDetector:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()
        self.results = None

    def findPose(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        return img

    def findPosition(self, img):
        lmList = []
        if self.results and self.results.pose_landmarks:
            h, w, _ = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList

# Function to overlay an image on the background
def overlayImage(background, overlay, position):
    x, y = position
    h, w = overlay.shape[:2]

    # Ensure overlay does not go out of bounds
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return background

    # Check if the overlay has valid shape
    if overlay.shape[0] == 0 or overlay.shape[1] == 0 or overlay.shape[2] < 4:
        return background

    # Extract alpha channel and ensure shapes can be broadcast
    try:
        alpha_overlay = overlay[:, :, 3] / 255.0  # Extract alpha channel
        for c in range(3):  # Loop over the color channels
            background[y:y+h, x:x+w, c] = (
                alpha_overlay * overlay[:, :, c] +
                (1 - alpha_overlay) * background[y:y+h, x:x+w, c]
            )
    except ValueError:
        pass  # Suppress broadcasting errors
    
    return background

# Function to remove background from the shirt image
def remove_background(input_image):
    return remove(input_image)

# Main function for the Streamlit app
def main():
    add_bg_from_local('pexels-hngstrm-1939485.jpg')  # Updated background image path

    # Main Title
    st.markdown('<div class="title">Wardrobe Assistant</div>', unsafe_allow_html=True)
    # Subtitle for Project Guide    
    st.markdown('<div class="subtitle">Project Guide: Prof. Mahesh Kumar Sir</div>', unsafe_allow_html=True)

    # Upload shirt image
    st.markdown('<div class="uploader-label">Upload Shirt Image</div>', unsafe_allow_html=True)
    shirt_file = st.file_uploader("Upload Shirt Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    
    # Load a sample shirt image as a fallback
    sample_image_path = 'sample_shirt.png'
    sample_shirt_image = cv2.imread(sample_image_path, cv2.IMREAD_UNCHANGED)
    if sample_shirt_image is None:
        st.error("Error loading sample shirt image.")
        return

    if shirt_file is not None:
        shirt_image = cv2.imdecode(np.frombuffer(shirt_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        shirt_image = remove_background(shirt_image)

        if shirt_image is None or shirt_image.size == 0:
            st.error("Error processing shirt image.")
            return

        # Start live video
        if st.button("Start Live Video"):
            cap = cv2.VideoCapture(0)  # Use default webcam
            if not cap.isOpened():
                st.error("Could not access webcam.")
                return

            detector = PoseDetector()
            stframe = st.empty()

            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    break

                img = detector.findPose(img)
                lmList = detector.findPosition(img)

                if lmList:
                    try:
                        # Check for valid shoulder landmarks
                        if len(lmList) > 12:
                            lm11, lm12 = lmList[11][1:3], lmList[12][1:3]  # Shoulder landmarks
                            shoulderWidth = abs(lm11[0] - lm12[0])
                            midShoulderX, midShoulderY = (lm11[0] + lm12[0]) // 2, (lm11[1] + lm12[1]) // 2

                            # Scale and resize the shirt image based on shoulder width
                            widthOfShirt = int(shoulderWidth * 1.4)
                            heightOfShirt = int((shirt_image.shape[0] / shirt_image.shape[1]) * widthOfShirt)

                            if widthOfShirt > 0 and heightOfShirt > 0:
                                imgShirt = cv2.resize(shirt_image, (widthOfShirt, heightOfShirt))
                                offsetX, offsetY = int(widthOfShirt / 2), int(heightOfShirt * 0.15)
                                img = overlayImage(img, imgShirt, (midShoulderX - offsetX, midShoulderY - offsetY))
                    except Exception:
                        pass  # Suppress unexpected errors

                stframe.image(img, channels="BGR", use_container_width=True)

            cap.release()

    # Team Members Section
    st.markdown(
        """
        <div class="team-section">
            <div class="team-header">Team Members</div>
            <div class="team-member">Nikhil Soni - 221b251</div>
            <div class="team-member">Nirmal Singh - 221b252</div>
            <div class="team-member">Prakash Mani Patel - 221b268</div>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
