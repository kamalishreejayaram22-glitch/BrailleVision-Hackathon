BrailleVision_Hackathon - Visionary-coders 
Desciption : BrailleVision is a simple assistive tool designed to make Braille easier to understand in the digital world. It uses computer vision and deep  learning to read Braille from images and convert it into normal text and speech. This helps visually impaired users access written information more easily and independently.
## Prerequisites
Python 3.9 or above
GPU T4 python 
## Setup 
Install the required libraries:
pip install -r requirements.txt
## Run the project 
Run the streamlit application using :
 streamlit run frontend/app.py
After execution, open the local URL displayed in the terminal (typically http://localhost:8501).
## Project structure 
frontend/     - Streamlit user interface  
backend/      - Backend processing logic  
model/        - Trained model files (best.pt)  
training/     - Model training scripts and notebooks  
dataset/      - Dataset configuration and information  
inference/    - Prediction and inference scripts 
## Project Workflow 
1. Takes the embossed braille scripts or characters as input  through webcam or uploads
2. Braille characters are detected using trained YOLOV8 model in backend , which undergoes anotation of braille cells and detection of letters or numbers
3. Detected Charactrers are converted to English text  which were displayed in user friendly streamlit frontend app.py
4. Additionally , the English text is converted into speech for more functionality and usability 
                     * gttx - online text to speech conversion
                     * pyttsx3 - offline text to speech conversion
## Team
Visionary Coders
Shivani – UI Development and YOLO Integration 
Kamali Shree – Image Processing, YOLO Model Development 


