# Face Recognition Based Attendance System

A web-based attendance system that uses face recognition to automatically mark attendance. Built with Streamlit, OpenCV, and face_recognition library.

## Features

- 📸 Multiple input methods: Upload image or use webcam
- 👤 Automatic face detection and recognition
- 📝 Daily attendance logging
- 📊 Real-time attendance dashboard
- 📱 Mobile-friendly interface
- 🔒 One attendance entry per person per day

## Prerequisites

- Python 3.10 or higher
- Conda (recommended) or pip
- Webcam (for live capture)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/100rabhsah/opencv-attendance-project.git
   cd opencv-attendance-project
   ```

2. **Create and activate conda environment:**
   ```bash
   conda create -n attendance python=3.10
   conda activate attendance
   ```

3. **Install dependencies:**
   ```bash
   conda install -c conda-forge opencv numpy pandas pillow streamlit face_recognition
   ```

## Usage

1. **Add known faces:**
   - Create an `images` folder in the project root
   - Add photos of people you want to recognize
   - Name each image file as the person's name (e.g., `John_Doe.jpg`)
   - Use clear, front-facing photos for best results

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Access the web interface:**
   - Open your browser and go to the URL shown in the terminal (usually http://localhost:8501)
   - Choose between uploading an image or using webcam
   - The system will automatically detect faces and mark attendance

## How It Works

1. **Face Detection:**
   - Uses OpenCV and face_recognition library to detect faces in images
   - Works with both uploaded images and webcam captures

2. **Face Recognition:**
   - Compares detected faces with known faces in the `images` folder
   - Uses face encodings for accurate matching

3. **Attendance Logging:**
   - Logs attendance in `Attendance.csv`
   - Records: Name, Timestamp, Date
   - Ensures one entry per person per day

4. **Dashboard:**
   - Shows today's attendance in the sidebar
   - Displays total number of attendees
   - Lists names of present individuals

## File Structure

```
opencv-attendance-project/
├── app.py              # Main application file
├── images/            # Directory for known face images
├── Attendance.csv     # Attendance log file
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Image Requirements

- Format: JPG, JPEG, or PNG
- Content: Clear, front-facing photo of one person
- Naming: Use underscores instead of spaces (e.g., `John_Doe.jpg`)
- Size: Recommended 300x300 pixels or larger

## Troubleshooting

1. **OpenCV Installation Issues:**
   ```bash
   conda remove opencv opencv-python --force
   conda install -c conda-forge opencv
   ```

2. **Face Recognition Not Working:**
   - Ensure images are clear and well-lit
   - Check if faces are properly detected
   - Verify image naming format

3. **Webcam Not Working:**
   - Check webcam permissions
   - Ensure no other application is using the webcam
   - Try using the upload image option instead

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- face_recognition library for face detection and recognition
- Streamlit for the web interface 