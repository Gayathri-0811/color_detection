# color_detection

# 🎨 Dominant Color Extractor

A web-based tool built with **Streamlit** and **OpenCV** that identifies and visualizes the dominant colors in an uploaded image using **KMeans clustering**. Ideal for artists, designers, and developers who need to extract a color palette from an image easily.

---

## 📸 Features

- Upload an image (`.jpg`, `.jpeg`, `.png`)
- Choose how many dominant colors to extract (1 to 5)
- View:
  - Individual color blocks with percentages
  - A proportion bar showing all dominant colors
  - Final annotated image with color palette overlay
- Download the final image with the embedded palette

---

## 🧠 How It Works

1. **Image Preprocessing**  
   - Uploaded image is resized for clustering.  
   - Pixel data is flattened into a 2D array.

2. **KMeans Clustering**  
   - Uses `sklearn.cluster.KMeans` to find dominant colors.  
   - Colors are ranked by their prevalence.

3. **Visualization**  
   - Displays color blocks and bar chart.  
   - Overlays dominant color palette on original image.

---

## 🛠 Installation

### 📦 Requirements

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- scikit-learn
- imutils
- Pillow

### 🔧 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Gayathri-0811/color_detection.git
cd color_detection

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
