import streamlit as st
import os
import cv2
from skimage.feature import hog
from joblib import load
import skimage


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'

# Load the pre-trained SVM model
loaded_svm = load('../Model_Final/hog_svm_model.pkl')  # Change the filename/path accordingly

# Helper functions for classification
def classify_vehicle(image_path):
    # Read and preprocess the image
    test_image = cv2.imread(image_path)
    test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image_resized = cv2.resize(test_image_gray, (128, 64))

    # Extract HOG features for the test image
    test_hog = hog(test_image_resized, orientations=9, pixels_per_cell=(5, 5),
                   cells_per_block=(2, 2), visualize=False)

    # Perform vehicle classification using the loaded SVM model
    predicted_label = loaded_svm.predict([test_hog])[0]
    class_labels = ["bike", "bus", "car", "truck"]
    vehicle_class = class_labels[predicted_label]

    return vehicle_class

def preprocess_image_1(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (128, 64))
    return img_resized

def extract_hog_features(image):
    hog_features = hog(image, orientations=9, pixels_per_cell=(5, 5),
                        cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return hog_features

def detect_and_classify_vehicles(image_path):
    # Load pre-trained cascade classifiers for vehicles
    car_cascade = cv2.CascadeClassifier('../Cascades/car_cascade.xml')
    truck_cascade = cv2.CascadeClassifier('../Cascades/truck_cascade.xml')
    bus_cascade = cv2.CascadeClassifier('../Cascades/bus_cascade.xml')
    bike_cascade = cv2.CascadeClassifier('../Cascades/bike_cascade.xml')

    # Load the pre-trained SVM model
    loaded_svm = load('../Model_Final/hog_svm_model.pkl')  # Change the filename/path accordingly

    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Convert the input image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect vehicles using Haar cascades
    cars = car_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10)
    trucks = truck_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10)
    buses = bus_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10)
    bikes = bike_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10)

    # Combine all detected vehicles
    all_vehicles = [cars, trucks, buses, bikes]
    total_vehicle_count = 0

    # Process each detected vehicle
    for vehicle_list in all_vehicles:
        for (x, y, w, h) in vehicle_list:
            # vehicle_roi = image[y:y + h, x:x + w]
            # preprocessed_vehicle = preprocess_image_1(vehicle_roi)
            # hog_features = extract_hog_features(preprocessed_vehicle)
            # vehicle_class = classify_vehicle(image_path)

            # Display the bounding box and classified label on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(image, vehicle_class, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Increment total vehicle count
            total_vehicle_count += 1

    # Save the annotated image with bounding boxes and classified labels
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_vehicles.jpg')
    cv2.imwrite(output_image_path, image)

    return output_image_path, total_vehicle_count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify_single_vehicle', methods=['POST'])
def classify_single_vehicle():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded file to the static folder
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Classify the uploaded image
            predicted_class = classify_vehicle(filename)
            
            # Pass the classified vehicle class and image path to display
            return render_template('index.html', vehicle_class=predicted_class, img_path=filename)

@app.route('/detect_and_classify_vehicles', methods=['POST'])
def detect_and_classify_vehicles_route():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded file to the static folder
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Detect and classify vehicles in the uploaded image
            output_image, total_vehicles = detect_and_classify_vehicles(filename)
            
            if output_image:
                return render_template('index.html', img_path=output_image, total_vehicles=total_vehicles)

if __name__ == '__main__':
    app.run(debug=True)
