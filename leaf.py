# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model  

filepath = 'C:/Users/G V SOWMIYAA/Desktop/leaf trial/Plant-Leaf-Disease-Prediction/model.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully") 

def pred_tomato_disease(tomato_plant):
    test_image = load_img(tomato_plant, target_size=(128, 128))  # load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # change dimension 3D to 4D

    result = model.predict(test_image)  # predict diseased plant or not
    print('@@ Raw result = ', result)

    pred = np.argmax(result, axis=1)
    print(pred)
    if pred == 0:
        return "Tomato - Bacteria Spot Disease", 'Tomato-Bacteria Spot.html'
    elif pred == 1:
        return "Tomato - Early Blight Disease", 'Tomato-Early_Blight.html'
    elif pred == 2:
        return "Tomato - Healthy and Fresh", 'Tomato-Healthy.html'
    elif pred == 3:
        return "Tomato - Late Blight Disease", 'Tomato - Late_blight.html'
    elif pred == 4:
        return "Tomato - Leaf Mold Disease", 'Tomato - Leaf_Mold.html'
    elif pred == 5:
        return "Tomato - Septoria Leaf Spot Disease", 'Tomato - Septoria_leaf_spot.html'
    elif pred == 6:
        return "Tomato - Target Spot Disease", 'Tomato - Target_Spot.html'
    elif pred == 7:
        return "Tomato - Tomato Yellow Leaf Curl Virus Disease", 'Tomato - Tomato_Yellow_Leaf_Curl_Virus.html'
    elif pred == 8:
        return "Tomato - Tomato Mosaic Virus Disease", 'Tomato - Tomato_mosaic_virus.html'
    elif pred == 9:
        return "Tomato - Two Spotted Spider Mite Disease", 'Tomato - Two-spotted_spider_mite.html'

# Create flask instance for the deployment
app = Flask(__name__) 

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

# get input image from the client then predict class and render the respective .html page for the solution
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # get input
        filename = file.filename
        print("@@ Input posted = ", filename)

        # Specify the directory to save the uploaded image
        upload_dir = r'C:\Users\G V SOWMIYAA\Desktop\leaf trial\Plant-Leaf-Disease-Prediction\Dataset\test'



        # Save the uploaded image
        file_path = os.path.join(upload_dir, filename)

        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_tomato_disease(tomato_plant=file_path)

        return render_template(output_page, pred_output=pred, user_image=file_path)

# For the local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, port=8080, debug=True)



