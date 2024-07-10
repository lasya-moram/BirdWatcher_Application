#importing necessary libraries
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
import requests
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
import boto3
import sqlite3

# Load the trained Xception model for species classification
species_xception = load_model('saved_models/species3_xception.h5')


# Load image metadata from CSV file
imagedata = pd.read_csv('data/imagedata.csv', index_col=0)
# Initialize Flask app
app = Flask(__name__)
# Configure secret key for CSRF protection and upload folder
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATABASE'] = 'uploads.db'

app.config.from_object('config.DevConfig')
# Define allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
# Function to check if the uploaded file's extension is allowed
def allowed_file(filename):
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)
# Function to create SQLite database table for uploaded files
def create_table():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS uploaded_files (
                        id INTEGER PRIMARY KEY,
                        filename TEXT NOT NULL,
                        notes TEXT
                    )''')
    conn.commit()
    conn.close()
# Function to insert uploaded file details into the database
def insert_file(filename, notes):
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO uploaded_files (filename, notes) VALUES (?, ?)''', (filename, notes))
    conn.commit()
    conn.close()
# Function to display uploaded files from the database
def display_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM uploaded_files''')
    data = cursor.fetchall()
    conn.commit()
    conn.close()
    print("Uploaded Files:")
    for row in data:
        print("Filename:", row[1])
        print("Notes:", row[2])
        print()
# Create SQLite database table
create_table()
# Initialize uploads folder and allowed file extensions
uploads= 'uploads' #initializing the uploads folder
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'} #accepts images with only specified extensions 
app.config['uploads']= uploads #configuring uploads folder to store uploaded images by the user

#helping function to validate the uploaded image's file extension
def allowed_file(filename): #function to check if the uploaded file's extension is allowed or not
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS #security measure to prevent wrong file uploading
#def upload_to_s3(data, filename):
#    s3_client.put_object(Bucket=bucket_name, Key=filename, Body=data)
# Function to make predictions using the trained model
def model_predict(filepath):
    img = Image.open(filepath)
    img_rs = np.array(img.resize((299,299)))/255
    prediction = species_xception.predict(img_rs.reshape(1,299,299,3))
    return np.round(prediction * 100, 1)[0]

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for 'mybirdwat' page
@app.route('/mybirdwat')
def mybirdwat_page():
    return render_template('mybirdwat.html')

# Route for image upload and prediction
@app.route('/predict', methods=['GET', 'POST']) #upload and prediction route
def predict():
    if request.method == 'POST':
        file = request.files['bird_image'] #gets the file from the key "bird_image"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print("saved")
            notes = request.form['notes']  # Get the notes from the form
            insert_file(filename, notes)
            labels = np.unique(np.array(imagedata['species_group'].values))
            prediction = model_predict(filepath)
            top_3 = prediction.argsort()[-1:-4:-1]
            display_db()
            return render_template('predict.html', prediction=prediction, labels=labels, top_3=top_3)
            #return jsonify({'message': 'Upload successful'})
        else:
            flash('An error occurred, try again.')
            return redirect(request.url)  
       

# Route for 'about' page
@app.route('/about')
def about_page():
    return render_template('about.html')

# Route for 'birds' page
@app.route('/birds')
def birds_page():
    return render_template('birds.html')

# Display uploaded files from the database
display_db()
# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)