from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

def read_img(filename):
    img = cv2.imread(filename)
    return img



def edge_detection(img, line_width, blur):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(gray, blur)
    edges = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_width, blur)
    return edges

def color_quantisation(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result



@app.route('/', methods=['GET', 'POST'])
def cartoonize():
    if request.method == 'POST':
      
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

       
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            try:
              
                filename = 'uploaded.jpg'
                file.save(filename)

             
                img = read_img(filename)

             
                line_width = 9
                blur_value = 7
                totalColor = 4
                edgeImg = edge_detection(img, line_width, blur_value)
                cartoon_img = color_quantisation(img, totalColor)

               
                blured = cv2.bilateralFilter(cartoon_img, 7, 200, 200)

               
                cartoon = cv2.bitwise_and(blured, blured, mask=edgeImg)

                
                output_filename = 'static/cartoonized.jpg'
                cv2.imwrite(output_filename, cartoon)

             
                return redirect(url_for('result'))

            except Exception as e:
                flash(f"Error processing image: {e}")
                return redirect(request.url)

    return render_template('upload.html')

# Route for displaying result page
@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
