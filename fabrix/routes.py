from fabrix.models import User
from fabrix import app,db,bcrypt
from flask import Flask, render_template,flash,redirect,url_for,abort,request, jsonify
import os
from fabrix.forms import RegistrationForm,LoginForm
from flask_login import login_user, current_user,logout_user,login_required
from werkzeug.utils import secure_filename

# import fastbook
# from fastbook import *
from fastai.vision.all import *
from fastai.learner  import load_learner
from fastai.vision import *
from fastai.vision.core import PILImage
from fastai import *
from io import BytesIO
from PIL import Image 
import torch
import torchvision.transforms.functional as F


import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# folder_path = 'C:/Users/SHREYAS/Desktop/Fabric Project/Final/fabrixx/fabrix/models/export.pkl'
# folder_path_fault = 'C:/Users/SHREYAS/Desktop/Fabric Project/Final/fabrixx/fabrix/models/fault.pkl'
folder_path = 'fabrix/models/export.pkl'
folder_path_fault = 'fabrix/models/fault.pkl'

learn = load_learner(folder_path)
classes = ['Defective', 'Non Defective']

learn_fault = load_learner(folder_path_fault)
classes_fault = ['Foreign Bodies on Tissue', 'Holes and Cuts', 'Oil Stains', 'Thread Error']

# import numpy as np
# import traceback
# import pickle
# import pandas as pd






# with open('./fabrix/models/Fast_ai.pkl', 'rb') as f:
#    learn = pickle.load(f)
#    classes = learn.data.classes
 
# with open('models/model_columns.pkl', 'rb') as f:
#    model_columns = pickle.load(f)
 
 

 
# @app.route('/', methods=['POST','GET'])
# @app.route('/predict', methods=['POST','GET'])
# def predict():
   
#     if flask.request.method == 'POST':
#        try:
#            json_ = request.form.to_dict()
#            print(json_)
#            query_ = pd.get_dummies(pd.DataFrame(json_, index = [0]), prefix=['job_state','Sector','job_sim'], columns=['job_state','Sector','job_sim'])
#            print(query_)
#            query = query_.reindex(columns = model_columns, fill_value= 0)
#            print(query)
#            prediction = list(regressor.predict(query))
#            final_val = round(prediction[0],2)
 
#            #return jsonify({
#            #    "prediction":str(prediction)
#            #})
#            return render_template('index.html', prediction = final_val)
 
#        except:
#            return jsonify({
#                "trace": traceback.format_exc()
#                })
#     else:
#         return render_template('index.html')

# @app.route("/")
# def hello():
# 	return render_template('login.html')

@app.route("/")
def home():
	if current_user.is_authenticated:
		return redirect(url_for('upload_file'))
	return render_template('index.html')


@app.route("/about")
def about():
	return render_template('about.html')

@app.route("/working")
def working():
	return render_template('working.html')


@app.route('/register',methods=['GET','POST'])
def register():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = RegistrationForm()
	if form.validate_on_submit():
		hashed = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
		user = User(username=form.username.data,email=form.email.data,password=hashed)
		db.session.add(user)
		db.session.commit()
		flash(f'Account created for {form.username.data}!','success')
		return redirect(url_for('login'))

	return render_template('register.html',title='Register',form=form)


@app.route('/login',methods=['GET','POST'])
def login():
	if current_user.is_authenticated:
		return redirect(url_for('home'))
	form = LoginForm()
	if form.validate_on_submit():
		user = User.query.filter_by(email=form.email.data).first()
		if user and bcrypt.check_password_hash(user.password,form.password.data):

			login_user(user,remember=form.remember.data)
			return redirect(url_for('upload_file'))
		else:
			flash('Login unsucessful','danger')

	return render_template('login.html',title='Login',form=form)


@app.route('/logout')
def logout():
	logout_user()
	return redirect(url_for('login'))




@app.route('/upload',methods=['GET','POST'])
@login_required
def upload_file():
	
	if request.method == 'POST':
		f = request.files['file']
		fname = f.filename
		basedir = os.path.abspath(os.path.dirname(__file__))
		file_path = os.path.join(
            basedir, 'static/uploads', secure_filename(f.filename))
		f.save(file_path)
		#run model here and return the result of classification 
		prediction = learn.predict(file_path)
		probs_list = prediction[2].numpy()
		print(probs_list)
		pred = {
        'category': classes[prediction[1].item()],
        'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}
		}
		if pred['category'] == 'Defective':
			prediction_fault = learn_fault.predict(file_path)
			probs_list_fault = prediction_fault[2].numpy()
			fault = {
				'category': classes_fault[prediction_fault[1].item()],
	        	'probs': {c: round(float(probs_list_fault[i]), 5) for (i, c) in enumerate(classes_fault)}
			}
			context = {
				'pred':pred,
				'fault':fault,
				'img':fname

			}
		else:
			context = {
				'pred':pred,
				'img':fname

			}
		print(context)
		return render_template('upload.html',title='Fabrix', context = context)
	
    
	return render_template('upload.html',title='Fabrix')


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)


def model_predict(img_path):
    """
       model_predict will return the preprocessed image
    """
   
    img = open_image(img_path)
    pred_class,pred_idx,outputs = learn.predict(img)
    return pred_class

# def open_image(fn):
#     """ Opens an image using OpenCV given the file path.

#     Arguments:
#         fn: the file path of the image

#     Returns:
#         The image in RGB format as numpy array of floats normalized     to range between 0.0 - 1.0
#     """
#     flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
#     if not os.path.exists(fn):
#         raise OSError('No such file or directory: {}'.format(fn))
#     elif os.path.isdir(fn):
#         raise OSError('Is a directory: {}'.format(fn))
#     else:
#         res = np.array(Image.open(fn), dtype=np.float32)/255
#         if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
#         return res
#         try:
#             im = cv2.imread(str(fn), flags).astype(np.float32)/255
#             if im is None: raise OSError(f'File not recognized by opencv: {fn}')
#             return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#         except Exception as e:
#             raise OSError('Error handling image at: {}'.format(fn)) from e
