import pickle
#from tensorflow.keras.models import load_model
from flask import Flask,render_template,request

from tensorflow.keras.models import load_model
#from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
model=load_model('sentient_model1.h5')
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
  
app = Flask(__name__)
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])


def predict():
    if request.method == 'POST':
        message = request.form['message']
        x_1 = tokenizer.texts_to_sequences([message])
        x_1 = pad_sequences(x_1, maxlen=500)
        predictions = model.predict(x_1)[0][0]
    return render_template('result.html',prediction = predictions)
if __name__ == '__main__':
    
	app.run( debug = False)