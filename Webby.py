# import m
from flask import Flask
from flask import render_template, request, flash, redirect, url_for
from Graph_creator import pickup_estimator
from flask_wtf import FlaskForm
from wtforms import DateTimeField, IntegerField, SubmitField, validators

app = Flask(__name__)

app.config['SECRET_KEY'] = 'b39b860b15b7e3a4b0cc201bc45ff268'
# Following code block is for adding a video w/ autoplay and loop
# <div align="right"; width: 100%; height: 100%">
#   <iframe frameborder="0" height="40%" width="40%" 
#     <iframe width="560" height="315" src="https://www.youtube.com/embed/p_Bxo91QSi4?controls=0I?&autoplay=1&loop=1&playlist=p_Bxo91QSi4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
#   </iframe>
# </div>


@app.route('/')
def home_page():
    form = SubmitStuff()
    return render_template('index.html', form=form)  # render a template


@app.route('/', methods=['POST'])
def home_page_post():
    entrydate = request.form['entrydate']
    pickup_estimator(entrydate)
    return redirect(url_for('home_page'))




class SubmitStuff(FlaskForm):
    entrydate = DateTimeField('Date (YYYY-MM-DD)', format='%Y-%m-%d')
    submit = SubmitField('Show Me')

if __name__ == '__main__':
    app.run(debug=True)
