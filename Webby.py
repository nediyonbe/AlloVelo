import json
from flask import Flask

app = Flask(__name__)
@app.route('/Chart')
def charty():
    return app.send_static_file('C:/Users/Ali/Documents/Insight/Bixi/Program/AlloVelo/Templates/charty/html')
