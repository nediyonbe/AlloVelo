from flask import Flask
app = Flask(__name__)
# Following code block is for adding a video w/ autoplay and loop
# <div align="right"; width: 100%; height: 100%">
#   <iframe frameborder="0" height="40%" width="40%" 
#     <iframe width="560" height="315" src="https://www.youtube.com/embed/p_Bxo91QSi4?controls=0I?&autoplay=1&loop=1&playlist=p_Bxo91QSi4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
#   </iframe>
# </div>
@app.route('/')
def hello_world():
    return '''
		<!DOCTYPE html>
		<html>
		<head>
		<title>AlloVélo</title>
		</head>
    <body>
		<h1 align = "center">
		Welcome to AlloVélo
		</h1>
		<p align = "center">
    You will be able to see the impact of temperature on Bixi demand on this site. Enjoy!</p>
    
    <div align="left">
			<a href="https://plot.ly/~nediyonbe/2/?share_key=cMta0BmBCnmea2AN5WfoWy" target="_blank" title="Bixi_Stations_over_Years_File" style="display: block; text-align: left;"><img src="https://plot.ly/~nediyonbe/2.png?share_key=cMta0BmBCnmea2AN5WfoWy" alt="Bixi_Stations_over_Years_File" style="max-width: 100%;width: 800px;"  width="800" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
			<script data-plotly="nediyonbe:2" sharekey-plotly="cMta0BmBCnmea2AN5WfoWy" src="https://plot.ly/embed.js" async></script>
		</div>
		</html>
    </body>
		'''
if __name__ == '__main__':
    app.run(debug=True)