from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return '''
		<!DOCTYPE html>
		<html>
		<head>
		<title>AlloVélo</title>
		</head>
		<h1>
		Welcome to AlloVélo
		</h1>
		<p>You will be able to see the impact of temperature on Bixi demand on this site.</p>
		<p>Enjoy!</p>
		<div>
			<a href="https://plot.ly/~nediyonbe/2/?share_key=cMta0BmBCnmea2AN5WfoWy" target="_blank" title="Bixi_Stations_over_Years_File" style="display: block; text-align: center;"><img src="https://plot.ly/~nediyonbe/2.png?share_key=cMta0BmBCnmea2AN5WfoWy" alt="Bixi_Stations_over_Years_File" style="max-width: 100%;width: 800px;"  width="800" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
			<script data-plotly="nediyonbe:2" sharekey-plotly="cMta0BmBCnmea2AN5WfoWy" src="https://plot.ly/embed.js" async></script>
		</div>
		</html>
		'''

if __name__ == '__main__':
    app.run(debug=True)