# AlloVelo

AlloVelo is a program looking at past data of bike share program of the city of Montreal (called Bixi hereafter to predict Bixi demand on a given day/hr. 

While the prophet method is used for the final model, other methods have also been tested. They are available  in the 'alternative models' folder and cover methods such as random forest and autoregression.

Data is provided by the city of Montreal. Originally an attempt had been made to use weather info as well. As it is found to have limited impact on final demand, they are not taken into account in the final model for the sake of simplicity. Should you prefer to use, hourly data is available on the Government of Canada's website.

The final model using prophet makes use only of the pickup demand info, using a time series approach. The final model makes use of the following files:
model check prophet.py: makes validation of the model, pickles the data.
Bixi.py: trains the model with given split date and pickles the model for every cluster
Graph_creator.py: unpickles the model, makes the prediction for the 24hours following the date given by the end user via the web app
Webby.py: creates the web app
index.html: designs the web app (see the templates folder)

A screenshot of the web app is shown below. The user...
- enters the date on the website for which he/she wants a pickup estimate the map of Montréal (selected date is shown on the left bottom corner here)
- sets the desired time within the day using the slider at the bottom of the graph (each step corresponds to an hour within day)
- hovers on the stations to see the station id, neignborhood id, pickup estimation and relative demand change.
The relative demand change gives the change with respect to the last 2 years' pickup realizations at the given weekday and month.
The highest relative demand change is shown by stations with red color.

Different map visuals are also possible: User can select one of the alternatives via the menu on the bottom left corner. In this view, Dark option is selected. 

![](2019-04-20-14-04-06.png)

Alternative view, namely the 'Light' option selected in this screenshot. Satellite and outdoor options are also available
![](2019-04-20-14-07-05.png)

## Installation

Use Python along with the following modules:
pandas / glob / numpy / scikit / sklearn / scipy

Note that alternative models may require additional modules. Other utility modules such as pickle are also not mentioned in the list above.

## Usage
The main use case considered is Bixi's daily operations. As different bike stations have varying demands at different times, additional bikes are transported by trucks to empty stations. A proactive demand prediction is key to better planning the route of trucks.

To use it you can connect to allovelo.info A request made on this website is treated by using the last trained model.

Additionally, bixi users can use the result to see what the demand will likely to be in their closest station on an hourly basis. That would give them heads up on what time they should be going to pick up their bike

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)