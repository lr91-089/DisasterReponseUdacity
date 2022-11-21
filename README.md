# Disaster Response Pipeline Project

### Instructions for the Udacity Workspace:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/DRAdaBoostclassifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Instructions to run from personal computer:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/DRAdaBoostclassifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to 'http://0.0.0.0:3001/' in your browser.

### Project Components

1. ETL Pipeline

The Python script ```process_data.py``` contains a data cleaning pipeline that:
	#Loads the messages and categories datasets
    #Merges the two datasets
    #Cleans the data
    #Stores it in a SQLite database

2. ML Pipeline

The script ```train_classifier.py``` builds a machine learning pipeline that:
	#Loads data from the SQLite database
    #Splits the dataset into training and test sets
    #Builds a text processing and machine learning pipeline
    #Trains and tunes a model using GridSearchCV
    #Outputs results on the test set
    #Exports the final model as a pickle file

3. Flask Web App

The flask web app visualizes a summary of the data in the database created by the ETL pipeline and used the model created by the ML pipeline to make predictions of the categories of a message inputted by the user.

### Project File Structure

1. app
 	#templates
 		#go.html
    	#master.html
 	#run.py
2. data
	#DisasterResponse.db
    #ETL Pipeline Preparation.ipynb
    #disaster_categories.csv
    #disaster_messages.csv
    #process_data.py
3. models
	#DRAdaBoostclassifier.pkl
    #ML Pipeline Preparation.ipynb
    #text_utils.py
    #train_classifier.py
    
The first folder containts the python code for the flask web app and the html templates.

The second folder 'data' contains the data processing stuff. It contains the database, the python notebook for the data exploration and processing part, the two data sets used in this project and the process-data.py script, which runs all steps done in the notebook to load and store the datasets into one sql alchemy database.

The third folder 'models' contains the pickled model, the notebook to create and test the model, an util python file for the nlp text transformation and the train_classifier.py which executes all steps into one file to create and train an AdaBoost classifier on the data set of this project.
