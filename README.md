# Kisan-CropAllot
A project entry for Hack In The North (HINT) 2020 Hackathon. <br>
Kisan CropAllot is a crop management system that allocates crops to farmers on the basis of the highest requirement crop predicted using machine learning from the crops that are suitable for their area according to the previous years crop production ,the season in which the crop is to be grown and the size of their farm-land. It consists of an App for the user, asking about their details and providing recommendations.

<br>
<br>

## Mobile Application

  - The first page after the homepage asks the user to login or register depending on their status. 
  - Registration page includes information about the user/farmer like name, Aadhar no., State, contact details. There is also a second page which asks the user for their district name, total area of farm-land and the season for which they want their crop allocated.
  - After he enters all these details, the user is guided to the page where there are two tabs, one ‘Preferred’ for the best recommendation and second ‘Others’ for closest four other crops.
  - The second tab (Others) shows all the crops that can be grown in his area.
  - When the user presses the Allocate button he/she is alloted the given crop with an ID.
  
<br>

## Machine Learning 

Dataset was obtained from <a href="https://data.gov.in/node/87630">here</a> .
- The data we got consisted of all the production of all the districts of all the states from 1997 - 2010. 
- The original dataset contains the columns for State, District, Year, Season, Crop, Area and Production.
- We decided to drop the state and year columns as they were not affecting the crop production directly.
- We then retrieved the input details of the user/farmer from Firebase Database (district, season and land area).
- Model training is done in real time.
- The modified dataframe for each input consisted of the District, season and area that they entered and another column of crops.
- The dataframe had the same values for each of the input columns but the crop column had all the crops in the dataset.
- Model Prediction was done via DecisionTree and RandomForest Regressors.
- The top 5 crops closest to the production predicted were selected.
  
  <br><br>
  
## Team: ***GrayMatters***
- [Arpit Agarwal](https://github.com/aarpit1010/)
- [Tamoghno Bhattacharya](https://github.com/TamoghnoBhattacharya)
- [Priyanshu Jain](https://github.com/priyanshu0405)
- [Smitesh Hadape](https://github.com/smitesh25)

<br>


