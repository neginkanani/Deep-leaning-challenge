Overview
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. Here, I have used machine learning neural networks on the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

This dataset has the following information:
•	EIN
•	NAME
•	Application type
•	Affiliation
•	Classification
•	Use_case
•	Organization
•	Status
•	Income_amount
•	Special considerations
•	Ask amount
•	IS_Successfull

Based on the information provided the target to predict the performance is the "IS_Successfull" column. The rest of the columns can be the features. 
Looking deeper into the dataset, “EIN” and “Name” columns do not present valuable information and are just tags for the dataset. Therefore, they are removed from the analysis and are not considered as features. All the rest of the columns were tagged as features. 
To improve the NN performance Application_type and Classification columns were binned into groups to reduce the number of extra information. 
Next, outliers were removed from the dataset to reduce the noise in the model. 
 

After that, categorical data were converted to numeric indexes using “get_dummies” function from Pandas library.
Then, a train_test_split function from scikit-learn machine learning library was used to split the data into 25% testing and 75% training sets. Feature columns were then scaled to reduce the bias from the dataset using the “StandardScaler” function.
To pick the optimal set of hyperparameters for our TensorFlow program, Keras Tuner library was used. The Hyperparameter model suggested using 3 hidden layers each with 7, 9, and 5 neurons and using the 'gelu' activation function. The last activation function used in the output layer was 'tanh'. The NN model was then compiled trained and tested. At the end, the model was saved as a .h5 file for future use. 

Overall, the optimization steps including: 
1: removal of unnecessary columns,
2: Binning the information,
3: Removing outliers, and
4: Using Hyperparameters improved the accuracy of the model from 0.72 to 0.75.
