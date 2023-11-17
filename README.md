# demoFaceRecog

## Setup
1. Create python virtual environment
    ```bash
    # create virtual environment
    python -m venv env

    # activate the environment
    source env/bin/activate
    ```
2. Install opencv requirement 
    ```bash
    # Install requirements.txt
    pip install -r requirements.txt
    ```

##  Collect image data and build datasets
*NOTE"  Make sure to have good camera*

3. Run datacollect.py and enter id of the user.
    ```bash
    #This id is to differentiate each users by id in an array list of name inside the main code
    Enter Your ID:
    ```
4. Run trainingdemo.py to create a yml file to store the trained  data of faces.
    ```bash 
    #Processing the datasets of a face and turn into yml file
    Training Completed............
    ``` 

## Image Recognition code
*NOTE"  Make sure to put the name of the users in the dataset. Sort by ID in an array*

5. Enter image recognition code, Modify the name_list array. number of array based on the id inside datasets.
``` bash
    #Input name of users
    name_list = ["", "Joshua",  "Vinilis", "Johnny"]
```

6. Run facial recognition code like default_Main.py
``` bash 
    #Output
    ID: 2 Confidence: 47.35577424898243
    Detected Face - Name: Vinilis
    ID: 0 Confidence: 35.25912327297898
    Detected Face - Name: Unknown
    ...
```
## Others
Kept a few other ways of coding for facial recognition.

```bash
#extra requirement may needed for code within others folder
pip install face_recognition
```