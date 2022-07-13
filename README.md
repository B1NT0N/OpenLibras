# OpenLibras
## Description
OpenLibras was created to be an easy way to learn the alphabet of Libras (Lingua Brasileira de Sinais) as well as ASL (American Sign Language). 
It uses computer aided vision to recognize and caracterize your hand gestures and the character it simbolizes.

## Instalation and Requirements
To install OpenLibras please fork this repository and install 3.9 or higher.

* ### Python link:
  https://www.python.org/downloads/
* ### Repository Link:
  ```
  git clone https://github.com/B1NT0N/OpenLibras.git
  ```
* ### Install Required External Packages
  ```
  pip install -r requirements.txt
  ```
## Usage
To use OpenLibras you must follow these 3 steps:*
* ### Collect Data
  * Run the Data Gathering Module and enter the character you want to get image data.
  ```
  python DGM.py
  ```
  * Position your hand within the camera frame and press ```S``` button on the keyboard to save the image.
  * I would recomend a bacth with a minimum of 300 images for each character you want.
  * After each batch press ```Q``` button on the keyboard to finish and exit.
  * Do this for each character.

* ### Train Data and get a Model
  * Go to https://teachablemachine.withgoogle.com/train/image.
  * Change the Classes with the respective character eg. (A, B ...) and upload each bath of images on the respective class.
  * Wait for yout model to train.
  * When done, export your model as Tensorflow/Keras Model.
  * Extract and replace the model to the "Model" folder.
  
* ### Use the Model
  * Run the Data Processing Module.**
  ```
  python DPM.py
  ```
  
*You can use OpenLibras with the Demo Model that comes default but it is only trained for the character A, B and C. Just follow the last step to use the Demo Model.

**The first time you run OpenLibras it may take some time to start, especially if you don't have a dedicated graphics card.
