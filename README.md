A face disparity checker that calculates the mean disparity with a picture you upload against the databases of images you have. 

The Process:

1) A folder data having 2 folders(pic and pic_aligned)

2) The folder pic_aligned has the aligned pictures from pic

3) dataset_preparation.py, face_embeddings.py and face_Alignment.py are helper functions and are being used in disparity.py

4) Face alignment and noise removal is done as part of the alignment and extraction process.

5) To run:
	Disparity.py which will calculate the disparity scores for 2 pictures depending on what we pass as arguments.
  
	The disparity right now is based on cosine. Euclidean distance is also being computed. 

6) If running the script directly, need to install libraries and change corresponding directory paths inside the helper functions and main module and also have the corresponding data images stored.



![](https://github.com/rakash/images1/blob/master/mean_dip.png?raw=true)
