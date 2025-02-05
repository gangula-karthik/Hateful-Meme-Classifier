hateful-meme-classifier
==============================
End to end deep learning project building a hateful meme classifier and deploying it


### Project Structure
- assets: contains some images and project proposal
- data: contains the datasets
- notebooks: contains all the notebooks: 
    - data_preparation folder: 
        - data_preparation.ipynb: all the data processing code
        - gen_ai_image_inpainting.ipynb: attempting to use gen ai for image inpainting task
        - image_inpatining.ipynb: the simpler image inpainting function that was used
        - processing_test_set.ipynb: data processing function for test set
        - smol-VLM_ocr_extraction: performing ocr on the webscraped images
    - modelling folder: 
        - modelling.ipynb: contains the modelling, hyperparameter tuning and evaluation code
- Dockerfile: used to deploy the webserver
- server folder: The code for the fastAPI server along with requirements.txt for environment
- model_checkpoints:
    - best_model.tflite: the model that I am using
    - v1 folder: this contains the old model that I was using (not required anymore)