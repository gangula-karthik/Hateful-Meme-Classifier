Hateful Meme Classifier
==============================
End to end deep learning project building a hateful meme classifier and deploying it

### Deployment
Both the frontend and the backend have been deployed by me on google cloud platform: 
Link to website: https://frontend-315644372008.us-central1.run.app/
Link to harmful meme backend: https://hateful-meme-classifier-315644372008.us-central1.run.app/


### Project Structure
```
Inside the Harmful-Meme-Classifier folder (deep learning model + fastAPI server): 
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
```


### Datasets
- Due to the large size of the datasets, it has been uploaded on kaggle: 
  - Processed training dataset: https://kaggle.com/datasets/d7fda92983d1e97946751ba8a58b26b883485f15c6ff7c4c3b62bfe6e0b60718
  - Processed testing dataset: https://kaggle.com/datasets/b58a8121d1f4135b6979ce74a9a64dae21c279a667c5e28b518dadf54271a091