# Creating a Dataset
The most effective way to create a dataset is to compile a .csv file of images, inputs, and a correct answer. 

For example, the Kvasir-VQA dataset is composed of an image column, question column, and answer column. For a Visual Question Answering task, this format should be maintained.

## Guide for dataset content
Ensure that images are homologated to 896x896 or 448x448 resolution for MedGemma. With the current setup, images are downscaled to 448x448 resolution. This means that if initial images are not square,
distortion could result. Also, large images could increase CPU load during processing and take up excessive storage space. 

In addition, ensure that all text columns (input and output) are formatted in the same way with minimal spelling issues. If the output has inconsistent formatting like leading or trailing spaces,
spelling errors, etc. the model can learn these behaviors and may be less useful.

## Guide for creation
There are several ways to make a `.csv` file, though other formats like JSON are also acceptable. Please review other guides to accomplish this process, including [https://huggingface.co/docs/datasets/en/create_dataset]. 

After compiling the dataset, the next step is to upload it to Hugging Face following this guide [https://huggingface.co/docs/datasets/en/upload_dataset]. Using Hugging Face to store datasets
is effective, as they can be loaded from any platform or device easily. If you wish to avoid having this data public, you can set the dataset as private. If you set your dataset as private, you
do need to include your Hugging Face access token to retrieve it. 

To host a dataset, upload your CSV/JSON file, and set the dataset availability to private so only you/your organization can access it (assuming you don't want to share your dataset publicly). Then, in 
the setup/config section, change the dataset name variable to "your_username(or organization username)/dataset_name" and the column/split names to your dataset column or split names.
> Note: A split refers to a section of a dataset. For example, "train" and "test" are two common dataset splits

In future updates, direct loading may be added in an automated way, but for now using Hugging Face is the easiest way to store, manage, and retrieve datasets.
