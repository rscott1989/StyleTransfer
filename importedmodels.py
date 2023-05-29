from fastai.vision.all import *
import streamlit as st
import PIL
import os 

PATH_TO_CWD = os.getcwd()

input_image_file = st.file_uploader("Upload an Input Image",type=['png','jpeg','jpg'])
input_file_path = ''

# Check if an input image is uploaded
if input_image_file is not None:

    # Create the file path for the input image
    input_file_path = f'{PATH_TO_CWD}/tmp/input_tmp.{input_image_file.type[-4:]}'
    file_details = {"FileName":input_image_file.name,"FileType":input_image_file.type}
    st.write(file_details)

    # Load and display the input image
    img = load_image(input_image_file)
    st.image(img)

    # Save the input image to the specified file path
    with open(input_file_path,"wb") as f: 
      f.write(input_image_file.getbuffer())         
    st.success("Saved Input Image File")

    style_image_file = st.file_uploader("Upload a Style Image",type=['png','jpeg','jpg'])
    style_file_path = ''

    # Check if a style image is uploaded
    if style_image_file is not None:

        # Create the file path for the style image
        style_file_path = f'{PATH_TO_CWD}/tmp/style_tmp.{style_image_file.type[-4:]}'
        file_details = {"FileName":style_image_file.name,"FileType":style_image_file.type}
        st.write(file_details)

        # Load and display the style image
        img = load_image(style_image_file)
        st.image(img)
        
        # Save the style image to the specified file path
        with open(style_file_path,"wb") as f: 
          f.write(style_image_file.getbuffer())         
        st.success("Saved Style Image File")

        # Create a dataset from the style image
        dset = Datasets(style_file_path, tfms=[PILImage.create])
        dl = dset.dataloaders(after_item=[ToTensor()], after_batch=[IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)])
        dl.show_batch(figsize=(7,7))
        style_im = dl.one_batch()[0]

        # Create a dataset from the input image
        dset = Datasets(input_file_path, tfms=[PILImage.create])
        dl = dset.dataloaders(after_item=[ToTensor()], after_batch=[IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)])
        dl.show_batch(figsize=(7,7))
        input_im = dl.one_batch()[0]

        # Load the pre-trained model
        learn = load_learner(f'{PATH_TO_CWD}/TrainedModels/uncorrupted.pkl', cpu=False)

        # Process the input image with the model
        with torch.no_grad():
          res = learn.model(input_im)

        # Convert the result to a numpy array
        img_np = res.squeeze().permute(1, 2, 0).cpu().numpy()

        # Normalize the pixel values to [0, 1]
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # Convert the numpy array to a PIL image
        img_pil = Image.fromarray((img_np * 255).astype('uint8'))

        # Remove the previous generated image if it exists
        if os.path.isfile(f'{PATH_TO_CWD}/generated_images/foo.jpg'):
          os.remove(f'{PATH_TO_CWD}/generated_images/foo.jpg')

        # Save the generated image
        img_pil.save(f'{PATH_TO_CWD}/generated_images/foo.jpg')

        try:
          # Try to display the generated image using Streamlit
          st.image(img_pil)
        except:
          # If an error occurs, open the generated image using PIL and display it
          image = PIL.Image.open(f'{PATH_TO_CWD}/generated_images/foo.jpg')
          st.image(image, caption='Generated image')