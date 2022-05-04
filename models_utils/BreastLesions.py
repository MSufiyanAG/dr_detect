import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player
import numpy as np
from bokeh.models.widgets import Div
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

class BreastLesions_Model:
    def BreastLesions_Predict(self):
        def import_and_predict(image_data, model):

            size = (256, 256)
            image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
            image = np.asarray(image)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

            img_reshape = img[np.newaxis, ...]
            prediction = model.predict(img_reshape)
            # pred=np.argmax(prediction)
            return prediction
        option = st.radio('',
                              ['Textual Explanation', 'Model', 'Visual Explanation'])
        st.write(
                '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        st.write(":heavy_minus_sign:" * 34)

        if option == 'Textual Explanation':
            st.header('OUTLINE')
            bl_description = st.expander('Description')
            bl_description.write('''
                                    Breast cancer is cancer that forms in the cells of the breasts.

                                    After skin cancer, breast cancer is the most common cancer diagnosed in women in the United States. 
                                    
                                    Breast cancer can occur in both men and women, but it's far more common in women.
                                    ''')
            bl_symptoms = st.expander('Symptoms')
            bl_symptoms.write('''
                                Signs and symptoms of breast cancer may include:
                                -   A breast lump or thickening that feels different from the surrounding tissue
                                -   Change in the size, shape or appearance of a breast
                                -   Changes to the skin over the breast, such as dimpling
                                -   A newly inverted nipple
                                -   Peeling, scaling, crusting or flaking of the pigmented area of skin surrounding the nipple (areola) or breast skin
                                -   Redness or pitting of the skin over your breast, like the skin of an orange
                                ''')
            bl_causes = st.expander('Causes')
            bl_causes.write('''
                                Factors that are associated with an increased risk of breast cancer include:
                                -   Being female. 
                                -   Increasing age. 
                                -   A personal history of breast conditions.
                                -   A personal history of breast cancer. 
                                -   A family history of breast cancer.
                                -   Radiation exposure.
                                -   Obesity. 
                                -   Beginning your period at a younger age. 
                                -   Having your first child at an older age.
                                -   Having never been pregnant. 
                                -   Drinking alcohol. 
                                ''')
            bl_prevention = st.expander('Prevention')
            bl_prevention.write('''
                                    Making changes in your daily life may help reduce your risk of breast cancer. Try to:
                                    -   Ask your doctor about breast cancer screening. 
                                    -   Become familiar with your breasts through breast self-exam for breast awareness. 
                                    -   Exercise most days of the week. 
                                    -   Maintain a healthy weight. 
                                    -   Choose a healthy diet. 
                                    -   Preventive medications (chemoprevention). 
                                    -   Preventive surgery
                                    ''')

        if option == 'Model':

            st.title("Welcome to Breast Lesions Tumor CLassifier")
            st.header("Identify what's the result!")
            breastModel = load_model('models/BreastCancerSegmentor.h5')

            file = st.file_uploader("Please upload a Breast ultrasound Image", type=[
                                    "jpg", "png", "jpeg"])

            if file is None:
                st.info("Please upload an Image file")
            else:
                image = Image.open(file)
                st.image(image, use_column_width=True)
                predictions = import_and_predict(image, breastModel)
                # Plotter(predictions)
                st.image(predictions)
                # st.write(predictions)

            def predict_sample_img(file_path_selected):
                image = Image.open(file_path_selected)
                #st.image(image, use_column_width=True)
                predictions = import_and_predict(image, breastModel)
                # Plotter(predictions)
                st.write('Lesions found at:')
                st.image(predictions)
                # st.write(predictions)


            ## nested button with session states
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked=False                
            def callback():
                st.session_state.button_clicked=True
            if (st.button('Predict on Sample Images',on_click=callback) or st.session_state.button_clicked):
                dirs=['Normal','Benign','Malignant']
                classChoice=st.selectbox("Class:",dirs)

                if classChoice=='Normal':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/breast/normal/normal (1).png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/breast/normal/normal (1).png'
                            predict_sample_img(file_path_selected)
                            
                    with col2:              
                        st.image("images/sample_images/breast/normal/normal (2).png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/breast/normal/normal (2).png'
                            predict_sample_img(file_path_selected)
                        
                    with col3:
                        st.image("images/sample_images/breast/normal/normal (4).png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/breast/normal/normal (4).png'
                            predict_sample_img(file_path_selected)

                if classChoice=='Benign':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/breast/benign/benign (2).png")
                        if st.button('Predict on this image',key="1"):                                
                            file_path_selected="images/sample_images/breast/benign/benign (2).png"
                            predict_sample_img(file_path_selected)
                            
                    with col2:              
                        st.image("images/sample_images/breast/benign/benign (25).png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected="images/sample_images/breast/benign/benign (25).png"
                            predict_sample_img(file_path_selected)
                        
                    with col3:
                        st.image("images/sample_images/breast/benign/benign (17).png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected="images/sample_images/breast/benign/benign (17).png"
                            predict_sample_img(file_path_selected)

                if classChoice=='Malignant':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/breast/malignant/malignant (21).png")
                        if st.button('Predict on this image',key="1"):                                
                            file_path_selected="images/sample_images/breast/malignant/malignant (21).pngg"
                            predict_sample_img(file_path_selected)
                            
                    with col2:              
                        st.image("images/sample_images/breast/malignant/malignant (210).png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected="images/sample_images/breast/malignant/malignant (210).png"
                            predict_sample_img(file_path_selected)
                        
                    with col3:
                        st.image("images/sample_images/breast/malignant/malignant (39).png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected="images/sample_images/breast/malignant/malignant (39).png"
                            predict_sample_img(file_path_selected)               

        if option == 'Visual Explanation':
            st_player('https://www.youtube.com/watch?v=ql11xKFMKg4')