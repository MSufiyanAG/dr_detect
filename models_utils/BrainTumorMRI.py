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
class BrainTumorMRI_Model:
    def BrainTumorMRI_Predict(self):
        option = st.radio('',
                              ['Textual Explanation', 'Model', 'Visual Explanation'])
        st.write(
                '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        st.write(":heavy_minus_sign:" * 34)

        if option == 'Textual Explanation':
            st.header('OUTLINE')
            bt_description = st.expander('Description')
            bt_description.write('''
                                    A brain tumor is a mass or growth of abnormal cells in your brain.

                                    Many different types of brain tumors exist.
                                    -   Some brain tumors are noncancerous (benign).
                                    -   Some brain tumors are cancerous (malignant).
                                    -   Brain tumors can begin in your brain (primary brain tumors).
                                    -   Cancer can begin in other parts of your body and spread to your brain as secondary (metastatic) brain tumors.

                                    ''')

            bt_symptoms = st.expander('Symptoms')
            bt_symptoms.write('''
                                The signs and symptoms of a brain tumor vary greatly and depend on the brain tumor's size, location and rate of growth.
                                
                                General signs and symptoms caused by brain tumors may include:
                                New onset or change in pattern of headaches
                                -   Headaches that gradually become more frequent and more severe
                                -   Unexplained nausea or vomiting
                                -   Vision problems, such as blurred vision, double vision or loss of peripheral vision
                                -   Gradual loss of sensation or movement in an arm or a leg
                                -   Difficulty with balance
                                -   Speech difficulties
                                -   Feeling very tired
                                -   Confusion in everyday matters
                                -   Difficulty making decisions
                                -   Inability to follow simple commands
                                -   Personality or behavior changes
                                -   Seizures, especially in someone who doesn't have a history of seizures
                                -   Hearing problems
                                ''')

            bt_causes = st.expander('Causes')
            bt_causes.write('''
                                **Brain tumors that begin in the brain**
                                ''')
            bt_causes.image('images/web_images/bt_causes1.png')
            bt_causes.write('''
                                Primary brain tumors originate in the brain itself or in tissues close to it,
                                such as in the brain-covering membranes (meninges), cranial nerves, pituitary gland or pineal gland.

                                Primary brain tumors begin when normal cells develop changes  in their DNA.
                                A cell's DNA contains the instructions that tell a cell what to do.
                                The mutations tell the cells to grow and divide rapidly and to continue living when healthy cells would die.
                                The result is a mass of abnormal cells, which forms a tumor.
                                ''')
            bt_causes.write('''
                                **Cancer that begins elsewhere and spreads to the brain**
                                ''')
            bt_causes.write('''
                                Secondary brain tumors are tumors that result from cancer that starts elsewhere in your body and then spreads (metastasizes) to your brain.

                                Secondary brain tumors most often occur in people who have a history of cancer.
                                Rarely, a metastatic brain tumor may be the first sign of cancer that began elsewhere in your body.

                                In adults, secondary brain tumors are far more common than are primary brain tumors.
                                ''')

            bt_prevention = st.expander('Treatment')
            bt_prevention.write('''
                                    Brain tumor treatment depends on the tumor’s location, size and type. Doctors often use a combination of therapies to treat a tumor.

                                    Your treatment options might include:
                                    Surgery
                                    -   Radiation therapy
                                    -   Chemotherapy 
                                    -   Immunotherapy 
                                    -   Targeted therapy 
                                    -   Laser thermal ablation
                                    ''')

        if option == 'Model':

            brainClasses = ['TUmor Not Found', 'TUmor Found']
            brainModel = load_model('models/brain_Classifier.h5')

            st.title("Welcome to Brain Tumor CLassifier")
            st.header("Identify what's the MRI result!")

            if st.checkbox("These are the classes of BRAIN Tumor "):
                st.write(brainClasses)

            file = st.file_uploader("Please upload a BRAIN MRI Image", type=[
                                    "jpg", "png", "jpeg"])

            if file is None:
                st.info("Please upload an Image file")
            else:
                image = Image.open(file)
                st.image(image, use_column_width=True)
                demo = np.array(image)
                demo = demo[:, :, ::-1].copy()
                demo = tf.image.convert_image_dtype(demo, tf.float32)
                demo = tf.image.resize(demo, size=[150, 150])
                demo = np.expand_dims(demo, axis=0)

                pred = brainModel.predict(demo)
                result = np.argmax(pred)
                st.write(result)
                if result == 0:
                    st.write("Prediction : Normal ")
                elif result == 1:
                    st.write("Prediction : BRAIN TUMOR FOUND")

            def predict_sample_img(file_path_selected):
                image = Image.open(file_path_selected)
                #st.image(image, use_column_width=True)
                demo = np.array(image)
                demo = demo[:, :, ::-1].copy()
                demo = tf.image.convert_image_dtype(demo, tf.float32)
                demo = tf.image.resize(demo, size=[150, 150])
                demo = np.expand_dims(demo, axis=0)

                pred = brainModel.predict(demo)
                result = np.argmax(pred)
                st.write(result)
                if result == 0:
                    st.write("Prediction : Normal ")
                elif result == 1:
                    st.write("Prediction : BRAIN TUMOR FOUND")
                                
            ## nested button with session states
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked=False                
            def callback():
                st.session_state.button_clicked=True
            if (st.button('Predict on Sample Images',on_click=callback) or st.session_state.button_clicked):
                dirs=['BRAIN_TUMOR','NO_TUMOR']
                classChoice=st.selectbox("Class:",dirs)

                if classChoice=='BRAIN_TUMOR':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/brain/BRAIN_TUMOR/Y18.JPG")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/brain/BRAIN_TUMOR/Y18.JPG'
                            predict_sample_img(file_path_selected)
                            
                    with col2:              
                        st.image("images/sample_images/brain/BRAIN_TUMOR/Y23.JPG")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/brain/BRAIN_TUMOR/Y23.JPG'
                            predict_sample_img(file_path_selected)
                        
                    with col3:
                        st.image("images/sample_images/brain/BRAIN_TUMOR/Y28.jpg")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/brain/BRAIN_TUMOR/Y28.JPG'
                            predict_sample_img(file_path_selected)

                if classChoice=='NO_TUMOR':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/brain/NO_TUMOR/11 no.jpg")
                        if st.button('Predict on this image',key="1"):                                
                            file_path_selected="images/sample_images/brain/NO_TUMOR/11 no.jpg"
                            predict_sample_img(file_path_selected)
                            
                    with col2:              
                        st.image("images/sample_images/brain/NO_TUMOR/19 no.jpg")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected="images/sample_images/brain/NO_TUMOR/19 no.jpg"
                            predict_sample_img(file_path_selected)
                        
                    with col3:
                        st.image("images/sample_images/brain/NO_TUMOR/50 no.jpg")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected="images/sample_images/brain/NO_TUMOR/No16.jpg"
                            predict_sample_img(file_path_selected)



        
        if option == 'Visual Explanation':
            st_player('https://www.youtube.com/watch?v=MnOITHXlW6U')

    