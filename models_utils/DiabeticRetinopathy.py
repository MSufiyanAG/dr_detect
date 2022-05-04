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

class DiabeticRetinopathy_Model:
    def DiabeticRetinopathy_Predict(self):
        option = st.radio('',
                              ['Textual Explanation', 'Model', 'Visual Explanation'])
        st.write(
                '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        st.write(":heavy_minus_sign:" * 34)

        if option == 'Textual Explanation':

            st.header('OUTLINE')
            dr_description = st.expander('Description')

            dr_description.write('''
                        Diabetic retinopathy is a diabetes complication that affects eyes.
                        It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).
                        ''')
            dr_description.write(
                    'There are two types of diabetic retinopathy:')
            dr_description.write('''
                        -   **Early diabetic retinopathy** :
                            In this more common form — called nonproliferative diabetic retinopathy (NPDR) — new blood vessels aren't growing (proliferating).

                            When you have NPDR, the walls of the blood vessels in your retina weaken. 
                            Tiny bulges protrude from the walls of the smaller vessels, sometimes leaking fluid and blood into the retina.
                            Larger retinal vessels can begin to dilate and become irregular in diameter as well. 

                        -   **Advanced diabetic retinopathy** :
                            Diabetic retinopathy can progress to this more severe type,
                            known as proliferative diabetic retinopathy. 

                            In this type, damaged blood vessels close off, causing the growth of new, abnormal blood vessels in the retina.
                            These new blood vessels are fragile and can leak into the clear, jellylike substance that fills the center of your eye

                        ''')

            dr_symptoms = st.expander('Symptoms')
            dr_symptoms.write('''
                                You might not have symptoms in the early stages of diabetic retinopathy.
                                As the condition progresses, you might develop:
                                ''')
            dr_symptoms.write('''
                                -   Spots or dark strings floating in your vision (floaters)
                                -   Blurred vision
                                -   Fluctuating vision
                                -   Dark or empty areas in your vision
                                -   Vision loss

                                ''')

            dr_causes = st.expander('Causes')
            dr_causes.write('''
                                Over time, too much sugar in your blood can lead to the blockage of the tiny blood vessels that nourish the retina,
                                cutting off its blood supply.
                                As a result, the eye attempts to grow new blood vessels.
                                But these new blood vessels don't develop properly and can leak easily.
                                ''')
            dr_causes.image('images/web_images/dr_causes.png')

            dr_prevention = st.expander('Prevention')
            dr_prevention.write('''
                                    You can't always prevent diabetic retinopathy.
                                    However, regular eye exams, good control of your blood sugar and blood pressure,
                                    and early intervention for vision problems can help prevent severe vision loss.
                                    ''')
            dr_prevention.write('''
                                    -   Manage your diabetes. 
                                    -   Ask your doctor about a glycosylated hemoglobin test.
                                    -   Keep your blood pressure and cholesterol under control. 
                                    -   If you smoke or use other types of tobacco, ask your doctor to help you quit. 
                                    ''')

        if option == 'Model':

            retinaClasses = ['NO DR', 'Mild', 'Moderate',
                                 'Severe', 'Proliferative Diabetic Retinopathy']
            retinaModel = load_model('models/densenet_.h5')

            st.title("Welcome to Diabetic-Retinopathy CLassifier")
            st.header("Identify what's the type of retina!")

            if st.checkbox("These are the classes of Diabetic Retinopathy"):
                st.write(retinaClasses)

            file = st.file_uploader("Please upload a retina Image", type=[
                                        "jpg", "png", "jpeg"])

            if file is None:
                st.info("Please upload an Image file")
            else:
                image = Image.open(file)
                st.image(image, use_column_width=True)
                demo = np.array(image)
                demo = demo[:, :, ::-1].copy()
                    #demo = cv2.imread(image)
                demo = tf.image.convert_image_dtype(demo, tf.float32)
                demo = tf.image.resize(demo, size=[300, 300])
                demo = np.expand_dims(demo, axis=0)
                pred = retinaModel.predict(demo)
                result = np.argmax(pred)
                st.write(result)

                if result == 0:
                        # no
                    st.write("Prediction : No DR")
                elif result == 1:
                        # MIld
                    st.write("Prediction : Mild")
                elif result == 2:
                        # moderate
                    st.write("Prediction : Moderate")
                elif result == 3:
                        # severe
                    st.write("Prediction : Severe")
                elif result == 4:
                        # proliferative
                    st.write("Prediction :  Proliferative DR")

                        
            def predict_sample_img(file_path_selected):
                image = Image.open(file_path_selected)
                #st.image(image, use_column_width=True)
                demo = np.array(image)
                demo = demo[:, :, ::-1].copy()
                #demo = cv2.imread(image)
                demo = tf.image.convert_image_dtype(demo, tf.float32)
                demo = tf.image.resize(demo, size=[300, 300])
                demo = np.expand_dims(demo, axis=0)
                pred = retinaModel.predict(demo)
                result = np.argmax(pred)
                st.write(result)

                if result == 0:
                    # no
                    st.write("Prediction : No DR")
                elif result == 1:
                    # MIld
                    st.write("Prediction : Mild")
                elif result == 2:
                    # moderate
                    st.write("Prediction : Moderate")
                elif result == 3:
                    # severe
                    st.write("Prediction : Severe")
                elif result == 4:
                    # proliferative
                    st.write("Prediction :  Proliferative_DR")


            ## nested button with session states
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked=False                
            def callback():
                st.session_state.button_clicked=True
            if (st.button('Predict on Sample Images',on_click=callback) or st.session_state.button_clicked):
                dirs=['No_DR','Mild','Moderate','Severe','Proliferative_DR']
                classChoice=st.selectbox("Class:",dirs)

                if classChoice=='No_DR':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/DR/No_DR/00cc2b75cddd.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/DR/No_DR/00cc2b75cddd.png'
                            predict_sample_img(file_path_selected)
                            
                    with col2:              
                        st.image("images/sample_images/DR/No_DR/01f7bb8be950.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/DR/No_DR/01f7bb8be950.png'
                            predict_sample_img(file_path_selected)
                        
                    with col3:
                        st.image("images/sample_images/DR/No_DR/0097f532ac9f.png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/DR/No_DR/0097f532ac9f.png'
                            predict_sample_img(file_path_selected)

                if classChoice=='Mild':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/DR/Mild/00cb6555d108.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/DR/Mild/00cb6555d108.png'
                            predict_sample_img(file_path_selected)
                            
                    with col2:              
                        st.image("images/sample_images/DR/Mild/04ac765f91a1.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/DR/Mild/04ac765f91a1.png'
                            predict_sample_img(file_path_selected)
                        
                    with col3:
                        st.image("images/sample_images/DR/Mild/0124dffecf29.png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/DR/Mild/0124dffecf29.png'
                            predict_sample_img(file_path_selected)

                if classChoice=='Moderate':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/DR/Moderate/000c1434d8d7.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/DR/Moderate/000c1434d8d7.png'
                            predict_sample_img(file_path_selected)
                            
                    with col2:              
                        st.image("images/sample_images/DR/Moderate/01c7808d901d.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/DR/Moderate/01c7808d901d.png'
                            predict_sample_img(file_path_selected)
                        
                    with col3:
                        st.image("images/sample_images/DR/Moderate/012a242ac6ff.png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/DR/Moderate/012a242ac6ff.png'
                            predict_sample_img(file_path_selected)

                if classChoice=='Severe':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/DR/Severe/070f67572d03.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/DR/Severe/070f67572d03.png'
                            predict_sample_img(file_path_selected)
                        

                if classChoice=='Proliferative_DR':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/DR/Proliferative_DR/034cb07a550f.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/DR/Proliferative_DR/034cb07a550f.png'
                            predict_sample_img(file_path_selected)
                            
                    with col2:              
                        st.image("images/sample_images/DR/Proliferative_DR/0083ee8054ee.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/DR/Proliferative_DR/0083ee8054ee.png'
                            predict_sample_img(file_path_selected)
                        
                    with col3:
                        st.image("images/sample_images/DR/Proliferative_DR/02685f13cefd.png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/DR/Proliferative_DR/02685f13cefd.png'
                            predict_sample_img(file_path_selected)            
                

        if option == 'Visual Explanation':
            st_player('https://www.youtube.com/watch?v=X17Q_RPUlYo')
