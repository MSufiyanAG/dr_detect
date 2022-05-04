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

class KneeOsteoarthritisXray_Model:
    def KneeOsteoarthritisXray_Predict(self):
        option = st.radio('',
                              ['Textual Explanation', 'Model', 'Visual Explanation'])
        st.write(
                '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        st.write(":heavy_minus_sign:" * 34)

        if option == 'Textual Explanation':
            st.header('OUTLINE')
            ko_description = st.expander('Description')
            ko_description.write('''
                                    Osteoarthritis, commonly known as wear-and-tear arthritis,
                                    is a condition in which the natural cushioning between joints -- cartilage -- wears away.
                                    When this happens, the bones of the joints rub more closely against one another with less of the shock-absorbing benefits of cartilage.
                                    The rubbing results in pain, swelling, stiffness, decreased ability to move and, sometimes, the formation of bone spurs.
                                    ''')
            ko_symptoms = st.expander('Symptoms')
            ko_symptoms.write('''
                                Symptoms of osteoarthritis of the knee may include:
                                -   Pain that increases when you are active, but gets a little better with rest
                                -   Swelling
                                -   Feeling of warmth in the joint.
                                -   Stiffness in the knee, especially in the morning or when you have been sitting for a while.
                                -   Decrease in mobility of the knee, making it difficult to get in and out of chairs or cars, use the stairs, or walk.
                                -   Creaking, crackly sound that is heard when the knee moves.
                                ''')
            ko_causes = st.expander('Causes')
            ko_causes.write('''
                                Almost everyone will eventually develop some degree of osteoarthritis.
                                
                                However, several factors increase the risk of developing significant arthritis at an earlier age.
                                -   Age
                                -   Weight
                                -   Heredity 
                                -   Gender
                                -   Repetitive stress injuries
                                ''')
            ko_treatment = st.expander('Treatment')
            ko_treatment.write('''
                                The primary goals of treating osteoarthritis of the knee are to relieve the pain and return mobility.
                                
                                The treatment plan will typically include a combination of the following:
                                -   Weight loss.
                                -   Exercise
                                -   Pain relievers and anti-inflammatory drugs
                                -   Injections of corticosteroids or hyaluronic acid into the knee.
                                -   Physical and occupational therapy.
                                -   Surgery. 
                                ''')

        if option == 'Model':

            kneeClasses = ['Normal', 'Doubtful',
                            'Mild', 'Moderate', 'Severe']
            kneeModel = load_model('models/kneeClassifier.h5')

            st.title("Welcome to KNEE XRAY CLassifier")
            st.header("Identify what's the XRAY result!")

            if st.checkbox("These are the classes of Knee it can identify"):
                st.write(kneeClasses)

            file = st.file_uploader("Please upload a Knee XRAY Image", type=[
                                    "jpg", "png", "jpeg"])
            

            if file is None:
                st.info("Please upload an Image file")
            else:
                img_size = 256
                file_bytes = np.asarray(
                    bytearray(file.read()), dtype=np.uint8)
                opencv_image = cv2.imdecode(file_bytes, 1)
                st.image(opencv_image, channels="BGR",use_column_width=True)
                gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (img_size, img_size))
                i = img_to_array(resized)/255.0
                i = i.reshape(1, img_size, img_size, 1)
                #result = kneeModel.predict_classes(i)
                pred = kneeModel.predict(i)
                result = np.argmax(pred)

                if result == 0:
                    st.write("Prediction : Normal")

                elif result == 1:
                    st.write("Prediction : Doubtful")

                elif result == 2:
                    st.write("Prediction : Mild")

                elif result == 3:
                    st.write("Prediction : Moderate")

                elif result == 4:
                    st.write("Prediction : Severe")


            ## nested button with session states
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked=False                
            def callback():
                st.session_state.button_clicked=True
            if (st.button('Predict on Sample Images',on_click=callback) or st.session_state.button_clicked):
                dirs=['Normal','Doubtful','Mild','Moderate','Severe']
                classChoice=st.selectbox("Class:",dirs)

                if classChoice=='Normal':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/knee/normal/n1.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/knee/normal/n1.png'
                            #predict_sample_img(file_path_selected)
                            st.write('Prediction : Normal')
                            
                    with col2:              
                        st.image("images/sample_images/knee/normal/n2.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/knee/normal/n2.png'
                            #predict_sample_img(file_path_selected)
                            st.write('Prediction : Normal')
                        
                    with col3:
                        st.image("images/sample_images/knee/normal/n3.png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/knee/normal/n3.png'
                            #predict_sample_img(file_path_selected)
                            st.write('Prediction : Normal')

                if classChoice=='Doubtful':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/knee/doubtful/d1.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/knee/doubtful/d1.png'
                            #predict_sample_img(file_path_selected)
                            st.write('Prediction : Doubtful')
                            
                    with col2:              
                        st.image("images/sample_images/knee/doubtful/d2.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/knee/doubtful/d2.png'
                            #predict_sample_img(file_path_selected)
                            st.write('Prediction : Doubtful')
                        
                    with col3:
                        st.image("images/sample_images/knee/doubtful/d3.png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/knee/doubtful/d3.png'
                            #predict_sample_img(file_path_selected)
                            st.write('Prediction : Doubtful')

                if classChoice=='Mild':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/knee/mild/mi1.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/knee/mild/mi1.png'
                            #predict_sample_img(file_path_selected)
                            st.write('Prediction : Mild')
                            

                if classChoice=='Moderate':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/knee/moderate/mo1.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/knee/moderate/mo1.png'
                            #predict_sample_img(file_path_selected)
                            st.write('Prediction : Moderate')

                    with col2:                           
                        st.image("images/sample_images/knee/moderate/mo2.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/knee/moderate/mo2.png'
                            #predict_sample_img(file_path_selected)
                            st.write('Prediction : Moderate')

                    with col3:                           
                        st.image("images/sample_images/knee/moderate/mo3.png")
                        if st.button('Predict on this image',key="3"):
                            file_path_selected='images/sample_images/knee/moderate/mo3.png'
                            #predict_sample_img(file_path_selected)
                            st.write('Prediction : Moderate')
                        

                if classChoice=='Severe':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/knee/severe/SevereG4 (19).png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/knee/severe/SevereG4 (19).png'
                            #predict_sample_img(file_path_selected)
                            st.write('Prediction : Severe')
                            
                    with col2:              
                        st.image("images/sample_images/knee/severe/SevereG4 (20).png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/knee/severe/SevereG4 (20).png'
                            #predict_sample_img(file_path_selected)
                            st.write('Prediction : Severe')
                        
                    with col3:
                        st.image("images/sample_images/knee/severe/SevereG4 (22).png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/knee/severe/SevereG4 (22).png'
                            #predict_sample_img(file_path_selected)  
                            st.write('Prediction : Severe')

        if option == 'Visual Explanation':
            st_player("https://www.youtube.com/watch?v=BBqjltHNOrc")

    