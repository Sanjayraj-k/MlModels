from streamlit_option_menu import option_menu
import streamlit as st
import joblib
import matplotlib.pyplot as plt
sentiment_mpdel=joblib.load(open('sentiment_model.pkl','rb'))
vectorizer = joblib.load("tfidf_vectorizer.pkl")
with st.sidebar:
    st.title("Machine Learning Models")
    selected = option_menu(
    "Select a ML model To Predict", 
    ['Sentiment Analysis', 'SleepTime Prediction'],icons=["emoji-smile-fill", "alarm-fill" ], 
    default_index=0
)
if selected=="Sentiment Analysis":
    st.title("Sentiment Analysis")
    text=st.text_area("Enter the Sentence to predict the sentiment")
    if st.button("Predict"):
        loaded_model = joblib.load("/workspaces/MlModels/app/sentiment_model.pkl")
        vectorizer = joblib.load("/workspaces/MlModels/app/tfidf_vectorizer.pkl")
    #processed_input = clean_text(text)
    #vectorized_input = vectorizer.transform([processed_input])
        prediction = loaded_model.predict(vectorizer.transform([text]))

        st.success((f"Prediction: {prediction[0]}"))
        if prediction[0]=='positive':
            pos=100
            neg=0
            neu=0
            sent={'positive':pos,'negative':neg,'neutral':neu}
            plt.bar(sent.keys(),sent.values())
            plt.xlabel('Sentiments')    
            plt.ylabel('Count') 
            plt.title('Sentiment Analysis')
            st.pyplot(plt)
        elif prediction[0]=="negative":
            pos=0
            neg=100
            neu=0
            sent={'positive':pos,'negative':neg,'neutral':neu}
            plt.bar(sent.keys(),sent.values())
            plt.xlabel('Sentiments')    
            plt.ylabel('Count') 
            plt.title('Sentiment Analysis')
            st.pyplot(plt)
        else:
            pos=0
            neg=0
            neu=100
            sent={'positive':pos,'negative':neg,'neutral':neu}
            plt.bar(sent.keys(),sent.values())
            plt.xlabel('Sentiments')    
            plt.ylabel('Count') 
            plt.title('Sentiment Analysis')
            st.pyplot(plt)
else:
    st.title("SleepTime Analysis")
    col1,col2=st.columns(2)
    with col1:
        work=st.number_input("Enter your Workout Time",min_value=0.0,step=0.1,max_value=10.0)
    with col2:
        study=st.number_input("Enter your Study Time",min_value=0.0,step=0.1,max_value=10.0)
    col3,col4=st.columns(2)
    with col3:
        travel=st.number_input("Enter your Phone usage Time",min_value=0.0,step=0.1,max_value=10.0)
    with col4:
        workhours=st.number_input("Enter your Work Hours Time",min_value=0.0,step=0.1,max_value=10.0)
    col5,col6=st.columns(2)
    with col5:
        CaffeineIntake=st.number_input("Enter your CaffeineIntake",min_value=0.0,step=0.1,max_value=300.0)
    with col6:
        RelaxationTime=st.number_input("Enter your Relaxation Time",min_value=0.0,step=0.1,max_value=10.0)
    predict=st.button("Predict Sleep Time")
    if predict:
        loaded_model = joblib.load("app/sleep.pkl")
        prediction = loaded_model.predict([[work,study,travel,workhours,CaffeineIntake,RelaxationTime]])
        st.success(f"Predicted Sleep Time is {prediction[0]} hours")
        sleep={'Sleep Time':prediction[0]}
        
    
        
        


