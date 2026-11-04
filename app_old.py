import streamlit as st 
st.title("MR. Analyst")
st.header("Your AI-powered data analysis companion. Upload your dataset, ask questions and discover insights.")

# we need a let sidebar (it should have title, subheader, name, fileuploader,and input for email)

st.sidebar.title("Hey User")
st.sidebar.title("Step: 1")

st.sidebar.subheader("Please provide your details below:")
st.sidebar.text_input("Name", placeholder="Enter your name")
st.sidebar.text_input("Email", placeholder="Enter your email",type="email")
st.sidebar.button("Submit", help="Click to submit your details")

st.sidebar.title("Step: 2")

st.sidebar.subheader("Upload your data file below:")
st.sidebar.file_uploader("Upload your file", type=["csv","xlsx","xls"], help="Please upload a CSV or Excel File")
st.sidebar.button("Upload", help="Click to upload your file")

# in main screen now we need a subheader to ask  user to enter any que based on the uploded file and then a input box and get answer

st.subheader("Step: 3")

st.text_input("Ask a question based on your uploaded data:", placeholder="Enter your question here")
st.button("Get Answer", help="Click to get answer based on your question and uploaded data")

st.subheader("Step: 4")
st.title("Your answer will be displayed here:")
st.subheader("Your AI Insights")

st.info(
    "Your analysis will appear here after you submit a question."
)