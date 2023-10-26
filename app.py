from io import BytesIO
import tempfile
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date, time
import base64
import subprocess
import cv2
from pytube import YouTube



st.title("Working With Streamlit")
st.markdown("---")
st.header("Header")
st.subheader("Subheader")
st.text("Text line")
st.write("Write Line")

st.markdown("---")
st.header("Variable declaration")
name = "sarfaraz"
st.write("Hello! ", name)

st.markdown("""---
            Use the **streamlit run app.py** command in the working directory to run the streamlit app.
            """)

st.markdown("---")
st.code("""
import streamlit as st

st.title("Welcome")
st.header("Main header")
st.subheader("subheader")
st.markdown("markdown")
st.text("Text")
st.write("Write")
       
name = "sarfaraz"
st.write("Hello! ", name)
        
""")

st.markdown("---")
st.header("How To Work With DataFrames")

df = pd.DataFrame(
    np.random.randn(50,20),
    columns= ('col %d' % (i+1) for i in range(20))            
    )
st.dataframe(df)

st.markdown("---")
st.header("How To Work With Tables")
tbl = pd.DataFrame(
    np.random.randn(10,5),
    columns= ('col %d' % (i+1) for i in range(5))            
    )
st.table(tbl)

st.markdown("---")
st.header("How To Work With JSON data")
st.json({"data" : {"channel_name":"Learn With Newton", "channel_url":"https://www.youtube.com/@LearnWithNewton"}})

st.markdown("---")
st.header("How To Work With Line Chart")
# Create a sample data frame with random data
data = pd.DataFrame({
    'x': np.arange(1, 11),
    'y': np.random.randn(10)
})
# Add the line chart to your app
st.line_chart(data.set_index('x'))

data2 = pd.DataFrame({
    'x': np.arange(1, 11),
    'y1': np.random.randn(10),
    'y2': np.random.randn(10),
    'y3': np.random.randn(10)
})
# Add the multiline chart to your app
st.line_chart(data2.set_index('x')[['y1', 'y2', 'y3']])

st.markdown("---")
st.header("How To Work With Area Chart")
# Create some sample data for a single area chart
data3 = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=20, freq='D'),
    'Value': np.random.randint(-10, 10, size=20)
})


st.subheader('Single Area Chart')
st.area_chart(data3.set_index('Date'))

# Create multiple data sets for multiple area charts
num_charts = 3
chart_data = []
for i in range(num_charts):
    data = {
        'Date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        f'Value_{i}': np.random.randint(-5, 5, size=10)
    }
    chart_data.append(data)

# Create a DataFrame to hold the combined data
combined_data = pd.DataFrame({'Date': chart_data[0]['Date']})

# Display multiple area charts
for i in range(num_charts):
    combined_data[f'Value_{i}'] = chart_data[i][f'Value_{i}']
st.subheader('Multi Area Chart')
st.area_chart(combined_data.set_index('Date'))

st.markdown("---")
st.header("How To Work With Bar Chart")

st.subheader('Single Bar Chart Example')
st.bar_chart(pd.DataFrame({'Values': [10, 25, 5, 15]}))

st.subheader('Multi-Bar Chart Example')
# Sample data
data = {'Category': ['A', 'B', 'C', 'D'],
        'Values_1': [10, 5, 15, 15],
        'Values_2': [15, 10, 10, 5]
        }

df = pd.DataFrame(data)
# Create a multi-bar chart using matplotlib
fig, ax = plt.subplots()
bar_width = 0.35
bar1 = ax.bar(df['Category'], df['Values_1'], bar_width, label='Values 1')
bar2 = ax.bar(df['Category'], df['Values_2'], bar_width, label='Values 2', bottom=df['Values_1'])
ax.set_xlabel('Category')
ax.set_ylabel('Values')
ax.set_title('Multi-Bar Chart')
ax.legend()

st.pyplot(fig)

st.markdown("---")
st.header("How To Work With Pyplot")
# Generate some random data for the histogram
data = np.random.randn(1000)

# Set up the Streamlit app
st.subheader('Histogram Example')
st.write('This is a simple example of a histogram plot in Streamlit.')

# Create a histogram using matplotlib
fig, ax = plt.subplots()
ax.hist(data, bins=50, color='red', alpha=0.5)
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.set_title('Histogram Plot')

# Display the plot in the Streamlit app
st.pyplot(fig)

st.markdown("---")
st.header("How To Work With Button Widget")

if st.button('Say Hello'):
    st.write('Hi !', name)
else:
    st.write('Goodbye ',name)


st.markdown("---")
st.header("How To Work With Download Buttons")

simpletxt = "This is a simple text line."
st.download_button('Text File Download', simpletxt, "file.txt")

with open("./files/image.jpeg","rb") as f:
    btn = st.download_button (
        label = "Image Download",
        data = f,
        file_name="photo.jpg",
        mime = "image/jpg"
    )

@st.cache_data

def convertDataFrame(df):
    return df.to_csv().encode("utf-8")

df1 = pd.read_excel("./files/SampleData.xlsx")
csv = convertDataFrame(df1)

st.download_button(
    label = "CSV File Download",
        data = csv,
        file_name="data.csv",
        mime = "text/csv"
)


st.markdown("---")
st.header("How To Work With Check Box")
agree = st.checkbox("Agree to the terms and conditions!")
if agree:
    st.write("Agreed Terms & Conditions!")

st.markdown("---")
st.header("How To Work With Radio Button")
choice = st.radio("Select your favourite movie genre!", ("Horror","Comedy","Thriller","Documentary"))
st.write("You selected " , choice)

st.markdown("---")
st.header("How To Work With Select Box")
choice = st.selectbox("Select your favourite movie genre!", ("Horror","Comedy","Thriller","Documentary"))
st.write("You selected " , choice)


st.markdown("---")
st.header("How To Work With Basic Slider")
value = st.slider("Select a value", 0.0, 100.0, 50.0)
st.write(f"Selected value: {value}")


st.header("How To Work With Integer Slider") 
value = st.slider("Select an integer value", 0, 100, 50)
st.write(f"Selected value: {value}")


st.header("How To Work With Float Slider with Step")
value = st.slider("Select a value with step", 0.0, 1.0, 0.1, 0.01)
st.write(f"Selected value: {value}")


st.header("How To Work With Date Slider")
start_date = date(2023, 1, 1)
end_date = date(2023, 12, 31)
selected_date = st.slider("Select a date", start_date, end_date)
st.write(f"Selected date: {selected_date}")


st.header("How To Work With Time Slider")
start_time = time(9, 0)
end_time = time(18, 0)
selected_time = st.slider("Select a time", start_time, end_time)
st.write(f"Selected time: {selected_time}")


st.header("How To Work With Range Slider")
start_range, end_range = st.slider("Select a range", 0.0, 100.0, (25.0, 75.0))
st.write(f"Selected range: {start_range} to {end_range}")


st.header("How To Work With Multiple Sliders")
value1 = st.slider("Slider 1", 0.0, 100.0, 50.0)
value2 = st.slider("Slider 2", 0.0, 100.0, 25.0)
st.write(f"Slider 1 value: {value1}")
st.write(f"Slider 2 value: {value2}")


st.markdown("---")
st.header("How To Work With Text Input")
text_input = st.text_input("Enter text", "Default Text")
st.write(f"You entered: {text_input}")

st.header("How To Work With Password Input")
password = st.text_input("Enter Password", type="password")
st.write(f"You entered a password: {password}")

st.header("How To Work With Text Area Input")
text_area = st.text_area("Enter a long text", "Default long text")
st.write(f"You entered: {text_area}")

st.header("How To Work With Number Input")
number = st.number_input("Enter a number", min_value=0, max_value=100, value=50)
st.write(f"You entered: {number}")

st.header("How To Work With Date Input")
selected_date = st.date_input("Pick a date", value=None)
st.write(f"You picked: {selected_date}")

st.header("How To Work With Time Input")
selected_time = st.time_input("Pick a time", value=None)
st.write(f"You picked: {selected_time}")

st.header("How To Work With File Uploader")
uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt"])
if uploaded_file is not None:
    file_contents = uploaded_file.read()
st.write(f"You uploaded: {file_contents}")

st.header("How To Work With Color Picker")
selected_color = st.color_picker("Pick a color", "#ff0000")
st.write(f"You picked color: {selected_color}")


st.markdown("---")
### Image Upload Example:
st.header("Image Upload Example")

# Upload an image file
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # You can also process the uploaded image, e.g., convert it to grayscale
    from PIL import Image
    image = Image.open(uploaded_image)
    grayscale_image = image.convert("L")
    st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)

### Image Download Example:
st.header("Image Download Example")

# Provide a URL for image download
image_url = st.text_input("Enter the image URL")
if st.button("Download Image"):
    response = requests.get(image_url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image.save("downloaded_image.png")
        st.success("Image saved as downloaded_image.png")
        # Display the downloaded image
        st.image(response.content, caption="Downloaded Image", use_column_width=True)

        # Add a download button to save the image
        # if st.button("Save Image"):
        #     image = Image.open(BytesIO(response.content))
        #     image.save("downloaded_image.png")
        #     st.success("Image saved as downloaded_image.png")
    else:
        st.error("Failed to download the image. Please check the URL.")


### Image Conversion Example (e.g., to Black and White):
st.header("Image Conversion Example")
# Upload an image file
#uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Original Image", use_column_width=True)

    # Perform image conversion (e.g., to black and white)
    image = Image.open(uploaded_image)
    converted_image = image.convert("1")
    st.image(converted_image, caption="Black and White Image", use_column_width=True)


st.header("Convert Image to Different Format")
uploaded_image = st.file_uploader("Upload an image", type=["png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    converted_image = image.convert("RGB")
    st.image(converted_image, caption="Converted Image", use_column_width=True)

st.header("Save Image to Disk")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # Perform image processing (e.g., resizing)
    resized_image = image.resize((300, 300))

    st.image(resized_image, caption="Processed Image", use_column_width=True)

    if st.button("Save Processed Image"):
        resized_image.save("processed_image.jpg")
        st.success("Image saved as processed_image.jpg")

st.header("Download Image")
if st.button("Show Image"):
    st.markdown(f"![Download Image]({image_url})")



st.markdown("---")
st.header("Video Upload and Display")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])

if uploaded_video is not None:
    st.subheader("Uploaded Video")
    st.video(uploaded_video)

st.header("Video Download")
video_path = "./files/example.mp4"  # Path to your video file
with open(video_path, "rb") as video_file:
    video_bytes = video_file.read()
    st.subheader("Download Video")
    st.markdown(
        f'<a href="data:video/mp4;base64,{base64.b64encode(video_bytes).decode()}" download="example.mp4">Download video</a>',
        unsafe_allow_html=True,
    )

st.header("Video Conversion")
#uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_video is not None:
    st.subheader("Uploaded Video")
    st.video(uploaded_video)

    # Convert the video to a different format
    if st.button("Convert to AVI"):
        # Assuming FFmpeg is installed
        conversion_command = f"ffmpeg -i './files/example.mp4' 'output.avi"
        subprocess.run(conversion_command, shell=True)
        st.success("Video converted to AVI format.")

st.header("Video Saving")
#uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_video is not None:
    st.subheader("Uploaded Video")
    st.video(uploaded_video)

    save_path = "saved_video.mp4"
    if st.button("Save Video"):
        with open(save_path, "wb") as f:
            f.write(uploaded_video.read())
        st.success(f"Video saved to {save_path}")

st.header("Video Processing with External Libraries")
#uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_video is not None:
    st.subheader("Uploaded Video")
    st.video(uploaded_video)

    # Save the uploaded video to a temporary file
    temp_video_path = os.path.join(tempfile.gettempdir(), "uploaded_video.mp4")
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

    video_capture = cv2.VideoCapture(temp_video_path)

    st.subheader("Processed Video")

    # Create a temporary file to save the processed video
    temp_output_path = os.path.join(tempfile.gettempdir(), "cv2_processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4 format
    out = cv2.VideoWriter(temp_output_path, fourcc, 30.0, (640, 480))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # You can perform any processing on the frame here
        # For example, you can apply filters or effects

        out.write(frame)

    out.release()

    # Display the processed video
    st.video(temp_output_path)


st.header("youtube Video Downloader")

# Input field for the video URL
video_url = st.text_input("Enter the URL of the video to download:")

if st.button("Download"):
    if video_url:
        try:
            yt = YouTube(video_url)

            # Choose the highest resolution stream
            stream = yt.streams.get_highest_resolution()

            # Set a file path for the downloaded video
            download_path = os.path.join("downloads", yt.title + ".mp4")

            # Download the video
            stream.download(output_path="downloads", filename=yt.title + ".mp4")

            st.success(f"Downloaded video: {yt.title}")
            st.text(f"File saved as: {download_path}")

            # Provide a link to download the video
            st.markdown(f"Download your video [here](downloads/{yt.title}.mp4)")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a valid video URL.")


st.sidebar.markdown("---")
# Create a sidebar for navigation
st.sidebar.title("Working With Sidebar")

# Define menu options
menu_options = ["Home", "Data Exploration", "Data Preprocessing", "Model Training", "Model Evaluation"]

# Create a radio button to select the menu option
selected_menu = st.sidebar.radio("Select an option", menu_options)

# Load data for demonstration
@st.cache_data
def load_data():
    data = pd.DataFrame({
        'Feature 1': [1, 2, 3, 4, 5],
        'Feature 2': [10, 20, 30, 40, 50],
        'Target': [0, 1, 0, 1, 0]
    })
    return data

data = load_data()

# Define functions for each menu option
def show_home():
    st.title("Welcome to the Machine Learning App")
    st.write("Use the sidebar to navigate through different options.")

def show_data_exploration():
    st.title("Data Exploration")
    st.write("You can explore the dataset and visualize data here.")
    st.dataframe(data)

def show_data_preprocessing():
    st.title("Data Preprocessing")
    st.write("This section allows you to preprocess the data.")

def show_model_training():
    st.title("Model Training")
    st.write("Train machine learning models here.")

def show_model_evaluation():
    st.title("Model Evaluation")
    st.write("Evaluate model performance and make predictions.")

# Display the selected menu option
if selected_menu == "Home":
    show_home()
elif selected_menu == "Data Exploration":
    show_data_exploration()
elif selected_menu == "Data Preprocessing":
    show_data_preprocessing()
elif selected_menu == "Model Training":
    show_model_training()
elif selected_menu == "Model Evaluation":
    show_model_evaluation()


st.sidebar.markdown("---")
# Create a sidebar for navigation
st.sidebar.title("Working With Sidebar2")

# Define menu options and their corresponding functions
menu_options = {
    "Home": None,
    "Data Exploration": "data_exploration",
    "Data Preprocessing": "data_preprocessing",
    "Model Training": "model_training",
    "Model Evaluation": "model_evaluation"
}

# Create a selectbox to select the menu option
selected_menu = st.sidebar.selectbox("Select an option", list(menu_options.keys()))

# Load data for demonstration
@st.cache_data
def load_data():
    data = pd.DataFrame({
        'Feature 1': [1, 2, 3, 4, 5],
        'Feature 2': [10, 20, 30, 40, 50],
        'Target': [0, 1, 0, 1, 0]
    })
    return data

data = load_data()

# Define functions for each menu option
def data_exploration():
    st.title("Data Exploration")
    st.write("You can explore the dataset and visualize data here.")
    st.dataframe(data)

def data_preprocessing():
    st.title("Data Preprocessing")
    st.write("This section allows you to preprocess the data.")

def model_training():
    st.title("Model Training")
    st.write("Train machine learning models here.")

def model_evaluation():
    st.title("Model Evaluation")
    st.write("Evaluate model performance and make predictions.")

# Display the selected menu option
if selected_menu:
    function_name = menu_options[selected_menu]
    if function_name:
        globals()[function_name]()
    else:
        st.title("Welcome to the Machine Learning App")
        st.write("Use the sidebar to navigate through different options.")


st.markdown("---")
st.header("Streamlit Column Layout")
# Define data for your columns
column_data = [
    {
        "header": "Column 1",
        "subheader": "Subheader 1",
        "image_url": "https://via.placeholder.com/150",
        "image_caption": "Image 1",
        "button_label": "Click Me 1",
        "star_rating": 4,
    },
    {
        "header": "Column 2",
        "subheader": "Subheader 2",
        "image_url": "https://via.placeholder.com/150",
        "image_caption": "Image 2",
        "button_label": "Click Me 2",
        "star_rating": 3,
    },
    {
        "header": "Column 3",
        "subheader": "Subheader 3",
        "image_url": "https://via.placeholder.com/150",
        "image_caption": "Image 3",
        "button_label": "Click Me 3",
        "star_rating": 5,
    },
]

# Create columns
columns = st.columns(len(column_data))

# Fill in the columns with content
for i, column in enumerate(columns):
    data = column_data[i]
    column.subheader(data["subheader"])
    column.image(data["image_url"], caption=data["image_caption"])
    
    # Ensure each button has a unique key based on its index
    button_key = f"button_{i}"
    if column.button(data["button_label"], key=button_key):
        st.write(f"You clicked {data['header']}")
    
    column.text(f"Star Rating: {data['star_rating']} 💓")


st.markdown("---")
st.header("expander in the main content area")
# Create an expander in the main content area
with st.expander("Expand in Main Content Area"):
    st.write("This is the content inside the expander.")
    st.write("You can add more elements here.")

st.sidebar.markdown("---")
st.sidebar.header("expander in thesidebar")
# Create an expander in the sidebar
with st.sidebar.expander("Expand in Sidebar"):
    st.write("This is the content inside the sidebar expander.")
    st.write("You can add more elements here.")


























# st.header("How To Work With Buttons and Files")

# # Function to enable file downloads
# def download_file(file_path):
#     with open(file_path, 'rb') as file:
#         file_contents = file.read()
#     return file_contents

# # Directory containing the files
# file_directory = 'files'

# # List of files in the directory
# file_list = os.listdir(file_directory)

# # Dropdown to select the file type
# file_type = st.selectbox("Select a file type to download:", ["Text", "Image", "CSV", "PDF"])

# # Dropdown to select the file to download
# selected_file = st.selectbox("Select a file to download:", file_list)

# if file_type == "Text":
#     st.markdown(f"**Selected Text File:** {selected_file}")
#     if st.button("Download Text File"):
#         file_path = os.path.join(file_directory, selected_file)
#         file_data = download_file(file_path)
#         st.download_button("Download Text File", file_data, key='text')

# elif file_type == "Image":
#     st.markdown(f"**Selected Image File:** {selected_file}")
#     if st.button("Download Image File"):
#         file_path = os.path.join(file_directory, selected_file)
#         file_data = download_file(file_path)
#         st.image(file_data, use_container_width=True)
#         st.download_button("Download Image File", file_data, key='image')

# elif file_type == "CSV":
#     st.markdown(f"**Selected CSV File:** {selected_file}")
#     if st.button("Download CSV File"):
#         file_path = os.path.join(file_directory, selected_file)
#         file_data = download_file(file_path)
#         st.download_button("Download CSV File", file_data, key='csv', key_display="Download CSV")

# elif file_type == "PDF":
#     st.markdown(f"**Selected PDF File:** {selected_file}")
#     if st.button("Download PDF File"):
#         file_path = os.path.join(file_directory, selected_file)
#         file_data = download_file(file_path)
#         st.download_button("Download PDF File", file_data, key='pdf', key_display="Download PDF")

