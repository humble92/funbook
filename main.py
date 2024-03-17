# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from PIL import UnidentifiedImageError
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import logging
import requests
import streamlit as st
import os

load_dotenv(find_dotenv())

# env. variables
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")
API_BASE_URL = 'https://api-inference.huggingface.co/models/'
headers = {"Authorization": f"Bearer {HF_HUB_TOKEN}"}


def home():
    st.title("Fun books on your fingertip")
    st.write('Make fun books (audio book or picture book) using only the first sentence or the first picture. '
             'You do not need to give a detailed information to complete that.'
             'AI helps you complete the rest of your story.')

    st.subheader("Genre", divider='rainbow')
    style = st.selectbox(
        'Which style would you like to draw?',
        ('Fantasy', 'Fairy tales', 'satirical', 'Realism', 'Educational', 'Experimental'))
    st.write('You selected:', style)

    st.subheader("Source", divider='rainbow')
    tab1, tab2 = st.tabs(["🖼️ Image", "🖹 Text"])

    # print_hi('hugging couples in the desert')

    with tab1:
        st.write('Make your audio book using only one picture. '
                 'AI will complete the rest of your story for you. Enjoy yourself!')

        generate_pages_by_picture(style)

    with tab2:
        st.write('Make your audio book using only the first sentence. '
                 'AI will complete the rest of your story for you. '
                 'Sometime it also generates picture matching the story. Have fun!'
                 )

        generate_pages_by_text(style)


def generate_pages_by_text(style):
    # st.write("Pages")
    # pages = st.slider('How many pictures would you like to draw?', 1, 4, 1)
    # st.write('You selected: ', pages, ' pages picture book.')
    # Fix to 1 for pages
    pages = 1

    st.write("Beginning story")
    input_text = st.text_area("What do you want to start your story from?")
    if st.button("Generate", type="secondary"):
        with st.spinner('Thinking and processing...'):
            story = ""

            # Generate a picture book
            for x in range(pages):
                story_piece = generating_story(input_text, style)

                try:
                    story_image = text2image(story_piece)
                    st.image(story_image, caption='Generated Image.', use_column_width=True)
                except UnidentifiedImageError:
                    # When it fails to draw pictures, just skip and go ahead
                    print("Temporary failure to draw pictures....")
                    pass

                input_text = story_piece
                story += story_piece

            # Generate an audio book
            audio_bytes = text2speech(story)

            with st.expander("story"):
                st.write(story)

            st.audio(audio_bytes)


def generate_pages_by_picture(style):
    uploaded_picture = st.file_uploader("Upload a picture.", type=['png', 'jpg'])

    if uploaded_picture:
        logging.info(uploaded_picture)

        bytes_data = uploaded_picture.getvalue()
        st.image(uploaded_picture, caption='Uploaded Image.', use_column_width=True)

        with st.spinner('Thinking and processing...'):
            full_path = os.path.join('uploads', uploaded_picture.name)
            with open(full_path, "wb") as file:
                file.write(bytes_data)

            image_text = img2txt(full_path)
            story = generating_story(image_text, style)

            audio_bytes = text2speech(story)

            with st.expander("image text"):
                st.write(image_text)
            with st.expander("story"):
                st.write(story)

            st.audio(audio_bytes)


def img2txt(path):
    # pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    # text = pipe(path)[0]["generated_text"]
    # logging.info("Extracted text:", text)
    # return text
    model_name = "Salesforce/blip-image-captioning-large"
    API_URL = f'{API_BASE_URL}{model_name}'

    with open(path, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    if len(response.json()):
        generated_text = response.json()[0]["generated_text"]
        logging.info("Generated text:", generated_text)
        return generated_text.replace('"', '')


def generating_story(context, style):
    text_divider = '----'
    prompt = f"""
    Your role is a story teller.
    The context is as follows: {context}
    Generate a short story under 30 words with style of {style}.
    {text_divider}
    """
    logging.info(prompt)
    model_name = "tiiuae/falcon-7b-instruct"
    # Needs HF pro membership to use
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    API_URL = f'{API_BASE_URL}{model_name}'
    payload = {
        "inputs": prompt
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    print("Generated text output:", response.json())
    try:
        if len(response.json()) and len(response.json()[0]["generated_text"].split(text_divider)):
            generated_text = response.json()[0]["generated_text"].split(text_divider)[1]
            logging.info("Generated text:", generated_text)
            return generated_text.replace('"', '')
    except KeyError:
        # Error: It should not reach this line in normal cases.
        return context


def text2speech(input_text):
    model_name = "espnet/kan-bayashi_ljspeech_vits"
    API_URL = f'{API_BASE_URL}{model_name}'
    payload = {
        "inputs": input_text
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    full_path = os.path.join('output', 'audio.flac')
    with open(full_path, 'wb') as file:
        file.write(response.content)

    return response.content


def text2image(input_text):
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    API_URL = f'{API_BASE_URL}{model_name}'
    payload = {
        "inputs": input_text
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    full_path = os.path.join('output', 'image.png')
    with open(full_path, 'wb') as file:
        file.write(response.content)

    return response.content


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    home()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
