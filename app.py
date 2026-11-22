
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from google import genai

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Flower Classification + AI Assistant",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Flower Classification Using CNN")
st.subheader("CNN + MobileNetV2 + Gemini LLM Assistant")

st.write(
    "Upload a flower image, predict its category "
    "and explore it using our AI-powered flower assistant."
)

# ---------------- LOAD CNN MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "flower_finetuned_model.keras",
        compile=False
    )

model = load_model()

# ---------------- LOAD CLASS NAMES ----------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ---------------- GEMINI CLIENT ----------------
@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)

api_key = st.secrets.get("GEMINI_API_KEY", "")

# ---------------- IMAGE UPLOAD ----------------
st.divider()
st.header("1. Upload Flower Image")

uploaded_file = st.file_uploader(
    "Choose a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    if st.session_state.get("image_name") != uploaded_file.name:
        st.session_state.image_name = uploaded_file.name
        st.session_state.prediction = None
        st.session_state.messages = []

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Flower",
            use_container_width=True
        )

    with col2:
        st.write("### CNN Prediction")

        if st.button("Classify Flower"):

            img = image.resize((224, 224))

            img_array = np.array(
                img,
                dtype=np.float32
            )

            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(
                img_array,
                verbose=0
            )

            index = np.argmax(predictions[0])

            flower = class_names[index]

            confidence = float(
                predictions[0][index] * 100
            )

            st.session_state.prediction = flower
            st.session_state.confidence = confidence
            st.session_state.scores = predictions[0].tolist()
            st.session_state.messages = []

        if st.session_state.get("prediction"):

            st.success(
                f"Predicted Flower: {st.session_state.prediction}"
            )

            st.metric(
                "Prediction Confidence",
                f"{st.session_state.confidence:.2f}%"
            )

            st.write("### Top 3 Predictions")

            scores = np.array(st.session_state.scores)

            for i in np.argsort(scores)[::-1][:3]:
                st.write(
                    f"{class_names[i]}: {scores[i]*100:.2f}%"
                )
                st.progress(float(scores[i]))

# ---------------- LLM FLOWER ASSISTANT ----------------
st.divider()
st.header("2. Gemini AI Flower Assistant")

if st.session_state.get("prediction"):

    flower = st.session_state.prediction

    st.info(f"AI Assistant is ready to discuss: {flower}")

    def ask_gemini(question):

        history = st.session_state.messages

        conversation = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in history
        ])

        prompt = f"""
        You are an expert Flower Information Assistant.

        The CNN model predicted this flower: {flower}.

        Answer questions about this flower, including:
        - Flower characteristics
        - Scientific name
        - Uses and benefits
        - Growing conditions
        - Watering and sunlight
        - Soil requirements
        - Interesting facts

        Be beginner-friendly and give clear answers.

        If asked about something unrelated to flowers,
        politely guide the user back to the flower topic.

        Previous conversation:
        {conversation}

        User question: {question}
        """

        client = get_gemini_client(api_key)

        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )

        return response.text

    # Display previous chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    question = st.chat_input(
        f"Ask something about {flower}..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            if not api_key:
                answer = (
                    "Gemini API key is missing. "
                    "Please configure .streamlit/secrets.toml."
                )
            else:
                try:
                    with st.spinner("Gemini is thinking..."):
                        answer = ask_gemini(question)

                except Exception as e:
                    answer = f"Gemini API error: {e}"

            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:
    st.warning(
        "Please upload a flower image and classify it "
        "to activate the Gemini AI Assistant."
    )

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "Developed using TensorFlow, CNN, MobileNetV2, "
    "Transfer Learning, Fine-Tuning, Gemini LLM and Streamlit."
)
