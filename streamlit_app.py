import streamlit as st

st.title("Mi primera aplicación Streamlit")

st.write("Hola Miguel!")

nombre = st.text_input("¿Cómo te llamas?")

if nombre:
    st.success(f"Hola {nombre}")
