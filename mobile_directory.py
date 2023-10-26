import io
import streamlit as st
import sqlite3
from PIL import Image

# Create SQLite Database and Table
conn = sqlite3.connect("mobile_directory.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS contacts (
        id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        mobile_number TEXT,
        address TEXT,
        profile_pic BLOB
    )
''')
conn.commit()

st.title("Mobile Number Storage Directory")

# Input Form
st.subheader("Add Contact")
name = st.text_input("Name")
email = st.text_input("Email")
mobile_number = st.text_input("Mobile Number")
address = st.text_input("Address")
profile_pic = st.file_uploader("Profile Picture", type=["jpg", "png"])

if st.button("Add Contact"):
    cursor.execute(
        "INSERT INTO contacts (name, email, mobile_number, address, profile_pic) VALUES (?, ?, ?, ?, ?)",
        (name, email, mobile_number, address, profile_pic.read() if profile_pic else None),
    )
    conn.commit()
    st.success("Contact added successfully!")

# Display Contacts
st.subheader("Contacts")
cursor.execute("SELECT name, email, mobile_number, address, profile_pic FROM contacts")
contacts = cursor.fetchall()

if not contacts:
    st.info("No contacts added yet.")
else:

    for contact in contacts:
        st.markdown("""
        |Name|Email|Contact|Adress|Profile Pic|
        |-|-|-|-|-|
        """)
        st.markdown(f"|{contact[0]}|{contact[1]}|{contact[2]}|{contact[3]}|{st.image(Image.open(io.BytesIO(contact[4])), use_column_width='never', output_format='JPEG', width=50)}|")       

conn.close()
