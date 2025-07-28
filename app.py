import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import imutils
from PIL import Image
import os

st.title("🎨 Dominant Color Extractor")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

clusters = st.slider("Select number of dominant colors", 1, 5, 5)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1) 

    org_img = img.copy()

    img = imutils.resize(img, height=200)

    flat_img = np.reshape(img, (-1, 3))

    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(flat_img)
    dominant_colors = np.array(kmeans.cluster_centers_,dtype='uint')

    percentages = np.bincount(kmeans.labels_) / len(kmeans.labels_)
    p_and_c = sorted(zip(percentages, dominant_colors), reverse=True)

    st.subheader("Dominant Colors")
    block = np.ones((50, 50, 3), dtype='uint8')
    color_columns = st.columns(clusters)
    for i in range(clusters):
        block[:] = p_and_c[i][1][::-1]  
        color_columns[i].image(block, channels="RGB", caption=f"{round(p_and_c[i][0]*100, 2)}%")

    bar = np.ones((50, 500, 3), dtype='uint8')
    start = 0
    for i, (p, c) in enumerate(p_and_c):
        end = start + int(p * bar.shape[1])
        if i == clusters - 1:
            bar[:, start:] = c[::-1]
        else:
            bar[:, start:end] = c[::-1]
        start = end

    st.image(bar, channels="RGB", caption="Proportions of Colors")

    rows = 1000
    cols_ratio = int((org_img.shape[0] / org_img.shape[1]) * rows)
    img = cv2.resize(org_img, dsize=(rows, cols_ratio), interpolation=cv2.INTER_LINEAR)

    copy = img.copy()
    cv2.rectangle(copy, (rows//2 - 250, cols_ratio//2 - 90), (rows//2 + 250, cols_ratio//2 + 110), (255, 255, 255), -1)
    final = cv2.addWeighted(img, 0.1, copy, 0.9, 0)
    cv2.putText(final, 'Most Dominant Colors in the Image', (rows//2 - 230, cols_ratio//2 - 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)



    start = rows // 2 - 220
    for i in range(clusters):
        end = start + 70
        final[cols_ratio//2:cols_ratio//2 + 70, start:end] = p_and_c[i][1]  
        cv2.putText(final, str(i+1), (start+25, cols_ratio//2 + 45),cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        start = end + 20

    final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    st.image(final_rgb, caption="Final Image with Dominant Color Palette", use_container_width=True)

    if st.button("Save Output Image"):
        output_path = "output.png"
        cv2.imwrite(output_path, final)
        with open(output_path, "rb") as file:
            st.download_button(label="Download Output Image", data=file, file_name="output.png", mime="image/png")
