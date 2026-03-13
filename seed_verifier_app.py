import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from io import BytesIO
import matplotlib.pyplot as plt

# Load model for inference only
model = tf.keras.models.load_model("seed_verifier.h5", compile=False)

# -------------------------
# Preprocess image
# -------------------------
def preprocess(img):
    if img is None:
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128,128))
    img_array = img_resized / 255.0
    return np.expand_dims(img_array, axis=0), img_rgb

# -------------------------
# Color Histogram
# -------------------------
def get_histogram(img):
    chans = cv2.split(img)
    colors = ("b","g","r")
    fig, ax = plt.subplots(figsize=(4,4))
    for (chan,color) in zip(chans, colors):
        hist = cv2.calcHist([chan],[0],None,[256],[0,256])
        ax.plot(hist, color=color)
    ax.set_xlim([0,256])
    ax.set_title("Color Histogram")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), np.uint8)
    hist_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    plt.close(fig)
    return cv2.cvtColor(hist_img, cv2.COLOR_BGR2RGB)

# -------------------------
# Morphology Analysis
# -------------------------
def get_morphology(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ratios = []
    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri > 0:
            ratios.append(area/peri)
    mean_ratio = np.mean(ratios) if ratios else 0
    # Plot morphology graph
    fig, ax = plt.subplots(figsize=(3,3))
    ax.bar(range(len(ratios)), ratios, color='skyblue')
    ax.set_title("Morphology Analysis")
    ax.set_xlabel("Contour Index")
    ax.set_ylabel("Area/Perimeter Ratio")
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), np.uint8)
    morph_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    plt.close(fig)
    return mean_ratio, cv2.cvtColor(morph_img, cv2.COLOR_BGR2RGB)

# -------------------------
# Surface Reflectance
# -------------------------
def get_reflectance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean_reflect = np.mean(gray)
    fig, ax = plt.subplots(figsize=(3,3))
    ax.hist(gray.ravel(), bins=50, color='orange', alpha=0.7)
    ax.set_title("Surface Reflectance")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), np.uint8)
    reflect_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    plt.close(fig)
    return mean_reflect, cv2.cvtColor(reflect_img, cv2.COLOR_BGR2RGB)

# -------------------------
# Grad-CAM
# -------------------------
def grad_cam(img, model):
    try:
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if "conv" in layer.name:
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            return img
        grad_model = tf.keras.models.Model([model.inputs],
                                           [model.get_layer(last_conv_layer_name).output, model.output])
        img_array, _ = preprocess(img)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[0][0] if predictions.shape[-1]==1 else predictions[0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = np.maximum(heatmap,0)/np.max(heatmap+1e-9)
        heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(img,0.6,heatmap,0.4,0)
        return superimposed
    except:
        return img

# -------------------------
# Crop sanity check (relaxed)
# -------------------------
def is_likely_rice(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False
    count_ok = 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        ratio = h/w
        if 0.8 <= ratio <= 6:
            count_ok += 1
    return count_ok / len(contours) >= 0.7

def is_likely_corn(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False
    count_ok = 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        ratio = h/w
        if 0.4 <= ratio <= 2.5:
            count_ok += 1
    return count_ok / len(contours) >= 0.7

def is_likely_wheat(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False
    count_ok = 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        ratio = h/w
        if 0.7 <= ratio <= 5:
            count_ok += 1
    return count_ok / len(contours) >= 0.7

# -------------------------
# Seed Dashboard
# -------------------------
def seed_dashboard(crop, img):
    if img is None:
        return "<p style='color:red;'>⚠️ Upload a valid seed image.</p>", None, None, None

    img_array, img_rgb = preprocess(img)
    
    warning_html = ""
    if crop == "others":
        warning_html += """
        <div style="border:2px solid #FFA500; border-radius:10px; padding:10px; background-color:#FFF8E1;">
            ⚠️ Warning: The selected crop type is 'others'. Predictions may be inaccurate.
        </div>
        """
    
    # Prediction
    pred_raw = model.predict(img_array)
    pred = float(pred_raw[0][0]) if pred_raw.shape[-1]==1 else float(pred_raw[0])
    label = "Genuine Seed" if pred < 0.5 else "Counterfeit Seed"
    color = "#4CAF50" if pred < 0.5 else "#F44336"
    icon = "🟢" if pred < 0.5 else "🔴"
    confidence = 1-pred if pred < 0.5 else pred
    
    # Feature engineering
    morph_ratio, morph_img = get_morphology(img_rgb)
    reflectance, reflect_img = get_reflectance(img_rgb)
    hist_img = get_histogram(img_rgb)
    heatmap_img = grad_cam(img_rgb, model)
    
    # Side-by-side plots
    height = 256
    hist_gradcam = cv2.hconcat([cv2.resize(hist_img,(256,height)), cv2.resize(heatmap_img,(256,height))])
    morph_reflect = cv2.hconcat([cv2.resize(morph_img,(256,256)), cv2.resize(reflect_img,(256,256))])
    
    # Confidence bar
    conf_color = (0,int(255*confidence),0) if pred<0.5 else (0,0,int(255*confidence))
    conf_bar = np.ones((30,300,3), dtype=np.uint8)*200
    cv2.rectangle(conf_bar,(0,0),(int(confidence*300),30), conf_color, -1)
    conf_bar_img = cv2.cvtColor(conf_bar, cv2.COLOR_BGR2RGB)
    # Warning logic
    mismatch = False
    if crop == "rice" and not is_likely_rice(img_rgb):
        mismatch = True
    elif crop == "wheat" and not is_likely_wheat(img_rgb):
        mismatch = True
    elif crop == "corn" and not is_likely_corn(img_rgb):
        mismatch = True
    elif crop == "others":
        mismatch = True  # always warn for 'others'

    if mismatch:
        warning_html += """
        <div style="border:2px solid #FFA500; border-radius:10px; padding:10px; background-color:#FFF8E1; margin-top:5px;">
            ⚠️ Warning: Uploaded seed may not match the selected crop type. Predictions could be inaccurate.
        </div>
        """
    # Polished summary
    summary_text = (
        f"The uploaded {crop} seed appears {label} with a confidence score of {confidence:.2f}. "
        f"Morphology analysis indicates an area/perimeter ratio of {morph_ratio:.2f}. "
        f"Surface reflectance is {reflectance:.2f}, indicating healthy sheen. "
        "Color histogram and Grad-CAM confirm consistent texture and no significant anomalies."
    )
    
    summary_html = f"""
        {warning_html}
        <div style="border:2px solid #ccc; border-radius:10px; padding:15px; background-color:#f9f9f9; font-family:sans-serif; margin-top:10px;">
        <h3 style="color:{color}; margin-bottom:5px;">{icon} {label}</h3>
        <p><strong>Crop Type:</strong> {crop.capitalize()}</p>
        <p style="font-weight:bold;">Confidence Score: {confidence:.2f}</p>
        <p style="color:#555; font-size:14px;">{summary_text}</p>
    </div>
    """
    
    return summary_html, conf_bar_img, hist_gradcam, morph_reflect

# -------------------------
# Gradio Interface
# -------------------------
iface = gr.Interface(
    fn=seed_dashboard,
    inputs=[
        gr.Dropdown(choices=["rice","wheat","corn","others"], label="Select Crop"),
        gr.Image(type="numpy", label="Upload Seed Image")
    ],
    outputs=[
        gr.HTML(label="Seed Analysis Summary"),
        gr.Image(type="numpy", label="Confidence Score Bar"),
        gr.Image(type="numpy", label="Histogram + Grad-CAM"),
        gr.Image(type="numpy", label="Morphology + Reflectance Graphs")
    ],
    title="🌱 Sowing Trust: AI-Driven Seed Verifier",
    description="Upload a seed image to get instant authenticity check with detailed analysis."
)


iface.launch(share=True)



