import tkinter as tk
import tkinter.messagebox

import pyttsx3
import win32gui
import numpy as np
import tensorflow as tf
from PIL import ImageGrab

def initialize():
    top = tk.Tk()
    top.geometry("300x350")
    top.title("HCR")
    model = tf.keras.models.load_model("model.h5")
    return top, model

def decode(num):
    if 0 <= num <= 9:
        return str(num)
    elif 10 <= num <= 35:
        return chr(int(num) + 55)
    else:
        return None  # Handle invalid input

digit_words = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine"
}

alphabet_words = {
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "O",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "X",
    34: "Y",
    35: "Z"
    # Add the rest of the alphabets here
}

def clear():
    canvas.delete("all")

def predict():
    # Step 1: Getting the canvas ID
    canvas_handle = canvas.winfo_id()
    # Step 2: Get the canvas from ID
    canvas_rect = win32gui.GetWindowRect(canvas_handle)
    # Step 3: Get the canvas content
    img = ImageGrab.grab(canvas_rect)
    # Step 4: Resize the content for CNN input
    img = img.resize((28, 28)).convert("L")
    img = np.array(img)
    img = img.reshape((1, 28, 28, 1))
    img = img / 255.0
    # Step 5: Predict the image drawn
    Y = model.predict([img])[0]
    predicted_label = np.argmax(Y)
    if 0 <= predicted_label <= 35:
        label_word = decode(predicted_label)

        accuracy = np.max(Y) * 100
        accuracy_str = "{:.2f}".format(accuracy)
        if label_word is not None:
            if predicted_label <= 9:
                message = f"It's a DIGIT : {digit_words[predicted_label]} \nAccuracy: {accuracy_str}%"
                engine = pyttsx3.init()
                engine.say(message)
                engine.runAndWait()
                tkinter.messagebox.showinfo("Prediction", message)

            else:
                message = f"It's an ALPHABET : {alphabet_words[predicted_label]} \nAccuracy: {accuracy_str}%"
                engine = pyttsx3.init()
                engine.say(message)
                engine.runAndWait()
                tkinter.messagebox.showinfo("Prediction", message)

        else:
            tkinter.messagebox.showinfo("Prediction", "Invalid label")
    else:
        tkinter.messagebox.showinfo("Prediction", "Invalid label")


def mouse_event(event):
    x, y = event.x, event.y
    canvas.create_oval(x, y, x, y, fill='white', outline='white', width=25)


(root, model) = initialize()
button_frame = tk.Frame(root)

canvas = tk.Canvas(root, bg="black", height=300, width=300)
canvas.bind('<B1-Motion>', mouse_event)
clear_button = tk.Button(button_frame, text="Clear", command=clear)
predict_button = tk.Button(button_frame, text="Predict", command=predict)

canvas.pack()
clear_button.pack(side="left")
predict_button.pack(side="right")

button_frame.pack()
root.mainloop()