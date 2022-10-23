import os
import threading
import cv2 as cv
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import matplotlib.pyplot as plt

credential_path = os.path.abspath("key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path

target_language = "hi"
img_name = os.path.join("img", "10.jpeg")

places = []
distances = []
y_coords_places = []
y_coords_dists = []
names_and_dists = {}  # {name: dist}
to_translate = []
translated_sentences = []
rect_x1 = {}
rect_y1 = {}
rect_x2 = {}
rect_y2 = {}
names_and_translations = {}
names_to_translate = []


def detect_text(path):
    """
    1. Performs OCR to detect text in the image.
    2. Finds the relevant bounded boxes to be overwritten.
    """
    from google.cloud import vision
    import io

    client = vision.ImageAnnotatorClient()

    with io.open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    for text in texts:
        is_alphanumeric = text.description.isalnum()
        is_ascii = text.description.isascii()
        is_alpha = text.description.isalpha()
        is_num = text.description.isnumeric()

        if is_alphanumeric and is_ascii:
            # print('\n"{}"'.format(text.description))
            vertices = [
                "({},{})".format(vertex.x, vertex.y)
                for vertex in text.bounding_poly.vertices
            ]
            # print('bounds: {}'.format(','.join(vertices)))

            if is_alpha:
                places.append(text.description)
                names_to_translate.append(text.description)
                rect_x1[text.description] = text.bounding_poly.vertices[0].x
                rect_y1[text.description] = text.bounding_poly.vertices[0].y
                rect_x2[text.description] = text.bounding_poly.vertices[2].x
                rect_y2[text.description] = text.bounding_poly.vertices[2].y
                y_mid = (
                    text.bounding_poly.vertices[0].y + text.bounding_poly.vertices[2].y
                ) / 2
                y_coords_places.append(y_mid)

    for text in texts:
        is_alphanumeric = text.description.isalnum()
        is_ascii = text.description.isascii()
        is_alpha = text.description.isalpha()
        is_num = text.description.isnumeric()

        if is_alphanumeric and is_ascii:

            vertices = [
                "({},{})".format(vertex.x, vertex.y)
                for vertex in text.bounding_poly.vertices
            ]

            if is_num:
                distances.append(text.description)
                rect_x1[text.description] = text.bounding_poly.vertices[0].x
                rect_y1[text.description] = text.bounding_poly.vertices[0].y
                rect_x2[text.description] = text.bounding_poly.vertices[2].x
                rect_y2[text.description] = text.bounding_poly.vertices[2].y
                y_mid = (
                    text.bounding_poly.vertices[0].y + text.bounding_poly.vertices[2].y
                ) / 2
                y_coords_dists.append(y_mid)

                # check which y_coord_place is closest to y_mid
                dist = 3000  # some large number
                for y_coord_place in y_coords_places:
                    if abs(y_mid - y_coord_place) < dist:
                        dist = abs(y_mid - y_coord_place)
                        name = places[y_coords_places.index(y_coord_place)]
                names_and_dists[name] = text.description

                # create a string of format 'name, dist kilometres'
                string = f"{name}, {text.description} kilometres"
                # print(string)
                to_translate.append(string)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return texts


def create_rect_mask(image, name, x1, y1, x2, y2):
    """
    Sets the background colour for the bounded boxes to be overwritten.
    """
    # get colour of pixel at x1, (y1+y2)/2
    colour = image[int((y1 + y2) / 2), x1 - 3]

    # convert the values in color to a tuple of integers
    colour = tuple([int(i) for i in colour])

    # draw filled rectangle of color on image
    cv.rectangle(image, (x1, y1), (x2, y2), colour, -1)

    translated_name = names_and_translations[name]

    # save image
    cv.imwrite("output.jpg", image)

    return image


def put_translation(image, name, x1, y1, x2, y2):
    """
    Writes the translated text over the drawn boxes.
    """
    # this time the image is a numpy one
    translated_name = names_and_translations[name]

    draw = ImageDraw.Draw(image)
    # set font size such that it fits in the rectangle
    font = ImageFont.truetype("Nirmala.ttf", 20)
    draw.text((x1, y1), translated_name, (255, 255, 255), font=font)

    # convert image back to opencv format
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

    # save to temp file
    cv.imwrite("output.jpg", image)
    return image


def translate_text(text, target_lang=target_language):
    """
    Translates text into the target language.
    """
    from google.cloud import translate_v2 as translate
    import six

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(
        text,
        target_language=target_lang,
        source_language="en",
    )

    return result["translatedText"]


def play_audio(text):
    """
    TTS
    """
    from gtts import gTTS
    from playsound import playsound

    tts_obj = gTTS(text=text, lang=target_language, slow=False)
    tts_obj.save("output.mp3")
    playsound(os.path.abspath("output.mp3"))
    os.remove("output.mp3")


def tts():
    """
    Run text-to-speech for all the translated sentences.
    """
    for sentence in translated_sentences:
        play_audio(sentence)


def show_new_image():
    """
    Display the output image
    """
    img = cv.imread(img_name)

    for place in places:
        img2 = create_rect_mask(
            img, place, rect_x1[place], rect_y1[place], rect_x2[place], rect_y2[place]
        )

    img2 = plt.imread("output.jpg")
    img2 = Image.fromarray(img2)
    img3 = None
    for place in places:
        img3 = put_translation(
            img2, place, rect_x1[place], rect_y1[place], rect_x2[place], rect_y2[place]
        )

    cv.imshow("image", img3)
    cv.waitKey(0)


def translate():
    """
    Translate all the text
    """
    for sentence in to_translate:
        translated_sentences.append(translate_text(sentence))

    for name in names_to_translate:
        translated_name = translate_text(name)
        names_and_translations[name] = translated_name


def show_output():
    """
    Run TTS and display the output image concurrently.
    """
    thread1 = threading.Thread(target=tts)
    thread1.start()

    thread2 = threading.Thread(target=show_new_image)
    thread2.start()

    thread1.join()
    thread2.join()


if __name__ == "__main__":
    texts = detect_text(img_name)
    translate()
    show_output()
