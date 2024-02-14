
import cv2
from typing import Tuple, List
import matplotlib.pyplot as plt

import cv2
import pytesseract
from langdetect import detect
from colormap import rgb2hex


def locate_image_on_image(locate_image: str, on_image: str, prefix: str = '', visualize: bool = False, color: Tuple[int, int, int] = (0, 0, 255)):
    try:

        image = cv2.imread(on_image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        template = cv2.imread(locate_image, 0)

        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        height, width = template.shape[:2]

        top_left = max_loc
        bottom_right = (top_left[0] + width, top_left[1] + height)

        if visualize:
            cv2.rectangle(image, top_left, bottom_right, color, 1)
            plt.figure(figsize=(10, 10))
            plt.axis('off')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)

        return {f'{prefix}top_left_pos': top_left, f'{prefix}bottom_right_pos': bottom_right}

    except cv2.error as err:
        print(err)



def extract_text_on_image(image_location: str) -> List[str]:
    """
    Extract text written on images using OCR (Optical Character Recognition).

    Args:
        image_location (str): The path to the image file.

    Returns:
        List[str]: A list of strings containing the extracted text from the image.
    """
    try:
        # Read and convert image to grayscale
        image = cv2.imread(image_location)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply blur and adaptive thresholding (adjust parameters if needed)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Filter contours based on aspect ratio and size
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_contours = [cnt for cnt in contours
                        if cv2.contourArea(cnt) > 100 and
                           0.2 < cv2.contourArea(cnt) / (cv2.arcLength(cnt, True) ** 2) < 1.5]

        # Extract text and perform basic cleaning
        extracted_text = []
        for cnt in text_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cropped = gray[y:y + h, x:x + w]
            text = pytesseract.image_to_string(cropped, config='--psm 6')  # Use psm 6 for layout analysis
            if text.strip():
                language = detect(text)  # Detect language automatically
                text = text.replace("\n", " ").replace("\x0c", "").replace(" ", " ").strip()
                extracted_text.append(f"{language}:{text}")  # Store language with text

        return extracted_text

    except Exception as e:
        # Log specific exceptions for better debugging
        if isinstance(e, pytesseract.TesseractNotFoundError):
            print("Tesseract not found. Please install it!")
        else:
            print(f"An unexpected error occurred: {e}")
        return []


def identify_color_composition(image,
                               tolerance: int = 12,
                               limit: int = 2,
                               visualize: bool = False) -> None:
    """Function that identifies the color composition of a
    given image path."""

    extracted_colors = extcolors.extract_from_path(
        image, tolerance=tolerance, limit=limit)

    identified_colors = color_to_df(extracted_colors)

    if not visualize:
        return identified_colors

    list_color = list(identified_colors['c_code'])
    list_percent = [int(i) for i in list(identified_colors['occurrence'])]

    text_c = [c + ' ' + str(round(p*100/sum(list_percent), 1)) + '%' for c, p in zip(list_color,
                                                                                     list_percent)]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(100, 100), dpi=10)
    wedges, _ = ax[0].pie(list_percent,
                          labels=text_c,
                          labeldistance=1.05,
                          colors=list_color,
                          textprops={'fontsize': 60, 'color': 'black'}
                          )

    plt.setp(wedges, width=0.3)

    # create space in the center
    plt.setp(wedges, width=0.36)

    ax[0].set_aspect("equal")
    fig.set_facecolor('grey')

    ax[1].imshow(Image.open(image))

    plt.show()

    return identified_colors


def color_to_df(extracted_colors: tuple):
    """Converts RGB Color values from extcolors output to HEX Values."""

    colors_pre_list = str(extracted_colors).replace(
        '([(', '').replace(')],', '), (').split(', (')[0:-1]
    df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
    df_percent = [i.split('), ')[1].replace(')', '')
                  for i in colors_pre_list]

    # convert RGB to HEX code
    df_rgb_values = [(int(i.split(", ")[0].replace("(", "")),
                      int(i.split(", ")[1]),
                      int(i.split(", ")[2].replace(")", ""))) for i in df_rgb]

    df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(", "")),
                           int(i.split(", ")[1]),
                           int(i.split(", ")[2].replace(")", ""))) for i in df_rgb]

    colors_df = pd.DataFrame(zip(df_color_up, df_rgb_values, df_percent),
                             columns=['c_code', 'rgb', 'occurrence'])

    return colors_df




