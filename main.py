import cv2
import os
import mediapipe as mp
import math
import time
import winsound
import numpy as np
import pytesseract
from PIL import ImageFont, ImageDraw, Image

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_AVAILABLE = True
except ImportError:
    ARABIC_AVAILABLE = False
    print("WARNING: Arabic libraries missing.")

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

FOLDER_PATH = os.path.join("Presentation", "Lect6")
SAMPLE_IMG_PATH = "sample.jpg"
BACKGROUND_PATH = "background.jpg"
ICON_POINTER_PATH = "pointer.png"

WIDTH, HEIGHT = 1366, 768
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mpHand = mp.solutions.hands
hands = mpHand.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mouse_x, mouse_y = 0, 0
mouse_click = False


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_click
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONDOWN:
        mouse_click = True


cv2.namedWindow("Smart System", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Smart System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Smart System", mouse_callback)

current_state = "LOGIN"

if os.path.exists(BACKGROUND_PATH):
    bg_login = cv2.imread(BACKGROUND_PATH)
    bg_login = cv2.resize(bg_login, (WIDTH, HEIGHT))
else:
    bg_login = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

imgCanvas = np.ones((HEIGHT, WIDTH, 3), np.uint8) * 255

icon_pointer = cv2.imread(ICON_POINTER_PATH, -1) if os.path.exists(ICON_POINTER_PATH) else None
if icon_pointer is not None:
    icon_pointer = cv2.resize(icon_pointer, (40, 40))

CORRECT_PASS = "12345"
input_pass = ""
gesture_timer = 0
curr_gesture = -1
pass_reset = False
login_status = ""
status_color = (150, 150, 150)
status_timer = 0

BTN_PRESENT = (150, 180, 600, 290)
BTN_IMG_PROC = (150, 320, 600, 430)
BTN_BOARD = (150, 460, 600, 570)
BTN_GEOMETRY = (766, 180, 1216, 290)
BTN_ART = (766, 320, 1216, 430)
BTN_LOGOUT = (766, 460, 1216, 570)

xp, yp = 0, 0
imgNumber = 0
pathImages = sorted(os.listdir(FOLDER_PATH), key=len) if os.path.exists(FOLDER_PATH) else []
slide_data = []
slide_analyzed = False
nav_cooldown = 0

imgBase = None
imgDisplay = None
brightness = 0
rotation = 0
curr_filter = 0
FILTER_NAMES = ["Original", "Hist Eq", "Blur", "Sharpen", "Threshold"]
view_mode = 0
action_lock = False
feedback_text = ""
feedback_timer = 0

px_per_cm = 40


def put_text_arabic(img, text, position, font_size=32, color=(255, 255, 255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    final_text = text
    if ARABIC_AVAILABLE:
        reshaped_text = arabic_reshaper.reshape(text)
        final_text = get_display(reshaped_text)
    draw.text(position, final_text, font=font, fill=color)
    return np.array(img_pil)


def play_sound(evt):
    try:
        if evt == "key":
            winsound.Beep(1200, 50)
        elif evt == "success":
            winsound.Beep(1000, 100)
        elif evt == "error":
            winsound.Beep(200, 400)
        elif evt == "action":
            winsound.Beep(800, 50)
        elif evt == "nav":
            winsound.Beep(600, 80)
    except:
        pass


def calc_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def overlay_icon(bg, icon, x, y):
    h, w, _ = icon.shape
    y1, y2 = y, y + h
    x1, x2 = x, x + w
    if y1 < 0 or y2 > bg.shape[0] or x1 < 0 or x2 > bg.shape[1]:
        return bg
    alpha_s = icon[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(3):
        bg[y1:y2, x1:x2, c] = (alpha_s * icon[:, :, c] + alpha_l * bg[y1:y2, x1:x2, c])
    return bg


def analyze_text(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        words = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 40 and data['text'][i].strip() != "":
                words.append({'x': data['left'][i], 'y': data['top'][i], 'w': data['width'][i], 'h': data['height'][i]})
        return words
    except:
        return []


def get_fingers(lmList):
    fingers = []
    if len(lmList) != 0:
        if calc_dist(lmList[4][1:], lmList[17][1:]) > calc_dist(lmList[3][1:], lmList[17][1:]) * 1.1:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if lmList[(id + 1) * 4][2] < lmList[(id + 1) * 4 - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
    return fingers


while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.resize(img, (WIDTH, HEIGHT))
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    all_hands = []
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(hand_lms.landmark):
                cx, cy = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                lmList.append([id, cx, cy])
            all_hands.append(lmList)

    if current_state == "LOGIN":
        ui_login = bg_login.copy()

        overlay = ui_login.copy()
        cv2.rectangle(overlay, (WIDTH // 2 - 350, HEIGHT // 2 - 150), (WIDTH // 2 + 350, HEIGHT // 2 + 150), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, ui_login, 0.3, 0, ui_login)
        cv2.rectangle(ui_login, (WIDTH // 2 - 350, HEIGHT // 2 - 150), (WIDTH // 2 + 350, HEIGHT // 2 + 150), status_color, 4)

        ui_login = put_text_arabic(ui_login, "نظام الحماية الذكي", (WIDTH // 2 - 150, HEIGHT // 2 - 120), 40, (255, 255, 255))

        disp_pass = "  ".join(["*" for _ in input_pass]) if input_pass else "_  _  _  _  _"
        font = cv2.FONT_HERSHEY_DUPLEX
        text_size = cv2.getTextSize(disp_pass, font, 1.5, 2)[0]
        text_x = (WIDTH - text_size[0]) // 2
        cv2.putText(ui_login, disp_pass, (text_x, HEIGHT // 2), font, 1.5, (255, 255, 255), 2)

        detected_num = 0

        if len(all_hands) > 0:
            lmList = all_hands[0]
            fingers = get_fingers(lmList)
            detected_num = fingers.count(1)

            if detected_num == 0:
                pass_reset = False
                if login_status != "WRONG":
                    login_status = "READY"
                    status_color = (0, 255, 255)
            else:
                if login_status == "READY":
                    login_status = ""

            if not pass_reset:
                tgt = "NONE"
                if fingers == [0, 0, 0, 0, 1]:
                    tgt = "DELETE"
                elif detected_num > 0 and len(input_pass) < 5:
                    tgt = str(detected_num)

                if calc_dist(lmList[8][1:], lmList[4][1:]) < 30 and fingers[2] == 0:
                    tgt = "ENTER"

                if tgt == str(curr_gesture) and tgt != "NONE":
                    if time.time() - gesture_timer > 0.7:
                        if tgt.isdigit():
                            input_pass += tgt
                            play_sound("key")
                            pass_reset = True
                        elif tgt == "DELETE" and len(input_pass) > 0:
                            input_pass = input_pass[:-1]
                            play_sound("key")
                            pass_reset = True
                        elif tgt == "ENTER":
                            if len(input_pass) == 5:
                                if input_pass == CORRECT_PASS:
                                    play_sound("success")
                                    current_state = "MENU"
                                    mouse_click = False
                                else:
                                    play_sound("error")
                                    input_pass = ""
                                    login_status = "WRONG"
                                    status_color = (0, 0, 255)
                                    status_timer = time.time()
                                    pass_reset = True
                else:
                    gesture_timer = time.time()
                    curr_gesture = tgt

        if time.time() - status_timer > 1.5 and login_status == "WRONG":
            login_status = ""
            status_color = (150, 150, 150)

        info_text = f"Input: {len(input_pass)}/5"
        if detected_num > 0:
            info_text += f" (Detected: {detected_num})"

        font_s = cv2.FONT_HERSHEY_SIMPLEX
        info_size = cv2.getTextSize(info_text, font_s, 0.8, 1)[0]
        info_x = (WIDTH - info_size[0]) // 2
        cv2.putText(ui_login, info_text, (info_x, HEIGHT // 2 + 100), font_s, 0.8, (200, 200, 200), 1)

        if login_status:
            status_size = cv2.getTextSize(login_status, font_s, 1, 2)[0]
            status_x = (WIDTH - status_size[0]) // 2
            cv2.putText(ui_login, login_status, (status_x, HEIGHT // 2 + 60), font_s, 1, status_color, 2)

        cv2.imshow("Smart System", ui_login)

    elif current_state == "MENU":
        imgMenu = np.zeros((HEIGHT, WIDTH, 3), np.uint8) + 30
        imgMenu = put_text_arabic(imgMenu, "القائمة الرئيسية", (WIDTH // 2 - 150, 50), 50, (255, 215, 0))

        btns = [
            (BTN_PRESENT, "عرض تقديمي", "PRESENT"),
            (BTN_IMG_PROC, "معالجة الصور", "IMG_PROC"),
            (BTN_BOARD, "السبورة الذكية", "BOARD"),
            (BTN_GEOMETRY, "الهندسة والقياس", "GEOMETRY"),
            (BTN_ART, "الفن التفاعلي", "ART"),
            (BTN_LOGOUT, "تسجيل الخروج", "LOGOUT")
        ]

        for rect, text, cmd in btns:
            x1, y1, x2, y2 = rect
            is_hover = x1 < mouse_x < x2 and y1 < mouse_y < y2
            color = (60, 60, 60) if not is_hover else (0, 100, 0)
            cv2.rectangle(imgMenu, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(imgMenu, (x1, y1), (x2, y2), (200, 200, 200), 2)
            cx, cy = x1 + 20, y1 + (y2 - y1) // 2 - 20
            imgMenu = put_text_arabic(imgMenu, text, (cx, cy), 40, (255, 255, 255))

            if is_hover and mouse_click:
                if cmd == "LOGOUT":
                    current_state = "LOGIN"
                    input_pass = ""
                    login_status = ""
                else:
                    current_state = cmd
                    if cmd == "BOARD":
                        imgCanvas[:] = 255
                    if cmd == "IMG_PROC":
                        imgDisplay = None
                play_sound("success")
                mouse_click = False

        if mouse_click:
            mouse_click = False
        cv2.circle(imgMenu, (mouse_x, mouse_y), 5, (0, 255, 255), -1)
        cv2.imshow("Smart System", imgMenu)

    elif current_state == "GEOMETRY":
        ui_geo = img.copy()
        cv2.rectangle(ui_geo, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(ui_geo, 0.7, img, 0.3, 0, ui_geo)
        ui_geo = put_text_arabic(ui_geo, "القياس (استخدم الإبهام والسبابة في كلتا اليدين)", (50, 50), 30, (0, 255, 255))

        if len(all_hands) == 2:
            h1, h2 = all_hands[0], all_hands[1]
            f1, f2 = get_fingers(h1), get_fingers(h2)

            if f1 == [1, 1, 0, 0, 0] and f2 == [1, 1, 0, 0, 0]:
                p1_x = int((h1[8][1] + h1[4][1]) / 2)
                p1_y = int((h1[8][2] + h1[4][2]) / 2)

                p2_x = int((h2[8][1] + h2[4][1]) / 2)
                p2_y = int((h2[8][2] + h2[4][2]) / 2)

                x_min, x_max = min(p1_x, p2_x), max(p1_x, p2_x)
                y_min, y_max = min(p1_y, p2_y), max(p1_y, p2_y)

                cv2.rectangle(ui_geo, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

                w_cm = int((x_max - x_min) / px_per_cm)
                h_cm = int((y_max - y_min) / px_per_cm)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(ui_geo, f"{w_cm}cm", ((x_min + x_max) // 2 - 20, y_min - 10), font, 0.6, (0, 255, 255), 2)
                cv2.putText(ui_geo, f"{w_cm}cm", ((x_min + x_max) // 2 - 20, y_max + 25), font, 0.6, (0, 255, 255), 2)
                cv2.putText(ui_geo, f"{h_cm}cm", (x_min - 60, (y_min + y_max) // 2), font, 0.6, (255, 255, 0), 2)
                cv2.putText(ui_geo, f"{h_cm}cm", (x_max + 10, (y_min + y_max) // 2), font, 0.6, (255, 255, 0), 2)

                cv2.circle(ui_geo, (p1_x, p1_y), 8, (0, 0, 255), -1)
                cv2.circle(ui_geo, (p2_x, p2_y), 8, (0, 0, 255), -1)

        cv2.imshow("Smart System", ui_geo)

    elif current_state == "ART":
        art_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        step = 8

        active_pointers = []
        if len(all_hands) > 0:
            for hand_lms in all_hands:
                if get_fingers(hand_lms) == [0, 1, 0, 0, 0]:
                    active_pointers.append((hand_lms[8][1], hand_lms[8][2]))

        for x in range(0, WIDTH, step):
            top_pt = (x, 0)
            bot_pt = (x, HEIGHT)

            influenced = False
            influence_pt = (x, 0)

            for px, py in active_pointers:
                dist_x = abs(x - px)
                if dist_x < 120:
                    influenced = True
                    offset = (120 - dist_x) * 1.2

                    if x < px:
                        curve_x = x - offset
                    else:
                        curve_x = x + offset

                    influence_pt = (int(curve_x), py)
                    break

            if influenced:
                color = (100, 255, 100)
                pts = np.array([[x, 0], influence_pt, [x, HEIGHT]], np.int32)
                cv2.polylines(art_canvas, [pts], False, color, 1, cv2.LINE_AA)
            else:
                cv2.line(art_canvas, (x, 0), (x, HEIGHT), (50, 150, 50), 1)

        for px, py in active_pointers:
            cv2.circle(art_canvas, (px, py), 35, (0, 0, 255), -1)

        ui_art = put_text_arabic(art_canvas, "الفن التفاعلي (السبابة فقط)", (WIDTH // 2 - 150, 50), 30, (200, 200, 200))
        cv2.imshow("Smart System", ui_art)

    elif current_state == "IMG_PROC":
        if imgBase is None:
            if os.path.exists(SAMPLE_IMG_PATH):
                imgBase = cv2.imread(SAMPLE_IMG_PATH)
            else:
                imgBase = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            imgBase = cv2.resize(imgBase, (WIDTH, HEIGHT))

        imgProc = imgBase.copy()

        if rotation == 90:
            imgProc = cv2.rotate(imgProc, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            imgProc = cv2.rotate(imgProc, cv2.ROTATE_180)
        elif rotation == 270:
            imgProc = cv2.rotate(imgProc, cv2.ROTATE_90_COUNTERCLOCKWISE)
        imgProc = cv2.resize(imgProc, (WIDTH, HEIGHT))

        if view_mode == 1:
            bg = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            mob_w = int(HEIGHT * 9 / 16)
            res = cv2.resize(imgProc, (mob_w, HEIGHT))
            off = (WIDTH - mob_w) // 2
            bg[:, off:off + mob_w] = res
            imgProc = bg

        if curr_filter == 1:
            yuv = cv2.cvtColor(imgProc, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            imgProc = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        elif curr_filter == 2:
            imgProc = cv2.GaussianBlur(imgProc, (21, 21), 0)
        elif curr_filter == 3:
            k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            imgProc = cv2.filter2D(imgProc, -1, k)
        elif curr_filter == 4:
            g = cv2.cvtColor(imgProc, cv2.COLOR_BGR2GRAY)
            _, imgProc = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
            imgProc = cv2.cvtColor(imgProc, cv2.COLOR_GRAY2BGR)

        if brightness != 0:
            hsv = cv2.cvtColor(imgProc, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.add(v, brightness)
            imgProc = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

        imgDisplay = imgProc

        if len(all_hands) > 0:
            fingers = get_fingers(all_hands[0])
            lmList = all_hands[0]
            if fingers.count(1) == 0:
                action_lock = False

            if not action_lock:
                if fingers == [0, 1, 1, 0, 0]:
                    view_mode = 1 if view_mode == 0 else 0
                    feedback_text = "Mobile View" if view_mode else "Full View"
                    feedback_timer = time.time()
                    play_sound("action")
                    action_lock = True
                elif fingers == [0, 1, 0, 0, 0]:
                    iy = lmList[8][2]
                    brightness = int(np.interp(iy, [100, HEIGHT - 100], [80, -80]))
                elif fingers == [1, 0, 0, 0, 0]:
                    curr_filter = (curr_filter + 1) % 5
                    feedback_text = FILTER_NAMES[curr_filter]
                    feedback_timer = time.time()
                    play_sound("action")
                    action_lock = True
                elif fingers == [0, 0, 0, 0, 1]:
                    rotation = (rotation + 90) % 360
                    feedback_text = "Rotate"
                    feedback_timer = time.time()
                    play_sound("action")
                    action_lock = True

        if time.time() - feedback_timer < 1.5:
            cv2.putText(imgDisplay, feedback_text, (WIDTH // 2 - 100, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Smart System", imgDisplay)

    elif current_state == "BOARD":
        imgBoardDisplay = imgCanvas.copy()
        imgBoardDisplay = put_text_arabic(imgBoardDisplay, "السبورة البيضاء", (WIDTH // 2 - 100, 20), 30, (200, 200, 200))

        if len(all_hands) > 0:
            fingers = get_fingers(all_hands[0])
            lmList = all_hands[0]
            x1, y1 = lmList[8][1], lmList[8][2]

            if fingers == [0, 1, 1, 1, 1]:
                cv2.circle(imgCanvas, (x1, y1), 50, (255, 255, 255), -1)
                cv2.circle(imgBoardDisplay, (x1, y1), 50, (0, 0, 0), 2)
                xp, yp = 0, 0
            elif fingers[1] == 1 and fingers[2] == 0:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 0), 4)
                cv2.line(imgBoardDisplay, (xp, yp), (x1, y1), (0, 0, 0), 4)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0
                cv2.circle(imgBoardDisplay, (x1, y1), 5, (0, 0, 255), -1)

        cv2.imshow("Smart System", imgBoardDisplay)

    elif current_state == "PRESENT":
        if pathImages:
            full_path = os.path.join(FOLDER_PATH, pathImages[imgNumber])
            imgSlide = cv2.imread(full_path)
            if imgSlide is None:
                imgSlide = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            imgSlide = cv2.resize(imgSlide, (WIDTH, HEIGHT))
            if not slide_analyzed:
                slide_data = analyze_text(imgSlide)
                slide_analyzed = True
        else:
            imgSlide = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        if len(all_hands) > 0:
            fingers = get_fingers(all_hands[0])
            lmList = all_hands[0]
            ix, iy = lmList[8][1], lmList[8][2]

            if fingers == [0, 1, 0, 0, 0]:
                if xp == 0 and yp == 0:
                    xp, yp = ix, iy
                ix = int(xp + (ix - xp) / 1.5)
                iy = int(yp + (iy - yp) / 1.5)
                xp, yp = ix, iy

                hovered = None
                for w in slide_data:
                    if w['x'] - 15 < ix < w['x'] + w['w'] + 15 and w['y'] - 15 < iy < w['y'] + w['h'] + 15:
                        hovered = w
                        break
                if hovered:
                    overlay = imgSlide.copy()
                    cv2.rectangle(overlay, (hovered['x'], hovered['y']), (hovered['x'] + hovered['w'], hovered['y'] + hovered['h']), (0, 255, 255), -1)
                    cv2.addWeighted(overlay, 0.4, imgSlide, 0.6, 0, imgSlide)
                if icon_pointer is not None:
                    imgSlide = overlay_icon(imgSlide, icon_pointer, ix, iy)
                else:
                    cv2.circle(imgSlide, (ix, iy), 10, (0, 0, 255), -1)
            else:
                if time.time() - nav_cooldown > 0.8:
                    if fingers == [1, 0, 0, 0, 0]:
                        if imgNumber < len(pathImages) - 1:
                            imgNumber += 1
                            slide_analyzed = False
                            nav_cooldown = time.time()
                            play_sound("nav")
                    elif fingers == [0, 0, 0, 0, 1]:
                        if imgNumber > 0:
                            imgNumber -= 1
                            slide_analyzed = False
                            nav_cooldown = time.time()
                            play_sound("nav")

        cv2.imshow("Smart System", imgSlide)

    if current_state != "LOGIN" and current_state != "MENU":
        if len(all_hands) > 0 and get_fingers(all_hands[0]) == [1, 1, 1, 1, 1]:
            current_state = "MENU"

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
