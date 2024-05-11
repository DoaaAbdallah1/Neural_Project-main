
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import tkinter.filedialog as tkFileDialog
import numpy as np
import os
from PIL import Image, ImageTk
import flet as ft
import matplotlib.pyplot as plt
import numpy as np
import asyncio
import base64
from io import BytesIO
 
 
 
 
 
allInputs = []
weights = np.array([]) 
T = []
imageDispaly='./none.png'
imgInput=ft.Image(src=imageDispaly)
imgSegment=ft.Image(src=imageDispaly)
imgPreprocess=ft.Image(src=imageDispaly)
imgEnhanced=ft.Image(src=imageDispaly)
textAccuracy=ft.Text("Accuracy: ",      style=ft.TextStyle(size=15,weight=ft.FontWeight.BOLD, color=ft.colors.BLACK))
textAccuracyChanged=ft.Text(value="0",     style=ft.TextStyle(size=15,weight=ft.FontWeight.BOLD, color=ft.colors.BLACK))
textPercent=ft.Text("%",     style=ft.TextStyle(size=12,weight=ft.FontWeight.BOLD, color=ft.colors.BLACK))


async def main(page: ft.Page):
 
 
 
    page.window_width=1120
    page.window_height=705
    page.title = "Healthy Or Unhealthy ???"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
 
 
    page.fonts = {
        "Kanit": "https://r...content-available-to-author-only...t.com/google/fonts/master/ofl/kanit/Kanit-Regular.ttf",
 
 
    }
    page.theme = ft.Theme(font_family="Kanit")
 
 
    
    image = None
    text = "_______"
 
    def to_base64(image):
        base64_image = cv2.imencode('.png', image)[1]
        base64_image = base64.b64encode(base64_image).decode('utf-8') 
        return base64_image
 
    async def preprocess_image(image_path):
        # Read the image
        img = cv2.imread(image_path)
 
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
        # Apply histogram equalization for enhancement
        equalized_img = cv2.equalizeHist(gray_img)
        # equalized_img = cv2.equalizeHist( cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY))
        if image_path!='/none.png':
          imgPreprocess.src_base64=f"{to_base64(equalized_img)}"
          await imgPreprocess.update_async()
        else:
          imgPreprocess.src_base64=f"{to_base64('/none.png')}"
          await imgPreprocess.update_async()
 
 
 
    async def segment_image(image_path, threshold_value):
 
        # Read the image
        img = cv2.imread(image_path)
 
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
        # Apply thresholding
        _, thresholded_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
 
        if image_path!='/none.png':
          imgSegment.src_base64=f"{to_base64(thresholded_img)}"
          await imgSegment.update_async()
        else:
          imgSegment.src_base64=f"{to_base64('/none.png')}"
          await imgSegment.update_async()
 

    async def enhance_contrast(image_path, alpha=1.0, beta=0):
        # Read the image
        img = cv2.imread(image_path)

        # Apply contrast enhancement
        enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        if image_path!='/none.png':
          imgEnhanced.src_base64=f"{to_base64(enhanced_img)}"
          await imgEnhanced.update_async()
        else:
          imgEnhanced.src_base64=f"{to_base64('/none.png')}"
          await imgEnhanced.update_async()
        
        
    async def preProcess(path):
        await segment_image(image_path=path, threshold_value=127)
        await preprocess_image(image_path=path)
        await enhance_contrast(image_path=path, alpha=1.5, beta=20)
        await neural(path)
        textAccuracyChanged.value=f"{accuracy()}"
        await textAccuracyChanged.update_async()
    
        print(textAccuracyChanged.value)
        chart.sections[0].value = float(textAccuracyChanged.value)
        chart.sections[1].value = 100-float(textAccuracyChanged.value)
        chart.sections[0].title = f"{float(textAccuracyChanged.value)}"
        chart.sections[1].title = f"{100-float(textAccuracyChanged.value)}"
        await chart.update()

    async def clearAll(path):
        
        imgInput.src=f"{path}"
        await imgInput.update_async()
        
        imgPreprocess.src_base64=f"iVBORw0KGgoAAAANSUhEUgAAAIAAAADGBAMAAADoEZWJAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAwUExURUxpcRAODhAODhcSDxISFA8NDRMLCBAQEQcQFhYXGCAPBTZGRS87PgIBAgEDCBMCAF6aq5cAAAANdFJOUwAqRN9pEb6R+6r45POJJy0iAAACiUlEQVRo3u2YQUsbURCAs+J6joKKtzTxYm9GaVEoEaVCvLiCGLw1OeRQKEpL6akIbUGCvRTsIb14FAohP6P/oM1l3iXSxsPO0dti33ublTR6yJsJLIX5DgEP73Pmzby3s5vJCIIgCIIgCIIgCEI6+NOG2YC4fC9fXm8BfH+y+Yii8PPbt9DnZjPrvN5bLPVXqxCgc+5q8Mqt5N8jgjG4ZeEvtZP1oOxv57WTYL8Ew/zccBEUIBxar/CpSwa76r6gc+wgODFrwlgS9QXqk0MNLuKNO89rtpJqdnOOEVxO29LtzZQiNMGoF26C2mzy58wzZXpBnY6ewy5AdaBz5m1RsDt6O+5AcbDz/Pd2G3qj16Fy+W/nVnQpQlBV8sUwcWQLsUq/Wgo2hwZdULHt9DtLFkza09XdoN+OR45luMcJV2BPBH6hCw5sHWt0wVQ0DgGsBswIuIL0I8AicxOxyhDYRvpKF8yZLeC0coF7Fi64xzkW5JgXyjX9RjqMHJ8sD18H9D7y4huN3gYLkZm16EXwvpnHu2pQj5K/HRkBLlMDKNsaYo+WgbfYjEc2vCJksFSv19dv+1PSW8JDeXBc/BWQqodhMq4+J+T/OFKxQDdBMSAJ7sbMU9I5uhMAbX0s0BMe1og9aAQKP/4gXwNW8Jnx1mYEyBEUuBGMRQDsCM64ggY3BbbgmiGYMymkL/jDEujXlCtmBMAVcCNQmKqg0tYpfOB8BnqzsrIWZASByXyz2cxx1r+MoLfG6OQdO1zQB1zvlUI9HtBPk9dCLUD6g2HCTjiMZ9vkeASQcgQK8IyxibqMiiHw2mZUZpQxftVy+RQ63MpbZhc7jG8/++8M2YwgCIIgCIIgCP8nfwEPXTq9ezHqmQAAAABJRU5ErkJggg=="
        await imgPreprocess.update_async()
        
        imgEnhanced.src_base64=f"iVBORw0KGgoAAAANSUhEUgAAAIAAAADGBAMAAADoEZWJAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAwUExURUxpcRAODhAODhcSDxISFA8NDRMLCBAQEQcQFhYXGCAPBTZGRS87PgIBAgEDCBMCAF6aq5cAAAANdFJOUwAqRN9pEb6R+6r45POJJy0iAAACiUlEQVRo3u2YQUsbURCAs+J6joKKtzTxYm9GaVEoEaVCvLiCGLw1OeRQKEpL6akIbUGCvRTsIb14FAohP6P/oM1l3iXSxsPO0dti33ublTR6yJsJLIX5DgEP73Pmzby3s5vJCIIgCIIgCIIgCEI6+NOG2YC4fC9fXm8BfH+y+Yii8PPbt9DnZjPrvN5bLPVXqxCgc+5q8Mqt5N8jgjG4ZeEvtZP1oOxv57WTYL8Ew/zccBEUIBxar/CpSwa76r6gc+wgODFrwlgS9QXqk0MNLuKNO89rtpJqdnOOEVxO29LtzZQiNMGoF26C2mzy58wzZXpBnY6ewy5AdaBz5m1RsDt6O+5AcbDz/Pd2G3qj16Fy+W/nVnQpQlBV8sUwcWQLsUq/Wgo2hwZdULHt9DtLFkza09XdoN+OR45luMcJV2BPBH6hCw5sHWt0wVQ0DgGsBswIuIL0I8AicxOxyhDYRvpKF8yZLeC0coF7Fi64xzkW5JgXyjX9RjqMHJ8sD18H9D7y4huN3gYLkZm16EXwvpnHu2pQj5K/HRkBLlMDKNsaYo+WgbfYjEc2vCJksFSv19dv+1PSW8JDeXBc/BWQqodhMq4+J+T/OFKxQDdBMSAJ7sbMU9I5uhMAbX0s0BMe1og9aAQKP/4gXwNW8Jnx1mYEyBEUuBGMRQDsCM64ggY3BbbgmiGYMymkL/jDEujXlCtmBMAVcCNQmKqg0tYpfOB8BnqzsrIWZASByXyz2cxx1r+MoLfG6OQdO1zQB1zvlUI9HtBPk9dCLUD6g2HCTjiMZ9vkeASQcgQK8IyxibqMiiHw2mZUZpQxftVy+RQ63MpbZhc7jG8/++8M2YwgCIIgCIIgCP8nfwEPXTq9ezHqmQAAAABJRU5ErkJggg=="
        await imgEnhanced.update_async()
        
        imgSegment.src_base64=f"iVBORw0KGgoAAAANSUhEUgAAAIAAAADGBAMAAADoEZWJAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAwUExURUxpcRAODhAODhcSDxISFA8NDRMLCBAQEQcQFhYXGCAPBTZGRS87PgIBAgEDCBMCAF6aq5cAAAANdFJOUwAqRN9pEb6R+6r45POJJy0iAAACiUlEQVRo3u2YQUsbURCAs+J6joKKtzTxYm9GaVEoEaVCvLiCGLw1OeRQKEpL6akIbUGCvRTsIb14FAohP6P/oM1l3iXSxsPO0dti33ublTR6yJsJLIX5DgEP73Pmzby3s5vJCIIgCIIgCIIgCEI6+NOG2YC4fC9fXm8BfH+y+Yii8PPbt9DnZjPrvN5bLPVXqxCgc+5q8Mqt5N8jgjG4ZeEvtZP1oOxv57WTYL8Ew/zccBEUIBxar/CpSwa76r6gc+wgODFrwlgS9QXqk0MNLuKNO89rtpJqdnOOEVxO29LtzZQiNMGoF26C2mzy58wzZXpBnY6ewy5AdaBz5m1RsDt6O+5AcbDz/Pd2G3qj16Fy+W/nVnQpQlBV8sUwcWQLsUq/Wgo2hwZdULHt9DtLFkza09XdoN+OR45luMcJV2BPBH6hCw5sHWt0wVQ0DgGsBswIuIL0I8AicxOxyhDYRvpKF8yZLeC0coF7Fi64xzkW5JgXyjX9RjqMHJ8sD18H9D7y4huN3gYLkZm16EXwvpnHu2pQj5K/HRkBLlMDKNsaYo+WgbfYjEc2vCJksFSv19dv+1PSW8JDeXBc/BWQqodhMq4+J+T/OFKxQDdBMSAJ7sbMU9I5uhMAbX0s0BMe1og9aAQKP/4gXwNW8Jnx1mYEyBEUuBGMRQDsCM64ggY3BbbgmiGYMymkL/jDEujXlCtmBMAVcCNQmKqg0tYpfOB8BnqzsrIWZASByXyz2cxx1r+MoLfG6OQdO1zQB1zvlUI9HtBPk9dCLUD6g2HCTjiMZ9vkeASQcgQK8IyxibqMiiHw2mZUZpQxftVy+RQ63MpbZhc7jG8/++8M2YwgCIIgCIIgCP8nfwEPXTq9ezHqmQAAAABJRU5ErkJggg=="
        await imgSegment.update_async()
        t.value=text
        await t.update_async()
        
      
        selected_files.value=""
        await selected_files.update()
      
        
        
 
 
 
    # <<<< upload image when  click button upload image
    async def pick_files_result(e: ft.FilePickerResultEvent):
        selected_files.value = (
            ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
        )
        selected_files.update()
 
        # print("Selected files:", e.files[0].path)
        path=e.files[0].path
        imageDispaly=path
        imgInput.src=f"{imageDispaly}"
        await imgInput.update_async()
        if path:
            img = cv2.imread(path,cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(img, (300, 300), interpolation = cv2.INTER_AREA)
            cv2.imwrite(f"test/test.jpg", resized)
            im = Image.open(f"test/test.jpg")
            print(im)
 
            
          
 
 
 
 
    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    selected_files = ft.Text()
 
    page.overlay.append(pick_files_dialog)
 
    # end code upload image when  click button upload image>>>
 
    def accuracy():
        global weights
        input = []
        counter = 0
        folder_dir = "test/"
        i=0
        for image in os.listdir(folder_dir):
            i+=1
            img = cv2.imread(f"{folder_dir}/{image}", cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256))  # Resize the image
                input = flatten(img)
                input = np.array(input)
                input = input.transpose()
                print(weights)
                print(input)
                a = np.dot(weights, input)
                print(a)
                print(a.argmax())
                counter += 1 if (a.argmax() == 0 and i < 4) or (a.argmax() == 1 and i > 3) else 0
            else:
                print(f"Failed to read the image {image}")
        ans = counter / 6
        print("Accuracy: ", ans*100 , "%")
        return ans*100
 
    def orthonormal(pp):
        for i in range(len(pp)):
            for j in range(len(pp[0])):
                if (i==j and pp[i][j] != 1) or (i != j and pp[i][j]!=0):
                    return False
 
        return True
 
    def training():
        global weights, T, allInputs
        S = 1
        folder_dir = "images/Healthy/"
        for image in os.listdir(folder_dir):
            allInputs.append(flatten(cv2.imread(f"{folder_dir}/{image}",cv2.IMREAD_GRAYSCALE)) )
            T.append([1 for _ in range(S)])
        folder_dir = "images/Diseased/"
        for image in os.listdir(folder_dir):
            allInputs.append(flatten(cv2.imread(f"{folder_dir}/{image}",cv2.IMREAD_GRAYSCALE)) )
            T.append([-1 for _ in range(S)])
        allInputs = np.array(allInputs)
        T = np.array(T).transpose()
 
        numP =  len(allInputs)
        R = len(allInputs[0])
 
        if orthonormal(np.dot(allInputs, allInputs.transpose())):
            weights = np.dot(T, allInputs)
        else:
            weights = np.dot(T, np.dot(np.linalg.inv(np.dot(allInputs, allInputs.transpose())),allInputs))
        
 
 
 
    def flatten(image):
        new_image = []
        for row in image:
            for el in row:
                new_image.append(-1 if el<128 else 1)
        return new_image
 
 
    async def neural(path):
        global weights,text
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)  # Resize to 256x256
        p = np.array(flatten(resized))
        p = p.transpose()
        #print(p.shape, weights.shape)
        a = np.dot(weights, p)
        t.value = f" Healthy" if a[0] >= 0 else f" Diseased"
        await t.update_async()
        # text.update()
        # label2.config(text=text)
        # label2.text = text      
        # print(weights[0])
        # print(p)
        # xpoints = np.array(p)
        # ypoints = np.array(weights[0])
 
        # plt.plot(xpoints, ypoints)
        # plt.show()

    
    normal_radius = 130
    hover_radius = 140
    normal_title_style = ft.TextStyle(
        size=12, color=ft.colors.WHITE, weight=ft.FontWeight.BOLD
    )
    hover_title_style = ft.TextStyle(
        size=16,
        color=ft.colors.WHITE,
        weight=ft.FontWeight.BOLD,
        shadow=ft.BoxShadow(blur_radius=2, color=ft.colors.BLACK54),
    )
    normal_badge_size = 40
    hover_badge_size = 50
    x=textAccuracyChanged.value
    def badge(icon, size):
        return ft.Container(
            ft.Icon(icon),
            width=size,
            height=size,
            border=ft.border.all(1, ft.colors.BROWN),
            border_radius=size / 2,
            bgcolor=ft.colors.WHITE,
        )

    def on_chart_event(e: ft.PieChartEvent):
        for idx, section in enumerate(chart.sections):
            if idx == e.section_index:
                section.radius = hover_radius
                section.title_style = hover_title_style
            else:
                section.radius = normal_radius
                section.title_style = normal_title_style
        chart.update()

    chart = ft.PieChart(
        sections=[
            ft.PieChartSection(
                int('60'),
                title=f"{int('60')}%",
                title_style=normal_title_style,
                color=ft.colors.GREEN,
                radius=normal_radius,
                badge=badge(ft.icons.FAVORITE, normal_badge_size),
                badge_position=0.98,
            ),
            ft.PieChartSection(
                100-int('60'),
                title=f"{100-int('60')}%",
                title_style=normal_title_style,
                color=ft.colors.RED,
                radius=normal_radius,
                badge=badge(ft.icons.HEART_BROKEN, normal_badge_size),
                badge_position=0.98,
            ),
        ],
        sections_space=0,
        center_space_radius=0,
        on_chart_event=on_chart_event,
        expand=True,
    )

    def showChart(path):
        global weights,text
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)  # Resize to 256x256
        p = np.array(flatten(resized))
        p = p.transpose()
        a = np.dot(weights, p)
        xpoints = np.array(p)
        ypoints = np.array(weights[0])
        plt.plot(xpoints, ypoints)
        plt.show()
 
    t = ft.Text(text, size=26, color=ft.colors.RED, weight=ft.FontWeight.BOLD)
    def route_change(route):

        page.views.clear()
        page.views.append(
            ft.View(
                bgcolor="#C9E6D1",
                route="/",
                controls=[
                
                    
                    ft.Column([
                      
                        ft.Container(height=5),
                        ft.Row([ft.Container(width=23), ft.Image("img\Logo.png"), ft.Container(width=7),
                                ft.Image("img\plant disease detection.png")]),

                        ft.Row(
                            [
                                ft.Container(content=ft.Container(

                                    content=ft.Column([
                                          ft.Container(
                                            content=ft.Row([
                                                ft.Icon(ft.icons.PREVIEW, color=ft.colors.BLACK),
                                                ft.Text("Training", color=ft.colors.BLACK,
                                                        style=ft.TextStyle(weight=ft.FontWeight.W_700))
                                            ]
                                                , alignment=ft.MainAxisAlignment.CENTER
                                            ),
                                            margin=ft.margin.only(top=30, left=15, right=10,bottom=25),
                                            padding=10,
                                            alignment=ft.alignment.center,
                                            bgcolor='#C9E6D1',
                                            width=177,
                                            height=42,
                                            border_radius=5,
                                            on_click=lambda e: training(),
                                        ),
                                        ft.Container(
                                            content=ft.Row([
                                                ft.Icon(ft.icons.UPLOAD_OUTLINED, color=ft.colors.BLACK),
                                                ft.Text("Upload Image", color=ft.colors.BLACK,
                                                        style=ft.TextStyle(weight=ft.FontWeight.W_700))
                                            ]
                                                , alignment=ft.MainAxisAlignment.CENTER
                                            ),
                                            margin=ft.margin.only(top=15, left=10, right=10, bottom=0),

                                            padding=10,
                                            alignment=ft.alignment.center,
                                            bgcolor='#C9E6D1',
                                            width=177,
                                            height=42,
                                            border_radius=5,
                                            on_click=lambda _: pick_files_dialog.pick_files(allow_multiple=True),
                                        ),
                                        selected_files,

                                        # ft.Container(content=ft.Text("Upload photo of your planet", color="#464646")
                                        #              ,
                                        #              margin=ft.margin.only(left=10, right=10),
                                        #              ),
                                        
                                      
                                        ft.Container(
                                            content=ft.Row([
                                                ft.Icon(ft.icons.PREVIEW, color=ft.colors.BLACK),
                                                ft.Text("PreProcess", color=ft.colors.BLACK,
                                                        style=ft.TextStyle(weight=ft.FontWeight.W_700))
                                            ]
                                                , alignment=ft.MainAxisAlignment.CENTER),
                                            margin=ft.margin.only(top=10, left=10, right=10, bottom=10),
                                            padding=10,
                                            alignment=ft.alignment.center,
                                            bgcolor='#C9E6D1',
                                            width=177,
                                            height=42,
                                            border_radius=5,
                                            on_click= lambda e : asyncio.run(preProcess(path=imgInput.src)),
                                        ),
                                        
                                        ft.Container(
                                            content=ft.Row([
                                                ft.Icon(ft.icons.DELETE, color=ft.colors.BLACK),
                                                ft.Text("Clear all", color=ft.colors.BLACK,
                                                        style=ft.TextStyle(weight=ft.FontWeight.W_700))
                                            ]
                                                , alignment=ft.MainAxisAlignment.CENTER
                                            ),
                                            margin=ft.margin.only(top=30, left=10, right=10, bottom=10),

                                            padding=10,
                                            alignment=ft.alignment.center,
                                            bgcolor='#C9E6D1',
                                            width=177,
                                            height=42,
                                            border_radius=5,
                                            on_click=lambda _:  asyncio.run(clearAll(path='./none.png')),
                                        ),
                                        
                                    ]
                                        ,
                                    ),
                                    margin=ft.margin.only(top=10, left=20, right=10, bottom=10),
                                    padding=ft.padding.only(top=10, left=20, right=20, bottom=10),
                                    alignment=ft.alignment.center,
                                    bgcolor="#88CDA8",
                                    width=256,
                                    height=415,
                                    border_radius=24,
                                ), height=500, ),
                                ft.Container(
                                    margin=ft.margin.only(top=40, left=30),
                                    width=430,
                                      #alignment=ft.alignment.center,
                                    #bgcolor=ft.colors.RED,
                                    height=480,
                                    content=ft.Column(
                                        [
                                            ft.Row([ft.Column([ft.Container(
                                                border=ft.border.all(width=1.2, color=ft.colors.BLACK),
                                                content=imgInput, width=181, height=178),
                                                               ft.Container(
                                                                   ft.Text("input image", weight=ft.FontWeight.W_700, color=ft.colors.BLACK),
                                                                   width=181, alignment=ft.alignment.center),
                                                               ft.Container(height=20), ]),
                                                    ft.Container(width=20)
                                                       , ft.Column([ft.Container(
                                                    border=ft.border.all(width=1.2, color=ft.colors.BLACK),
                                                    content=imgPreprocess, width=181, height=178),
                                                                    ft.Container(ft.Text("Gary conversion",
                                                                                         weight=ft.FontWeight.W_700, color=ft.colors.BLACK),
                                                                                 width=181,
                                                                                 alignment=ft.alignment.center),
                                                                    ft.Container(height=20), ]), ]),
                                            ft.Row([ft.Column([ft.Container(
                                                border=ft.border.all(width=1.2, color=ft.colors.BLACK),
                                                content=imgEnhanced, width=181, height=178),
                                                               ft.Container(
                                                                   ft.Text("Enhancement", weight=ft.FontWeight.W_700, color=ft.colors.BLACK),
                                                                   width=181, alignment=ft.alignment.center),
                                                               ft.Container(height=10), ]),
                                                    ft.Container(width=20)
                                                       , ft.Column([ft.Container(
                                                    border=ft.border.all(width=1.2, color=ft.colors.BLACK),
                                                    content=  imgSegment, width=181, height=178),
                                                                    ft.Container(ft.Text("segmentation",
                                                                                         weight=ft.FontWeight.W_700, color=ft.colors.BLACK),
                                                                                 width=181,
                                                                                 alignment=ft.alignment.center),
                                                                    ft.Container(height=10), ])])

                                        ,
                                            ft.Row([
                                                ft.Text("Healthy Or Diseased ? :", size=23, weight=ft.FontWeight.BOLD, color=ft.colors.BLACK)
                                                ,
                                                t,
                                          

                                            ]
                                                , alignment=ft.MainAxisAlignment.CENTER)
                                        ]
                                    )
                                    ,

                                ),

                                ft.Container(

                                    width=230,
                                    alignment=ft.alignment.center,
                                    height=428,
                                   #bgcolor=ft.colors.RED,
                                    margin=ft.margin.only(bottom=50),

                                    content=ft.Column([
                                        ft.Container(
                                          content=ft.Text("Parameters", style=ft.TextStyle(size=24.7, weight=ft.FontWeight.BOLD), color=ft.colors.BLACK),
                                          margin=ft.margin.only(left=17),
                                        ),
                                        ft.Container(

                                            content=ft.Row([textAccuracy,textAccuracyChanged,textPercent],
                                                                 alignment=ft.MainAxisAlignment.CENTER
                                                              )
                                            , bgcolor="#33D45F"
                                            , width=145, height=65
                                            , padding=5
                                            , margin=ft.margin.only(left=15,top=20, bottom=43)
                                        ),
                                        ft.Container(content=ft.Text("Diagrams", style=ft.TextStyle(size=26, weight=ft.FontWeight.BOLD), color=ft.colors.BLACK),
                                              margin=ft.margin.only(left=20),
),
                                        ft.Container(

                                            content=ft.Text("Show Diagram 1",color=ft.colors.BLACK,  style=ft.TextStyle(size=18, weight=ft.FontWeight.W_700))
                                            , bgcolor="#33D45F"
                                            , padding=10
                                            , width=160, height=45
                                            , margin=ft.margin.only(top=20, bottom=10)
                                            ,on_click=lambda e: showChart(path=imgInput.src)
                                        ),
                                        
                                          ft.Container(

                                            content=ft.Text("Show Diagram 2", color=ft.colors.BLACK, style=ft.TextStyle(size=18, weight=ft.FontWeight.W_700))
                                            , bgcolor="#33D45F"
                                            , padding=10
                                            , width=160, height=45
                                            , margin=ft.margin.only(top=10, bottom=10)
                                            ,on_click=lambda e: page.go("/showDigrame")
                                        )
                                    ]
                                        ,
                                        alignment=ft.MainAxisAlignment.CENTER))

                            ],
                            alignment=ft.MainAxisAlignment.CENTER,

                        ),

                    ]

                    )
                ],
            )
        )
        if page.route == "/showDigrame":
            page.views.append(
                ft.View(
                    "/showDigrame",
                    [
                      
                      
                         ft.Row([ft.Container(width=23), ft.Image("img\Logo.png"), ft.Container(width=7),
                                ft.Image("img\plant disease detection.png")]),

                        ft.Container(height=30),
                       ft.Container(
                           width=1000,
                           alignment=ft.alignment.center,
                           padding=ft.padding.only(left=320),
                           content=ft.Column([
                           ft.Container(
                             margin=ft.margin.only(right=200),
                              
                               content=chart
                               
                               ,),
                           ft.Container(height=50),
                           ft.Row([
                               ft.Container(width=40),
                               ft.ElevatedButton("Back to Result", color=ft.colors.BLACK, bgcolor="#33D45F",
                                                 on_click=lambda _: page.go("/")),
                              ft.Container(width=40),
                               ft.ElevatedButton("Download Diagram", color=ft.colors.BLACK, bgcolor="#33D45F",
                                                 on_click=lambda _: print("Downloads"))]),
                       ])
                       )

                    ],
                    bgcolor="#C9E6D1",
                )
            )
        page.update()

    def view_pop(view):
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop
    page.go(page.route)
    

ft.app(target=main )