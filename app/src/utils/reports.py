# %%
import dataframe_image as dfi
import os
import plotly

from fpdf import FPDF


# %%
class PDF(FPDF):

    "Constructs a pdf report. Use the grid() function for vertical and horizontal alignment of report items."

    def grid(self, x1: int, y1: int, x2: int, y2: int):
        self.set_line_width(0.0)
        self.line(x1, y1, x2, y2)

    def lines(self):
        self.set_fill_color(32.0, 47.0, 250.0)  # color for outer rectangle
        self.rect(5.0, 5.0, 200.0, 287.0, "DF")
        self.set_fill_color(255, 255, 255)  # color for inner rectangle
        self.rect(8.0, 8.0, 194.0, 282.0, "FD")

    def img(self, x: float, y: float, w: float, h: float, pltx=None, tablex=None):
        self.set_xy(x, y)
        if pltx:
            self.image(pltx, link="", type="", w=w, h=h)
        else:
            self.image(tablex, w=w, h=h)

    def titles(self, txt: str):
        self.set_xy(0.0, 0.0)
        self.set_font("Arial", "B", 16)
        self.set_text_color(139, 136, 120)
        self.cell(w=210.0, h=40.0, align="C", txt=txt, border=0)

    def texts(self, txt: str, x: float, y: float, color: str, weight: str, size: float):
        self.set_xy(x, y)
        self.set_text_color(color[0], color[1], color[2])
        self.set_font("Arial", weight, size)
        self.multi_cell(190, 3, txt)


# %%
pdf_w = 210
pdf_h = 297

# %%
def convert_to_image(file: str, pltx=None, tablex=None):

    "Takes a plotly figure or pandas table and converts it to an image."

    if pltx:
        plotly.io.write_image(pltx, file=file, format="png", width=700, height=450)
        img = os.getcwd() + "/" + file
    else:
        dfi.export(tablex, file)
        img = os.getcwd() + "/" + file
    return img


# %%
