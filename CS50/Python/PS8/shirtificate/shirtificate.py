from fpdf import FPDF


shirt = FPDF(orientation='portrait', unit='mm', format='A4')
shirt.add_page()
shirt.set_auto_page_break(False)
shirt.image("shirtificate.png", x=0, y=50)

shirt.set_font("helvetica", "B", 40)
shirt.cell(0, 30, "CS50 Shirtificate", align="C")

shirt.ln(100)
shirt.set_font("helvetica", "B", 25)
shirt.set_text_color(255, 255, 255)
shirt.cell(0, 30, input("Name: ") + " took CS50", align="C")


shirt.output("shirtificate.pdf")