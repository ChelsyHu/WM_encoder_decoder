from PIL import Image,ImageFont,ImageDraw
import math

def  Generate_Watermark_image(width=256, height=256, text='Chelsy_Hu',number_of_sign=4, text_color=(255,255,255),background_color='black'):
    image = Image.new('RGB', (width, height),background_color)
    
    s1=40  ##字体大小和 起始点调节项
    # font = ImageFont.truetype("Arial Black.TTF", size= math.floor((width-5*(number_of_sign)-20)/number_of_sign))
    font = ImageFont.truetype("Arial Black.TTF", size=s1)
    text_positions=[]
    hei=math.floor((width-5*(number_of_sign-1))/number_of_sign)


    h=0
    for i in range(number_of_sign):
        text_positions.append( (h,s1-20 ) )
        h+=hei+5
    print(text_positions)

    ##添加文字到图片
    draw = ImageDraw.Draw(image)
    
    for text_position in text_positions:
        i,j=text_position[0],text_position[1]
        draw.text(xy=[j,i],text=text,font=font,fill=text_color)

    image = image.convert('L')
    #保存修改的图片
    image.save(f"{text}1.png")
    return image


Generate_Watermark_image(text='Chelsy_Hu')