from PIL import Image
from matplotlib import pyplot as plt

# Load images
# mesure/plot/imgs/0C309DE20E98901FB4E2D6C4354B2DF6.png
# mesure/plot/imgs/007D3829A9D4A83837B0BED3B867B04C.png
# mesure/plot/imgs/60F4BF7AC10FF030224E0729A64107BA.png
# mesure/plot/imgs/0570916EDFD327D7B0E7CE177AA1E1AE.png
image1 = Image.open('./imgs/0C309DE20E98901FB4E2D6C4354B2DF6.png')
image2 = Image.open('./imgs/007D3829A9D4A83837B0BED3B867B04C.png')
image3 = Image.open('./imgs/60F4BF7AC10FF030224E0729A64107BA.png')
image4 = Image.open('./imgs/0570916EDFD327D7B0E7CE177AA1E1AE.png')

# Determine the width and height for the final image
width = max(image1.width, image2.width, image3.width, image4.width)
height = image1.height + image2.height + image3.height + image4.height

# Create a new image with the determined width and height
new_image = Image.new('RGB', (width, height))

# Paste the images into the new image
new_image.paste(image1, (0, 0))
new_image.paste(image2, (0, image1.height))
new_image.paste(image3, (0, image1.height + image2.height))
new_image.paste(image4, (0, image1.height + image2.height + image3.height))

# Display the combined image
plt.imshow(new_image)
plt.axis('off')
plt.show()