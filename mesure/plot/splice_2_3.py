import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# balance line
# abalone_img = mpimg.imread('./imgs/Abalone_line.png')
# adult_img = mpimg.imread('./imgs/Adult_line.png')
# hcvdat0_img = mpimg.imread('./imgs/Hcvdat0_line.png')
# obesity_img = mpimg.imread('./imgs/Obesity_line.png')
# room_img = mpimg.imread('./imgs/Room_line.png')
# whosale_img = mpimg.imread('./imgs/Whosale_line.png')

# sc
abalone_img = mpimg.imread('./imgs/sc/Abalone.png')
adult_img = mpimg.imread('./imgs/sc/Adult.png')
hcvdat0_img = mpimg.imread('./imgs/sc/Hcvdat0.png')
obesity_img = mpimg.imread('./imgs/sc/Obesity.png')
room_img = mpimg.imread('./imgs/sc/Room.png')
whosale_img = mpimg.imread('./imgs/sc/Wholesale.png')

# Create a figure to arrange the subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 10))

# Add images to the subplots
axs[0, 0].imshow(abalone_img)
axs[0, 0].axis('off')
# axs[0, 0].set_title('Abalone')

axs[0, 1].imshow(adult_img)
axs[0, 1].axis('off')
# axs[0, 1].set_title('Adult')

axs[1, 0].imshow(hcvdat0_img)
axs[1, 0].axis('off')
# axs[1, 0].set_title('Hcvdat0')

axs[1, 1].imshow(obesity_img)
axs[1, 1].axis('off')
# axs[1, 1].set_title('Obesity')

axs[2, 0].imshow(room_img)
axs[2, 0].axis('off')
# axs[2, 0].set_title('Room')

axs[2, 1].imshow(whosale_img)
axs[2, 1].axis('off')
# axs[2, 1].set_title('Wholesale')

plt.subplots_adjust(left=0.21, right=0.35,
                    top=0.96, bottom=0.95,
                    hspace=0.1, wspace=0.9)

plt.tight_layout(pad=0)  # Removed padding
plt.savefig('combined_image.png')
plt.show()
