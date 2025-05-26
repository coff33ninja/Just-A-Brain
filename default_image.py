from PIL import Image
import os

# Create directory if it doesn't exist
# Using exist_ok=True to prevent error if directory already exists
os.makedirs("data/images", exist_ok=True)

# Create image1.png (red square)
img1 = Image.new("RGB", (50, 50), color="red")
img1.save("data/images/image1.png")

# Create image2.png (blue square)
img2 = Image.new("RGB", (50, 50), color="blue")
img2.save("data/images/image2.png")

# Create default_image.png (gray square)
default_img = Image.new("RGB", (32, 32), color="gray")
default_img.save("data/images/default_image.png")

print("images created successfully in data/images/")
