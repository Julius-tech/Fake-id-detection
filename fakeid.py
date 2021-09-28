#from requests.api import request
from skimage.metrics import structural_similarity
#import imutils
import cv2
from PIL import Image
import requests

original_url = "https://www.thestatesman.com/wp-content/uploads/2019/07/pan-card.jpg"
fake_url = "https://assets1.cleartax-cdn.com/s/img/20170526124335/Pan4.png"

original = Image.open(requests.get(original_url, 
                                    stream=True).raw)
tampered = Image.open(requests.get(fake_url,
                                    stream=True).raw)

original = original.resize((250, 160))
tampered = tampered.resize((250, 160))

original.save('sample_data/original.png')
tampered.save('sample_image/tampered.png')

original = cv2.imread('sample_image/original.png')
tampered = cv2.imread('sample_image/tampered.png')

original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

(score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
diff = (diff * 255).astype('uint8')
print("SSIM: {}".format(score))