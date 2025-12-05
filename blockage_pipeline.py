# Code from Rajiv Joshi for blockage detection pipeline
import cv2 as cv
import os
    
def blockage_pipeline(image_path, save_folder):
    
    # create folder to save results if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # load the image
    img = cv.imread(image_path)
    
    # Cut out border
    img = img[20:-20, 20:-20]
    
    # convert to grayscale image
    grayscale_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # brighten the image
    grayscale_image = cv.convertScaleAbs(grayscale_image, alpha=1.5, beta=20)
    
    # show the grayscale image
    cv.imshow("Grayscale Image", grayscale_image)
    cv.waitKey(0)
    
    # pseudocolor the image
    pseudocolored_image = cv.applyColorMap(grayscale_image, cv.COLORMAP_JET)
    
    # Save the pseudocolored image
    pseudocolor_output_path = os.path.join(save_folder, image_path.replace('.png', '_pseudocolored.png'))
    cv.imwrite(pseudocolor_output_path, pseudocolored_image)
    
    # show the pseudocolored image
    cv.imshow("Pseudocolored Image", pseudocolored_image)
    cv.waitKey(0)
    
    # detect light blue regions (potential blockages)
    lower_blue = (200, 150, 0)
    upper_blue = (255, 255, 150)
    mask = cv.inRange(pseudocolored_image, lower_blue, upper_blue)
    
    # show the mask image
    cv.imshow("Mask Image", mask)
    cv.waitKey(0)
    
    # Save the mask image
    mask_output_path = os.path.join(save_folder, image_path.replace('.png', '_mask.png'))
    cv.imwrite(mask_output_path, mask)
    
    # find contours of the masked regions
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image = img, contours = contours, contourIdx = -1, color = (0, 255, 0), thickness = 2)
    
    # remove really small contours and really large contours
    contours = [cnt for cnt in contours if 500 < cv.contourArea(cnt) < 1000]
    
    # redraw found contours on original image
    cv.drawContours(image = img, contours = contours, contourIdx = -1, color = (0, 255, 0), thickness = 2)
    

    
    # save the final image with detected blockages
    output_path = os.path.join(
        save_folder,
        os.path.basename(image_path).replace('.png', '_blockages_detected.png')
    )
    
    # # show the final image with detected blockages
    # cv.imshow("Detected Blockages", img)
    # cv.waitKey(0)
    

# Example usage
if __name__ == "__main__":
    blockage_pipeline("Angiogram_1.png", "Angiogram_1_output")
    blockage_pipeline("Angiogram_2.png", "Angiogram_2_output")
    blockage_pipeline("Angiogram_3.png", "Angiogram_3_output")