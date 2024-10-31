import os
import ollama
import matplotlib.pyplot as plt
from glob import glob
import re

# Functions

# Function to convert from camelCase to a readable name
def camelCaseToSpaces(name):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', name).lower()

# Fucntion to obtain the image file name
def getImageName(image_path):
    return re.search(r'[^\\/]+$', image_path).group()

# Function to get all images from a directory
def getImageFromDirectory(directory):
    supported_image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
    image_files = []
    for extension in supported_image_extensions:
        image_files.extend(glob(os.path.join(directory, extension)))
    return image_files

# Function to get all subfolders from the root directory and and treat each subfolder as a gesture
def getGestureDirectories(root_directory):
    gesture_directories = {}
    for gesture_dir in os.listdir(root_directory):
        full_path = os.path.join(root_directory, gesture_dir)
        if os.path.isdir(full_path):
            gesture_name = camelCaseToSpaces(gesture_dir)
            gesture_directories[gesture_name] = full_path
    return gesture_directories

# Function to get a response from the model considering the given country and gesture
def classifyGesture(image_paths, country, gesture):
    responses = []
    for image_path in image_paths:
        try:
            response = ollama.chat(
                model="llava:13b",
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            f"You are an expert in cultural hand gestures from different countries. "
                            f"In the context of {country}, classify the hand gesture based on cultural appropriateness. "
                            "Respond with 'offensive' or 'not offensive'. Provide no explanations."
                        )
                    },
                    {
                        'role': 'user',
                        'content': f"In {country}, the hand gesture '{gesture}' is: offensive or not offensive? Respond ONLY with one word: either 'offensive' or 'not offensive'",
                        'images': [image_path]
                    }
                ],
                options={'temperature': 0}
            )
            
            # Store the model response
            result = response['message']['content'].strip().lower()
            print(f"Image: {image_path}, Gesture: {gesture}, Country: {country}, Result: {result}")
            responses.append(result)
            
        except Exception as e:
            print(f"Error processing image{image_path}: {e}")
    
    return responses

# Variables 

# Root directory where the gesture subfolders are located
root_directory = "D:/TCC/base"

# Get all gesture subfolders name
gesture_directories = getGestureDirectories(root_directory)

# Define all countries
countries = ["Brazil", "United States", "Japan", "China", "Mexico", "Russia", "India", "France", "Germany", "Iraq", "Egypt"]

# Iterate over countries and gestures
for country in countries:
    print(f"Classifying gestures for the country: {country}")
    
    for gesture, directory in gesture_directories.items():

        # Obtain images and classify based on the gesture and country
        images = getImageFromDirectory(directory)
        responses = classifyGesture(images, country, gesture)

        # Count how many are "offensive" and "not offensive"
        offensive_count = responses.count("offensive")
        not_offensive_count = responses.count("not offensive")

        # Get all categories and counts
        categories = ['Offensive', 'Not Offensive']
        counts = [offensive_count, not_offensive_count]

        # Create bar chart with results
        plt.figure(figsize=(8, 6))
        plt.bar(categories, counts, color=['red', 'green'], width=0.5)

        # Add title and subtitles
        plt.title(f'Results for the "{gesture}" gesture in "{country}"')
        plt.xlabel('Classification')
        plt.ylabel('Quantity')

        # Show graph
        plt.show()