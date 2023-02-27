from flask import Flask,request, url_for, redirect, render_template
#from flask_socketio import SocketIO, emit, send
import pandas as pd
#from bot import*
import openai
import regex
import re
app = Flask(__name__, template_folder='templates')
#app.config[ 'SECRET_KEY' ] = 'mysecret'
#socketio = SocketIO(app, cors_allowed_origins= "*")
import cv2
import numpy as np
import re
import pickle
import os
import json
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from IPython.display import display



def get_stable_diff_output(WEIGHTS_DIR, prompt, strip):
   model_path = "/Users/yvielcastillejos/Thesis_web/800" # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive

   scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
   pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.bfloat16).to("cpu")

   i = 0
   g_cuda = None
   prompt = prompt #"partyxyz girl playing a guitar in a concert" #@param {type:"string"}
   negative_prompt = "" #@param {type:"string"}
   num_samples = 4 #@param {type:"number"}
   guidance_scale = 8 #@param {type:"number"}
   num_inference_steps = 50 #@param {type:"number"}
   height = 512 #@param {type:"number"}
   width = 512 #@param {type:"number"}

   print("done loading pipe") 
   with autocast("cpu"), torch.inference_mode():
       images = pipe(
           prompt,
           height=height,
           width=width,
           negative_prompt=negative_prompt,
           num_images_per_prompt=num_samples,
           num_inference_steps=num_inference_steps,
           guidance_scale=guidance_scale,
           generator=g_cuda
       ).images

   for img in images:
        img.save(f"static/strips_{strip}/{i}.jpg")
        i+=1

def separate_string(input_string):
    return [word.strip() for word in input_string.split(".") if word.strip()]

def get_storyIdeas():
    text = "Give a list of 5 story plot ideas."
    api = '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' #'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'
    openai.api_key = api #os.getenv(api)

    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=text,
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )


    output = response["choices"][0]["text"]
    i = separate_string(output)
    print(i)
    return i

def get_characterIdeas(story):
    text = f"Give a list of 5 characters with the format <character name>:<character description>. The character is the main character to the story and the story is about {story}."
    api = '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' #'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'
    openai.api_key = api #os.getenv(api)

    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=text,
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )


    output = response["choices"][0]["text"]
    i = separate_string(output)
    print(i)
    return i



def get_chatgpt(prompt, charname, chardesc):

    Character1_name = charname
    Character1_description = chardesc  #"a 10 year old boy who is learning how to ride a bicycle for the first time."

    text = f"Generate a story of '{prompt}' with Dialogue. The story SHOULD only contain one character talking to himself/herself and the Dialogue SHOULD only contain 3 lines of dialogue.\n CHARACTERS: {Character1_name}: {Character1_description}\n SCENE:"
    api = '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' #'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'
    openai.api_key = api #os.getenv(api)

    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=text,
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    #print(response)
    output = response["choices"][0]["text"]
    return output, Character1_name, Character1_description

def get_characters(output):
    output_list = output.split("\n")
    Characters = []
    Lines = []
 
    for line in output_list:
     if ":" in line:
       string = ""
       A = False
       for letter in line:
    
          if letter == ":":
              A = True
              Characters.append(string)
              string = ""
          string += letter
       if A:
          Lines.append(string)

  
    out = list(zip(Characters, Lines))
    #print(out)
    return out, Lines


def get_description(Lines, char_name, char_desc):
    Description = []
    for line in Lines:
       # print(description)
        text_description = f"Create an objective description and observation of {char_name}, {char_desc}, and the objects from the dialogue. The description must be one sentence and must describe the characters actions or physical state as specifically as possible. For example, 'the girl is carrying a book', or 'John is standing in front of a door'. Here is the dialogue: \n\nDIALOGUE:\n {line[1::]}\n\nDESCRIPTION:\n"

        description = openai.Completion.create(
            model="text-davinci-003",
            prompt=text_description,
            temperature=0.7,
            max_tokens=75,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    
        desc = description["choices"][0]["text"]
        Description.append(desc)
    return Description

def add_bubble(img, txt, num):
     img_name = img
     img = cv2.imread(img)    
     h,w, _ = img.shape
     CENTER = (64, 64)

     cv2.circle(img, CENTER, 60, (255,255,255), -1)

     TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
     TEXT_SCALE = 0.5
     TEXT_THICKNESS = 1
     TEXT = txt

     text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
     text_origin = (int(CENTER[0] - text_size[0] / 2), int(CENTER[1] + text_size[1] / 2))

     cv2.putText(img, TEXT, text_origin, TEXT_FACE, TEXT_SCALE, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)
     cv2.imwrite(f"static/strips/{num}.png",img) 

@app.route('/', methods=['POST', 'GET'])
def home():
   print("You are now on the idea page...")
   global global_counter_story 
   global_counter_story= 0
   global global_counter_char 
   global_counter_char = 0
   global ideas 
   ideas = []
   global char_ideas 
   char_ideas = []
   if request.method =="POST":
      user_input = request.form["idea"]
      print("The user has inputted: " + user_input + " .")
      if user_input=="":
         print("Attempting to generate plot...")
         return redirect(url_for("generatePlot", usr=user_input+"no-idea"))
      else:
         print("We will now ask for characters")
         return redirect(url_for("characterAsk", story=user_input))
 
   return render_template("page1.html")

@app.route('/character_ask-<story>', methods=['POST', 'GET'])
def characterAsk(story):
   print("Asking for character and character description...")
   print(story)
   if request.method =="POST":
       character_name = request.form["charname"]
       character_desc = request.form["chardesc"]
       
       if character_name == "":
          return redirect(url_for("generateCharacter", story=story))
       else:
          return redirect(url_for("generateComic", story=story, char_name=character_name, char_desc=character_desc))
   return render_template("page2.html")


@app.route('/idea-<usr>', methods=['POST', 'GET'])
def generatePlot(usr):
   print("Generating the plot...")
   if not os.path.exists("story.json"):
       ideas = get_storyIdeas()
       with open("story.json", 'w') as f:
           json.dump(ideas, f, indent=2) 

   if request.method == "POST":
       with open("story.json", 'r') as f:
           ideas = json.load(f)
       idea_idx = int(list(request.form.keys())[0])
       story = ideas[idea_idx]
       print(story)
       os.remove("story.json")
       return redirect(url_for("characterAsk2", story1=story))
   print("generating")
   return render_template("page3_suggestedideas.html", message=ideas[1], message1=ideas[3], message2 = ideas[5], message3 = ideas[7], message4 = ideas[9])


@app.route('/character_ask2-<story1>', methods=['POST', 'GET'])
def characterAsk2(story1):
   story = story1
   print("222Asking for character and character description...")
   print(story)
   if request.method =="POST":
       character_name = request.form["charname"]
       character_desc = request.form["chardesc"]
          
       if character_name == "": 
          return redirect(url_for("generateCharacter", story=story))
       else:
          return redirect(url_for("generateComic", story=story, char_name=character_name, char_desc=character_desc))
   return render_template("page2.html")


@app.route('/idea-characterGen-<story>', methods=['POST', 'GET'])
def generateCharacter(story):
   
   print("generating the characters...")
   if not os.path.exists("char_ideas.json"):
       while True:
           char_ideas = get_characterIdeas(story)
           for i in [1,3,5,7,9]:
              elem = char_ideas[i]
              if elem.isdigit():
                 char_ideas = get_characterIdeas(story)
           break
       with open("char_ideas.json", 'w') as f:
           json.dump(char_ideas, f, indent=2)  
   
   if request.method == "POST":
       with open("char_ideas.json", 'r') as f:
           char_ideas = json.load(f)
       idea_idx = int(list(request.form.keys())[0])
       charactername_desc = char_ideas[idea_idx]
       charnd = re.split('[:]', charactername_desc)
       char_name = charnd[0]
       char_desc = charnd[1]
       os.remove("char_ideas.json")
       return  redirect(url_for("generateComic", story=story, char_name=char_name, char_desc=char_desc))
   return render_template("page4_suggestedcharacters.html", story=story, message=char_ideas[1], message1=char_ideas[3], message2 = char_ideas[5], message3 = char_ideas[7], message4 = char_ideas[9])


#@app.route('/<char_name>-<char_desc>-<story>')
@app.route('/idea-<story>/<char_name>-<char_desc>')
def generateComic(story, char_name, char_desc):
    print("generating the comic_here..." +  char_name + "," + char_desc + " " + story)
    gpt3_output, char_name, char_desc = get_chatgpt(story, char_name, char_desc)
    gpt3_output, lines =  get_characters(gpt3_output)
    while len(lines)!=3:
       gpt3_output, char_name, char_desc = get_chatgpt(story, char_name, char_desc)
       gpt3_output, lines =  get_characters(gpt3_output)
    token_prompt = "partyxyz girl"

    for line in lines:
      print("line: ", line, "\n")
    
    sd_desc = get_description(lines, char_name, char_desc)
    strip = 0
    return redirect(url_for("temp_page", desc1=sd_desc[0].replace(char_name, token_prompt), desc2=sd_desc[1].replace(char_name, token_prompt), desc3=sd_desc[2].replace(char_name, token_prompt), line1=lines[0], line2=lines[1], line3=lines[2]))
    for desc in sd_desc:
       print("desc: " + desc.replace(char_name, token_prompt) + "\n\n")
#       get_stable_diff_output("/Users/yvielcastillejos/Thesis_web/800", desc.replace(char_name, token_prompt), strip)
       strip += 1
    add_bubble("sd_images/1.png", gpt3_output[0][1], 1)
    return redirect(url_for("generateComicpage", line1=lines[0], line2=lines[1], line3=lines[2]))
    #return render_template("home.html", message1=lines[0], message2=lines[1], message3=lines[2])


@app.route('/comicpage-<desc1>-<desc2>-<desc3>-<line1>-<line2>-<line3>',  methods=['POST', 'GET'])
def temp_page(desc1, desc2, desc3, line1, line2, line3):
    if request.method == "POST":
       return redirect(url_for("generateComicpage", line1=line1, line2=line2, line3=line3))
    return render_template("temp.html", desc1=desc1, desc2=desc2, desc3=desc3)
    
  
@app.route('/comicpage-<line1>-<line2>-<line3>',  methods=['POST', 'GET'])
def generateComicpage(line1,line2,line3):
   counter1 = open("counters/counter1.txt", "r").read().replace("\n", "")
   counter2 = open("counters/counter2.txt", "r").read().replace("\n", "")
   counter3 = open("counters/counter3.txt", "r").read().replace("\n", "")
   img1 = "/static/strips_1/" + counter1 + ".jpg"
   img2 = "/static/strips_2/" + counter2 + ".jpg"
   img3 = "/static/strips_3/" + counter3 + ".jpg"
   print("img1 ", img1, "\t", "img2 ", img2, "\t", "img3 ", img3, "\t")
   if request.method == "POST":
      if request.form.get('action1') == "Click to get next image for strip 1":
          counter1 = (int(counter1) + 1)%4
          img1 = "/static/strips_1/" + str(counter1) + ".jpg"
          with open('counters/counter1.txt', 'w') as f:
              f.writelines(str(counter1))
      if request.form.get('action2') == "Click to get next image for strip 2": 
          counter2 = (int(counter2) + 1)%4
          img2 = "/static/strips_2/" + str(counter2) + ".jpg"
          with open('counters/counter2.txt', 'w') as f:
              f.writelines(str(counter2))  
    
      if request.form.get('action3') == "Click to get next image for strip 3": 
          counter3 = (int(counter3) + 1)%4
          img3 = "/static/strips_3/" + str(counter3) + ".jpg"
          with open('counters/counter3.txt', 'w') as f:
              f.writelines(str(counter3))
   print("img1 ", img1, "\t", "img2 ", img2, "\t", "img3 ", img3, "\t")
   return render_template("home.html", message1=line1, message2=line2, message3=line3, img1=img1, img2=img2, img3=img3)



if __name__ == '__main__':
  # wraps around the socket io
    # Flask app waits for request respons
    # Socket iois a real time 
    # adding to the standard server when running app and 
    # have a real time functionality
  #get_stable_diff_output("", "")
  app.run(debug=True, port=5005, use_reloader=True)
#  socketio.run( app, debug = True,use_reloader=False )
