# Comic generator
Blog: https://medium.com/@yvielcastillejos8/comic-generator-using-large-language-models-a-tutorial-13c4e33fad3

## Repo Description
| Scripts/Folders | Description|
|:--:|:--:|
|`static`| Contains the css files and the folder with the generated images |
|`templates`| Contains the html |
|`server.py`| Server for hosting the stable diffusion and gpt models|
|`counters`| counters for the "next image" function|

## COLAB Use
To host the server using google colab, please run https://colab.research.google.com/drive/1WkVlG8LyapurDNgKWpBZ28Du5r2xeroz?usp=sharing 

## Comic Generator using GPT-3 and Stable Diffusion

This project is a comic generator that uses the power of GPT-3 for generating the story and Stable Diffusion for creating the visuals. The Stable Diffusion model is fine-tuned on a custom character and can be used to visualize the character performing various actions.

## How it Works

The comic generator uses GPT-3 to generate a storyline based on a given prompt. The generated text is then processed and divided into panels and pages, which are then passed on to the Stable Diffusion model. The Stable Diffusion model generates the visual for each panel, incorporating the custom character and their actions.

## Features

- GPT-3 powered storyline generation
- Custom fine-tuned Stable Diffusion model for character visualization
- Ability to visualize custom character performing actions
- Generate comic pages and panels
## Model Library 
| Model | Link|
|:--:|:--:|
|Character 1 (Party Girl) | https://drive.google.com/drive/folders/1--Gbk15DRwHeuB5OCLIuA5xnPA4eiJ4k  |
|Character 2 | |

## Getting Started

First. Download the stable diffusion model with the specific character of your choice. To get started with the comic generator, you'll need access to an OpenAI API key for GPT-3 and a GPU for running the Stable Diffusion model. To run:

`` python3 server.py ``

A website will be hosted and the user must follow the instructions on the website.

## Future Improvements

- Expand the custom character library
- Improve the Stable Diffusion model for more realistic visuals
- Add speech bubbles to the generated comics

## Conclusion

The Comic Generator using GPT-3 and Stable Diffusion is a powerful tool for creating comics with custom storylines and characters. With its ability to generate both the story and visuals, this tool can save comic creators a lot of time and effort.
