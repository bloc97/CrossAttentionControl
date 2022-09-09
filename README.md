# Cross Attention Control with Stable Diffusion
Unofficial implementation of "Prompt-to-Prompt Image Editing with Cross Attention Control" with Stable Diffusion  

Paper: https://arxiv.org/abs/2208.01626

## What is Cross Attention Control?
Large-scale language-image models (eg. Stable Diffusion) are usually hard to control just with changing prompts and can be very unpredictable and unintuitive for users. Cross Attention Control allows much finer control of the prompt by modifying the internal attention maps of the diffusion model during inference with minimal performance penalities (compared to clip guidance) and requires no additional training or fine-tuning.

## Getting started
This notebook uses the following libraries: `torch transformers diffusers numpy PIL tqdm difflib`  
Simply install the required libraries using `pip` and run the jupyter notebook, some examples are given inside.

# Results/Demonstrations

## Reducing unpredictability when modifying prompts

Left image prompt: `a fantasy landscape with a pine forest`  
Right image prompt: `a winter fantasy landscape with a pine forest`  
Middle image: Cross attention enabled prompt editing (left image -> right image)  
![Demo](https://github.com/bloc97/CrossAttentionControl/blob/main/images/a%20fantasy%20landscape%20with%20a%20pine%20forest%20-%20a%20winter%20fantasy%20landscape%20with%20a%20pine%20forest.png?raw=true)

Left image prompt: `a fantasy landscape with a pine forest`  
Right image prompt: `a watercolor painting of a landscape with a pine forest`  
Middle image: Cross attention enabled prompt editing (left image -> right image)  
![Demo](https://github.com/bloc97/CrossAttentionControl/blob/main/images/a%20fantasy%20landscape%20with%20a%20pine%20forest%20-%20a%20watercolor%20painting%20of%20a%20landscape%20with%20a%20pine%20forest.png?raw=true)

Left image prompt: `a fantasy landscape with a pine forest`  
Right image prompt: `a fantasy landscape with a pine forest and a river`  
Middle image: Cross attention enabled prompt editing (left image -> right image)  
![Demo](https://github.com/bloc97/CrossAttentionControl/blob/main/images/a%20fantasy%20landscape%20with%20a%20pine%20forest%20-%20A%20fantasy%20landscape%20with%20a%20pine%20forest%20and%20a%20river.png?raw=true)

## Direct prompt control
Left image prompt: `a fantasy landscape with a pine forest`  
Towards the right: `-fantasy`
![Demo](https://github.com/bloc97/CrossAttentionControl/blob/main/images/a%20fantasy%20landscape%20with%20a%20pine%20forest%20-%20decrease%20fantasy.png?raw=true)

Left image prompt: `a fantasy landscape with a pine forest`  
Towards the right: `+fantasy` and `+forest` 
![Demo](https://github.com/bloc97/CrossAttentionControl/blob/main/images/a%20fantasy%20landscape%20with%20a%20pine%20forest%20-%20increase%20fantasy%20and%20forest.png?raw=true)

Left image prompt: `a fantasy landscape with a pine forest`  
Towards the right: `-fog` 
![Demo](https://github.com/bloc97/CrossAttentionControl/blob/main/images/a%20fantasy%20landscape%20with%20a%20pine%20forest%20-%20decrease%20fog.png?raw=true)

Left image: from previous example  
Towards the right: `-rocks` 
![Demo](https://github.com/bloc97/CrossAttentionControl/blob/main/images/a%20fantasy%20landscape%20with%20a%20pine%20forest%20-%20decrease%20rocks.png?raw=true)
