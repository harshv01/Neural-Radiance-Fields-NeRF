# Phase 2 - NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

In Phase 2, we utilize a Deep Learning approach to construct a 3D scene from a given set of images. Our implementation is based on the paper 'NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis' by Mildenhall et al.

<div style="text-align:center;">
<img src="./Phase 2/NeRF.gif" alt="Rendered Lego GIF" width="100" height="100">
</div>

## Input
We begin by loading the images and Camera Poses of each image. The camera's Focal Length is calculated using image dimensions and camera angle (FOV). 
The images depict the same object but from different camera angles and positions.

We initiate the scene construction process by generating rays (as ray directions and ray origins) through each pixel. Subsequently, we sample points on those rays. The clipping threshold for the depth values is set to 2 for the Near Threshold and 6 for the far threshold. These thresholds determine the min-max values of the ray (z-axis) between which we sample points for querying the model, which then produces an output. This raw model output is then converted to RGB and $\alpha$ values using the render function.

## Positional Encoding
The reference paper employs Positional Encoding to enhance the resolution of the resulting image, enabling better modeling of higher frequencies. We use 10 frequencies for the positional encoding (L=10) to encode the ray samples before inputting them into the network. Equation \eqref{eqn:positional_encoding} represents the positional encoding function.

$$
\text{PE}(l) = \begin{cases} 
            \sin(2^{l-1} \pi x) & \text{if } l \text{ is odd} \\
            \cos(2^{l-1} \pi x) & \text{if } l \text{ is even}
         \end{cases}
$$


Here, \(l\) represents the frequency index, and \(x\) is the input coordinate.


## How to Run Code

- Download the dataset from the link: https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi?usp=drive_link
- Place it in the directory with the code files and rename the folder to 'Dataset'. The final directory should look like: 
'./Dataset/('lego' or 'ship')/(test, train, val, ...)'
- To train the model, go to the bottom of the Wrapper.py file and uncomment the main() function call and comment the test() function call and run the file.
- To train the model without positional encoding, set the value of num_encoding_fucntions variable to 0, and in the run_model_once() function, instead of inputting the encoded points directly input the flattened_query_points variable into the get_minibatches() function so that those batches can be input into the model.
- Similarly to test the model, comment out the main() fucntion call and uncomment the test() function call and run the Wrapper.py file.
- Directories will automatically be created to store the results and novel views, etc.

## References
- Mildenhall, B., et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis."
