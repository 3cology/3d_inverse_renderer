# 3D simulator for computer vision and inverse rendering AI

Provided a 3D model and a set of N camera views, system outputs:
![](https://github.com/3cology/3D_computer_vision_simulator/blob/master/media/3d_computer_vision_simulator.png)

## Motivation
This is a 3D graphics simulator for generating per-view, per-pixel 3D model ground truth outputs.  It can support AI and 3D computer vision researchers developing inverse rendering systems, for shaping a photorealistic 3D metaverse.  

#### Let there be renders: example code
The simulator is based in simulator.py, with a simple API for creating renders with ground truth metadata: 

```python
if __name__ == "__main__":  
  # instantiate our 3D computer vision system with a 3D model (compressed glTF)
  iris = Iris(model="models/shoe.glb", resolution=1.0)

  # view a particular 6D perspective 
  iris.view(x=133.037, y=3.604, z=11.048, rotation_x=88.326, rotation_y=0.835, rotation_z=91.31)

  # capture 3D scan data
  iris.scan(exposure_time=3.0, scan_id=0)

  # view another perspective
  iris.view(x=8.352, y=1.878, z=128.7, rotation_x=1.123, rotation_y=0.393, rotation_z=89.881)

  # capture a 2nd 3D scan from another perspective
  iris.scan(exposure_time=3.0, scan_id=1)

```

## Installation
Instructions for installing and developing on the simulator, with optics and photonics modeled after the Iris 3D scanning system by 3co.

#### Install via command line terminal
0. Get this directory on your computer  
   `git clone https://github.com/3cology/inverse_render_simulator.git`  
   `cd inverse_render_simulator`

1. Download Blender LTS Release 2.83.13 [here](https://www.blender.org/download/lts/ "here"). Unzip.

2. Add Blender to command line path ([instructions for Linux, Mac, Windows](https://docs.blender.org/manual/en/2.79/render/workflows/command_line.html "instructions")).  
   For Mac:  
   ```echo "alias blender=/Applications/Blender.app/Contents/MacOS/Blender" >> ~/.bash_profile```  
   For Ubuntu:  
   ```echo "alias blender=/home/3co/blender-2.83.13-linux64/blender" >> ~/.bashrc```  
3. `source ~/.bash_profile` (Mac) or `source ~/.bashrc` (Linux)
4. Run Blender command to get path of its Python installation:  
   `blender -b -P check_python_executable_path.py`
5. Copy and paste into terminal the output line that includes "blender_py".  
   For Mac:  
   ```echo "alias blender_py=/Applications/Blender.app/Contents/Resources/2.82/python/bin/python3.7m" >> ~/.bash_profile```  
   For Ubuntu:   
   ```echo "alias blender_py=/home/3co/blender-2.83.13-linux64/2.83/python/bin/python3.7m" >> ~/.bashrc```  
6. `source ~/.bash_profile` (Mac) or `source ~/.bashrc` (Linux)
7. Prepare to install new modules into this Python:  
   ```blender_py -m ensurepip```  
   ```blender_py -m pip install --upgrade pip setuptools wheel```
8. Here's how to install any missing modules, including these that will be needed:  
   ```blender_py -m pip install Pillow```  
   ```blender_py -m pip install opencv-python```  
   ```blender_py -m pip install imageio```  

#### Launch on CPU or GPU via command line
Decide whether to run the code on gpu or cpu by setting `device=` to either.  
To run with CPU only:
  `blender --python simulator.py -- device=cpu`

To run with GPU:
  `blender --python simulator.py -- device=gpu`

#### Launch with a single texture over entire 3D model  
If you want to add a render configuration for the Principled BSDF shader, to override all textures:
`blender --python simulator.py -- device=cpu render_config=render_config.json`

This `render_config.json` file should use the names that blender uses. For example:
```
{
"Base Color" : [0.23, 0.87, 0.48, 1.0],
"Metallic" : 0.1,
"Subsurface": 0.2,
"Specular": 0.3,
"Roughness": 0.4,
"Specular Tint": 0.5,
"Anisotropic": 0.6,
"Sheen": 0.7,
"Sheen Tint": 0.8,
"Clearcoat": 0.9,
"Clearcoat Roughness" : 1.0
}
```

#### Launch in the cloud 
To use the code in the cloud, this will work nicely:  
  `DISPLAY=:0 blender --python simulator.py -- device=gpu`

This opens a dummy display for the GUI to virtually show up in. If you want to run multiple simulations at the same time, or for some reason a display doesn't work, set up a new one as follows.
  
Make sure you are in the `3D_computer_vision` directory, which is where the dummy display configuration lives.

To set up a display N, run 
`sudo X :N -config dummy-1920x1080.conf`
To run:
`DISPLAY=:N blender --python simulator.py -- device=gpu`
So for example, a new display 1:
`sudo X :1 -config dummy-1920x1080.conf`
`DISPLAY=:1 blender --python simulator.py -- device=gpu`
